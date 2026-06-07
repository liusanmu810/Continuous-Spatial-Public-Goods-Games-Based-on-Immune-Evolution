import numpy as np
import time
import math
import numba
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float64, xoroshiro128p_uniform_float64


# ==============================================================================
# PART 1: CUDA GPU 核函数 (物理引擎和免疫引擎)
# ==============================================================================

@cuda.jit
def calc_instant_reputations_kernel(I_grid, R_grid, inst_U, omega_I, omega_R, grid_size):
    r, c = cuda.grid(2)
    if r < grid_size and c < grid_size:
        inst_U[r, c] = omega_I * I_grid[r, c] + omega_R * R_grid[r, c]


@cuda.jit
def update_U_kernel(U_grid, inst_U, alpha_U, max_U, grid_size):
    r, c = cuda.grid(2)
    if r < grid_size and c < grid_size:
        val = (1.0 - alpha_U) * U_grid[r, c] + alpha_U * inst_U[r, c]
        if val < 0.0: val = 0.0
        if val > max_U: val = max_U
        U_grid[r, c] = val


@cuda.jit
def calc_payoffs_kernel(I_grid, R_grid, U_grid, payoffs, grid_size, r_param, reward_cost_factor, reward_mult, beta,
                        max_U):
    r, c = cuda.grid(2)
    if r >= grid_size or c >= grid_size:
        return

    # Von Neumann 邻域偏移
    offsets_r = cuda.local.array(5, numba.int32)
    offsets_c = cuda.local.array(5, numba.int32)
    offsets_r[0] = 0;
    offsets_c[0] = 0
    offsets_r[1] = -1;
    offsets_c[1] = 0
    offsets_r[2] = 1;
    offsets_c[2] = 0
    offsets_r[3] = 0;
    offsets_c[3] = -1
    offsets_r[4] = 0;
    offsets_c[4] = 1

    total_payoff = 0.0

    for i in range(5):
        dr = offsets_r[i]
        dc = offsets_c[i]
        center_r = (r - dr + grid_size) % grid_size
        center_c = (c - dc + grid_size) % grid_size

        group_I_sum = 0.0
        group_R_sum = 0.0
        group_weighted_I_sum = 0.0

        for j in range(5):
            member_r = (center_r + offsets_r[j] + grid_size) % grid_size
            member_c = (center_c + offsets_c[j] + grid_size) % grid_size

            I_j = I_grid[member_r, member_c]
            R_j = R_grid[member_r, member_c]
            U_j = U_grid[member_r, member_c]
            U_j_norm = U_j / max_U if max_U > 0 else 0.0

            group_I_sum += I_j
            group_R_sum += R_j
            group_weighted_I_sum += I_j * (1.0 + beta * U_j_norm)

        group_avg_I = group_I_sum / 5.0
        focal_I = I_grid[r, c]
        focal_R = R_grid[r, c]
        focal_U = U_grid[r, c]
        focal_U_norm = focal_U / max_U if max_U > 0 else 0.0

        benefit = r_param * group_avg_I
        cost = focal_I
        reward_cost = reward_cost_factor * focal_R

        reward_pool = reward_mult * group_avg_I * group_R_sum
        reward_received = 0.0

        if group_I_sum > 0 and group_weighted_I_sum > 0:
            focal_weighted_I = focal_I * (1.0 + beta * focal_U_norm)
            reward_received = reward_pool * (focal_weighted_I / group_weighted_I_sum)

        total_payoff += (benefit - cost - reward_cost + reward_received)

    payoffs[r, c] = total_payoff


@cuda.jit
def calc_concentrations_kernel(I_grid, R_grid, concentrations, grid_size, delta):
    r, c = cuda.grid(2)
    if r >= grid_size or c >= grid_size:
        return

    offsets_r = cuda.local.array(4, numba.int32)
    offsets_c = cuda.local.array(4, numba.int32)
    offsets_r[0] = -1;
    offsets_c[0] = 0
    offsets_r[1] = 1;
    offsets_c[1] = 0
    offsets_r[2] = 0;
    offsets_c[2] = -1
    offsets_r[3] = 0;
    offsets_c[3] = 1

    focal_I = I_grid[r, c]
    focal_R = R_grid[r, c]
    sum_sim = 0.0

    for k in range(4):
        nr = (r + offsets_r[k] + grid_size) % grid_size
        nc = (c + offsets_c[k] + grid_size) % grid_size
        diff_I = focal_I - I_grid[nr, nc]
        diff_R = focal_R - R_grid[nr, nc]
        dist_sq = diff_I * diff_I + diff_R * diff_R

        if delta == 0.0:
            sum_sim += 1.0 if dist_sq == 0.0 else 0.0
        else:
            sum_sim += math.exp(-dist_sq / delta)

    concentrations[r, c] = sum_sim / 4.0


@cuda.jit
def immune_sweep_kernel(rng_states, I_grid, R_grid, payoffs, reputations, concentrations,
                        new_I, new_R, grid_size, lambda_param, alpha_conc, gamma_param,
                        sigma_min, sigma_max, global_avg_U, K_P, K_U):
    r, c = cuda.grid(2)
    if r >= grid_size or c >= grid_size:
        return

    offsets_r = cuda.local.array(5, numba.int32)
    offsets_c = cuda.local.array(5, numba.int32)
    offsets_r[0] = 0;
    offsets_c[0] = 0
    offsets_r[1] = -1;
    offsets_c[1] = 0
    offsets_r[2] = 1;
    offsets_c[2] = 0
    offsets_r[3] = 0;
    offsets_c[3] = -1
    offsets_r[4] = 0;
    offsets_c[4] = 1

    raw_payoffs = cuda.local.array(5, numba.float64)
    raw_reps = cuda.local.array(5, numba.float64)
    n_concs = cuda.local.array(5, numba.float64)
    raw_affs = cuda.local.array(5, numba.float64)
    stims = cuda.local.array(5, numba.float64)

    sum_p = 0.0
    for i in range(5):
        nr = (r + offsets_r[i] + grid_size) % grid_size
        nc = (c + offsets_c[i] + grid_size) % grid_size
        p_val = payoffs[nr, nc]
        raw_payoffs[i] = p_val
        raw_reps[i] = reputations[nr, nc]
        n_concs[i] = concentrations[nr, nc]
        sum_p += p_val

    local_avg_P = sum_p / 5.0

    for i in range(5):
        exponent_P = -(raw_payoffs[i] - local_avg_P) / K_P
        if exponent_P > 20.0:
            p_norm = 0.0
        elif exponent_P < -20.0:
            p_norm = 1.0
        else:
            p_norm = 1.0 / (1.0 + math.exp(exponent_P))

        exponent_U = -(raw_reps[i] - global_avg_U) / K_U
        if exponent_U > 20.0:
            u_norm = 0.0
        elif exponent_U < -20.0:
            u_norm = 1.0
        else:
            u_norm = 1.0 / (1.0 + math.exp(exponent_U))

        n_affinity = lambda_param * p_norm + (1.0 - lambda_param) * u_norm
        if n_affinity < 0: n_affinity = 0.0
        if n_affinity > 1.0: n_affinity = 1.0
        raw_affs[i] = n_affinity

        gated_penalty = alpha_conc * n_concs[i] * (1.0 - u_norm)
        stims[i] = n_affinity * math.exp(-gated_penalty)

    best_idx = -1
    max_stim = -1.0
    for i in range(5):
        if stims[i] > max_stim:
            max_stim = stims[i]
            best_idx = i

    flat_idx = r * grid_size + c

    if best_idx == -1:
        u = xoroshiro128p_uniform_float64(rng_states, flat_idx)  # 加上 _float64
        parent_idx = int(u * 5)
        if parent_idx == 5: parent_idx = 4
    else:
        parent_idx = best_idx

    parent_r = (r + offsets_r[parent_idx] + grid_size) % grid_size
    parent_c = (c + offsets_c[parent_idx] + grid_size) % grid_size
    parent_affinity = raw_affs[parent_idx]

    sigma = sigma_min + (sigma_max - sigma_min) * math.exp(-gamma_param * parent_affinity)
    mut_I = xoroshiro128p_normal_float64(rng_states, flat_idx) * sigma  # 加上 _float64
    mut_R = xoroshiro128p_normal_float64(rng_states, flat_idx) * sigma  # 加上 _float64

    child_I = I_grid[parent_r, parent_c] + mut_I
    child_R = R_grid[parent_r, parent_c] + mut_R

    if child_I < 0.0:
        child_I = 0.0
    elif child_I > 1.0:
        child_I = 1.0

    if child_R < 0.0:
        child_R = 0.0
    elif child_R > 1.0:
        child_R = 1.0

    new_I[r, c] = child_I
    new_R[r, c] = child_R


# ==============================================================================
# PART 2: 控制器类 (GPU 环境调度)
# ==============================================================================

class SPGG_ImmuneSystem_GPU:
    def __init__(self, r=3.5, grid_size=100, max_iterations=1000,
                 delta=1, alpha_conc=3, sigma_min=0.001, sigma_max=0.03, gamma=1,
                 reward_cost_factor=0.3, reward_multiplier=0.7,
                 beta=0.5, omega_I=1.0, omega_R=0.5, alpha_U=0.5,
                 lambda_param=0.9, K_P=0.2, K_U=0.2, seed=None):

        self.r = r
        self.grid_size = grid_size
        self.max_iterations = max_iterations

        self.delta = delta
        self.alpha_conc = alpha_conc
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.gamma = gamma

        self.reward_cost_factor = reward_cost_factor
        self.reward_multiplier = reward_multiplier
        self.beta = beta
        self.omega_I = omega_I
        self.omega_R = omega_R
        self.alpha_U = alpha_U
        self.max_U = self.omega_I * 1.0 + self.omega_R * 1.0

        self.lambda_param = lambda_param
        self.K_P = K_P
        self.K_U = K_U

        # CUDA 线程块配置
        self.threadsperblock = (8, 8)
        blocks_x = math.ceil(grid_size / self.threadsperblock[0])
        blocks_y = math.ceil(grid_size / self.threadsperblock[1])
        self.blockspergrid = (blocks_x, blocks_y)

        # 初始化 CPU 数组
        np.random.seed(int(time.time()) if seed is None else seed)
        I_host = np.random.rand(grid_size, grid_size)
        R_host = np.random.rand(grid_size, grid_size)
        U_host = np.full((grid_size, grid_size), self.max_U / 2.0)

        # 拷贝至显存 (GPU Arrays)
        self.d_I = cuda.to_device(I_host)
        self.d_R = cuda.to_device(R_host)
        self.d_U = cuda.to_device(U_host)
        self.d_P = cuda.device_array((grid_size, grid_size), dtype=np.float64)
        self.d_inst_U = cuda.device_array((grid_size, grid_size), dtype=np.float64)
        self.d_C = cuda.device_array((grid_size, grid_size), dtype=np.float64)
        self.d_new_I = cuda.device_array((grid_size, grid_size), dtype=np.float64)
        self.d_new_R = cuda.device_array((grid_size, grid_size), dtype=np.float64)

        # Numba CUDA 并发随机状态
        self.rng_states = create_xoroshiro128p_states(grid_size * grid_size, seed=(seed if seed else int(time.time())))

    def run_simulation(self):
        steady_I = 0.0
        steady_R = 0.0

        for iteration in range(self.max_iterations):
            # 1. 更新瞬时声誉及累积声誉
            calc_instant_reputations_kernel[self.blockspergrid, self.threadsperblock](
                self.d_I, self.d_R, self.d_inst_U, self.omega_I, self.omega_R, self.grid_size
            )
            update_U_kernel[self.blockspergrid, self.threadsperblock](
                self.d_U, self.d_inst_U, self.alpha_U, self.max_U, self.grid_size
            )

            # 2. 计算网格总体收益
            calc_payoffs_kernel[self.blockspergrid, self.threadsperblock](
                self.d_I, self.d_R, self.d_U, self.d_P,
                self.grid_size, self.r, self.reward_cost_factor, self.reward_multiplier,
                self.beta, self.max_U
            )

            # 将收益与声誉提取到 CPU 计算全局平均
            h_P = self.d_P.copy_to_host()
            h_U = self.d_U.copy_to_host()
            global_avg_P = np.mean(h_P)
            global_avg_U = np.mean(h_U)

            # 3. 免疫浓度计算
            calc_concentrations_kernel[self.blockspergrid, self.threadsperblock](
                self.d_I, self.d_R, self.d_C, self.grid_size, self.delta
            )

            # 4. 核心免疫扫描 (克隆/成熟变异)
            immune_sweep_kernel[self.blockspergrid, self.threadsperblock](
                self.rng_states, self.d_I, self.d_R, self.d_P, self.d_U, self.d_C,
                self.d_new_I, self.d_new_R, self.grid_size,
                self.lambda_param, self.alpha_conc, self.gamma,
                self.sigma_min, self.sigma_max, global_avg_U, self.K_P, self.K_U
            )

            # GPU 指针地址互换
            self.d_I, self.d_new_I = self.d_new_I, self.d_I
            self.d_R, self.d_new_R = self.d_new_R, self.d_R


            if iteration >= self.max_iterations - 50:
                h_I = self.d_I.copy_to_host()
                h_R = self.d_R.copy_to_host()  # 提取 R 矩阵到内存
                steady_I += np.mean(h_I)
                steady_R += np.mean(h_R)       # 累加 R 的均值


        return steady_I / 50.0, steady_R / 50.0