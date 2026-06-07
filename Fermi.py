import numpy as np
import time
import math
import numba
import matplotlib.pyplot as plt

from numba import cuda
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_uniform_float64,
    xoroshiro128p_normal_float64
)


# ==============================================================================
# PART 1: 基础 CUDA 核函数：免疫记忆、收益、Fermi 更新
# ==============================================================================

@cuda.jit
def calc_instant_reputations_kernel(I_grid, R_grid, inst_U,
                                    omega_I, omega_R, grid_size):
    r, c = cuda.grid(2)

    if r < grid_size and c < grid_size:
        inst_U[r, c] = omega_I * I_grid[r, c] + omega_R * R_grid[r, c]


@cuda.jit
def update_U_kernel(U_grid, inst_U, alpha_U, max_U, grid_size):
    r, c = cuda.grid(2)

    if r < grid_size and c < grid_size:
        val = (1.0 - alpha_U) * U_grid[r, c] + alpha_U * inst_U[r, c]

        if val < 0.0:
            val = 0.0
        if val > max_U:
            val = max_U

        U_grid[r, c] = val


@cuda.jit
def calc_payoffs_kernel(I_grid, R_grid, U_grid, payoffs,
                        grid_size, r_param,
                        reward_cost_factor, reward_mult,
                        beta, max_U):
    r, c = cuda.grid(2)

    if r >= grid_size or c >= grid_size:
        return

    offsets_r = cuda.local.array(5, numba.int32)
    offsets_c = cuda.local.array(5, numba.int32)

    offsets_r[0] = 0
    offsets_c[0] = 0

    offsets_r[1] = -1
    offsets_c[1] = 0

    offsets_r[2] = 1
    offsets_c[2] = 0

    offsets_r[3] = 0
    offsets_c[3] = -1

    offsets_r[4] = 0
    offsets_c[4] = 1

    total_payoff = 0.0

    # 个体 (r,c) 参与以自己和四个邻居为中心的 5 个公共物品博弈
    for i in range(5):
        dr = offsets_r[i]
        dc = offsets_c[i]

        center_r = (r - dr + grid_size) % grid_size
        center_c = (c - dc + grid_size) % grid_size

        group_I_sum = 0.0
        group_R_sum = 0.0
        group_weighted_I_sum = 0.0

        # 计算该组的投资总和、奖励意愿总和、记忆加权投资总和
        for j in range(5):
            member_r = (center_r + offsets_r[j] + grid_size) % grid_size
            member_c = (center_c + offsets_c[j] + grid_size) % grid_size

            I_j = I_grid[member_r, member_c]
            R_j = R_grid[member_r, member_c]
            U_j = U_grid[member_r, member_c]

            U_j_norm = U_j / max_U if max_U > 0.0 else 0.0

            group_I_sum += I_j
            group_R_sum += R_j
            group_weighted_I_sum += I_j * (1.0 + beta * U_j_norm)

        group_avg_I = group_I_sum / 5.0

        focal_I = I_grid[r, c]
        focal_R = R_grid[r, c]
        focal_U = U_grid[r, c]
        focal_U_norm = focal_U / max_U if max_U > 0.0 else 0.0

        benefit = r_param * group_avg_I
        cost = focal_I
        reward_cost = reward_cost_factor * focal_R

        reward_pool = reward_mult * group_avg_I * group_R_sum

        reward_received = 0.0
        if group_I_sum > 0.0 and group_weighted_I_sum > 0.0:
            focal_weighted_I = focal_I * (1.0 + beta * focal_U_norm)
            reward_received = reward_pool * (focal_weighted_I / group_weighted_I_sum)

        total_payoff += benefit - cost - reward_cost + reward_received

    payoffs[r, c] = total_payoff


@cuda.jit
def fermi_sweep_kernel(rng_states,
                       I_grid, R_grid, payoffs,
                       new_I, new_R,
                       grid_size, K_fermi,
                       mutation_sigma):
    r, c = cuda.grid(2)

    if r >= grid_size or c >= grid_size:
        return

    offsets_r = cuda.local.array(4, numba.int32)
    offsets_c = cuda.local.array(4, numba.int32)

    offsets_r[0] = -1
    offsets_c[0] = 0

    offsets_r[1] = 1
    offsets_c[1] = 0

    offsets_r[2] = 0
    offsets_c[2] = -1

    offsets_r[3] = 0
    offsets_c[3] = 1

    flat_idx = r * grid_size + c

    # 随机选择一个 von Neumann 邻居
    u = xoroshiro128p_uniform_float64(rng_states, flat_idx)
    nb_idx = int(u * 4.0)
    if nb_idx >= 4:
        nb_idx = 3

    nr = (r + offsets_r[nb_idx] + grid_size) % grid_size
    nc = (c + offsets_c[nb_idx] + grid_size) % grid_size

    p_self = payoffs[r, c]
    p_nb = payoffs[nr, nc]

    # Fermi adoption probability:
    # W = 1 / (1 + exp((P_self - P_neighbor) / K))
    if K_fermi <= 0.0:
        if p_nb > p_self:
            prob = 1.0
        else:
            prob = 0.0
    else:
        exponent = (p_self - p_nb) / K_fermi

        if exponent > 20.0:
            prob = 0.0
        elif exponent < -20.0:
            prob = 1.0
        else:
            prob = 1.0 / (1.0 + math.exp(exponent))

    u2 = xoroshiro128p_uniform_float64(rng_states, flat_idx)

    if u2 < prob:
        child_I = I_grid[nr, nc]
        child_R = R_grid[nr, nc]
    else:
        child_I = I_grid[r, c]
        child_R = R_grid[r, c]

    # 默认 mutation_sigma = 0.0，即传统 Fermi 复制更新
    # 如果想让连续策略有微小扰动，可以设为 0.001 或更小
    if mutation_sigma > 0.0:
        child_I += xoroshiro128p_normal_float64(rng_states, flat_idx) * mutation_sigma
        child_R += xoroshiro128p_normal_float64(rng_states, flat_idx) * mutation_sigma

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
# PART 2: Fermi baseline GPU 类
# ==============================================================================

class SPGG_FermiBaseline_GPU:
    def __init__(self,
                 r=3.5,
                 grid_size=100,
                 max_iterations=10000,
                 K_fermi=0.1,
                 mutation_sigma=0.0,
                 reward_cost_factor=0.3,
                 reward_multiplier=0.7,
                 beta=0.5,
                 omega_I=1.0,
                 omega_R=0.5,
                 alpha_U=0.5,
                 seed=None):

        self.r = r
        self.grid_size = grid_size
        self.max_iterations = max_iterations

        self.K_fermi = K_fermi
        self.mutation_sigma = mutation_sigma

        self.reward_cost_factor = reward_cost_factor
        self.reward_multiplier = reward_multiplier
        self.beta = beta

        self.omega_I = omega_I
        self.omega_R = omega_R
        self.alpha_U = alpha_U
        self.max_U = self.omega_I + self.omega_R

        self.threadsperblock = (8, 8)

        blocks_x = math.ceil(grid_size / self.threadsperblock[0])
        blocks_y = math.ceil(grid_size / self.threadsperblock[1])

        self.blockspergrid = (blocks_x, blocks_y)

        seed_val = int(time.time()) if seed is None else seed
        np.random.seed(seed_val)

        I_host = np.random.rand(grid_size, grid_size)
        R_host = np.random.rand(grid_size, grid_size)
        U_host = np.full((grid_size, grid_size), self.max_U / 2.0)

        self.d_I = cuda.to_device(I_host)
        self.d_R = cuda.to_device(R_host)
        self.d_U = cuda.to_device(U_host)

        self.d_inst_U = cuda.device_array((grid_size, grid_size), dtype=np.float64)
        self.d_P = cuda.device_array((grid_size, grid_size), dtype=np.float64)

        self.d_new_I = cuda.device_array((grid_size, grid_size), dtype=np.float64)
        self.d_new_R = cuda.device_array((grid_size, grid_size), dtype=np.float64)

        self.rng_states = create_xoroshiro128p_states(
            grid_size * grid_size,
            seed=seed_val
        )

    def step(self):
        # 1. 计算瞬时免疫记忆 U_inst
        calc_instant_reputations_kernel[self.blockspergrid, self.threadsperblock](
            self.d_I,
            self.d_R,
            self.d_inst_U,
            self.omega_I,
            self.omega_R,
            self.grid_size
        )

        # 2. 更新累积免疫记忆 U
        update_U_kernel[self.blockspergrid, self.threadsperblock](
            self.d_U,
            self.d_inst_U,
            self.alpha_U,
            self.max_U,
            self.grid_size
        )

        # 3. 计算收益
        calc_payoffs_kernel[self.blockspergrid, self.threadsperblock](
            self.d_I,
            self.d_R,
            self.d_U,
            self.d_P,
            self.grid_size,
            self.r,
            self.reward_cost_factor,
            self.reward_multiplier,
            self.beta,
            self.max_U
        )

        # 4. Fermi 同步更新策略
        fermi_sweep_kernel[self.blockspergrid, self.threadsperblock](
            self.rng_states,
            self.d_I,
            self.d_R,
            self.d_P,
            self.d_new_I,
            self.d_new_R,
            self.grid_size,
            self.K_fermi,
            self.mutation_sigma
        )

        # 5. 交换指针，实现同步更新
        self.d_I, self.d_new_I = self.d_new_I, self.d_I
        self.d_R, self.d_new_R = self.d_new_R, self.d_R

    def run_simulation(self, record_every=1, steady_window=50):
        iterations = []
        avg_I_history = []
        avg_R_history = []
        avg_U_history = []

        steady_I = 0.0
        steady_R = 0.0
        steady_count = 0

        for iteration in range(self.max_iterations):
            self.step()

            if record_every is not None and record_every > 0:
                if iteration % record_every == 0 or iteration == self.max_iterations - 1:
                    h_I = self.d_I.copy_to_host()
                    h_R = self.d_R.copy_to_host()
                    h_U = self.d_U.copy_to_host()

                    iterations.append(iteration + 1)
                    avg_I_history.append(np.mean(h_I))
                    avg_R_history.append(np.mean(h_R))
                    avg_U_history.append(np.mean(h_U))

            if iteration >= self.max_iterations - steady_window:
                h_I = self.d_I.copy_to_host()
                h_R = self.d_R.copy_to_host()

                steady_I += np.mean(h_I)
                steady_R += np.mean(h_R)
                steady_count += 1

        return {
            "iterations": np.array(iterations),
            "avg_I": np.array(avg_I_history),
            "avg_R": np.array(avg_R_history),
            "avg_U": np.array(avg_U_history),
            "steady_I": steady_I / steady_count,
            "steady_R": steady_R / steady_count
        }

    def get_current_grids(self):
        return {
            "I": self.d_I.copy_to_host(),
            "R": self.d_R.copy_to_host(),
            "U": self.d_U.copy_to_host(),
            "P": self.d_P.copy_to_host()
        }


# ==============================================================================
# PART 3: 多次独立运行函数
# ==============================================================================

def run_fermi_multiple_runs(num_runs=10,
                            r=3.5,
                            grid_size=100,
                            max_iterations=10000,
                            K_fermi=0.1,
                            mutation_sigma=0.0,
                            reward_cost_factor=0.3,
                            reward_multiplier=0.7,
                            beta=0.5,
                            omega_I=1.0,
                            omega_R=0.5,
                            alpha_U=0.5,
                            record_every=1,
                            steady_window=50,
                            base_seed=2026):
    all_avg_I = []
    all_avg_R = []
    all_avg_U = []
    steady_I_list = []
    steady_R_list = []
    iterations = None

    for run in range(num_runs):
        print(f"Running Fermi baseline: run {run + 1}/{num_runs}")

        model = SPGG_FermiBaseline_GPU(
            r=r,
            grid_size=grid_size,
            max_iterations=max_iterations,
            K_fermi=K_fermi,
            mutation_sigma=mutation_sigma,
            reward_cost_factor=reward_cost_factor,
            reward_multiplier=reward_multiplier,
            beta=beta,
            omega_I=omega_I,
            omega_R=omega_R,
            alpha_U=alpha_U,
            seed=base_seed + run
        )

        result = model.run_simulation(
            record_every=record_every,
            steady_window=steady_window
        )

        if iterations is None:
            iterations = result["iterations"]

        all_avg_I.append(result["avg_I"])
        all_avg_R.append(result["avg_R"])
        all_avg_U.append(result["avg_U"])

        steady_I_list.append(result["steady_I"])
        steady_R_list.append(result["steady_R"])

    all_avg_I = np.array(all_avg_I)
    all_avg_R = np.array(all_avg_R)
    all_avg_U = np.array(all_avg_U)

    return {
        "iterations": iterations,

        "avg_I_mean": np.mean(all_avg_I, axis=0),
        "avg_I_std": np.std(all_avg_I, axis=0),

        "avg_R_mean": np.mean(all_avg_R, axis=0),
        "avg_R_std": np.std(all_avg_R, axis=0),

        "avg_U_mean": np.mean(all_avg_U, axis=0),
        "avg_U_std": np.std(all_avg_U, axis=0),

        "steady_I_mean": np.mean(steady_I_list),
        "steady_I_std": np.std(steady_I_list),

        "steady_R_mean": np.mean(steady_R_list),
        "steady_R_std": np.std(steady_R_list),

        "steady_I_each_run": np.array(steady_I_list),
        "steady_R_each_run": np.array(steady_R_list)
    }


# ==============================================================================
# PART 4: 示例运行与绘图
# ==============================================================================

if __name__ == "__main__":
    result = run_fermi_multiple_runs(
        num_runs=10,
        r=3.5,
        grid_size=100,
        max_iterations=10000,
        K_fermi=0.1,
        mutation_sigma=0.0,

        # 与你的 AIS 模型保持一致的收益和奖励参数
        reward_cost_factor=0.3,
        reward_multiplier=0.7,
        beta=0.5,
        omega_I=1.0,
        omega_R=0.5,
        alpha_U=0.5,

        record_every=1,
        steady_window=50,
        base_seed=2026
    )

    print("Fermi steady Avg I:",
          result["steady_I_mean"],
          "±",
          result["steady_I_std"])

    print("Fermi steady Avg R:",
          result["steady_R_mean"],
          "±",
          result["steady_R_std"])

    iterations = result["iterations"]

    avg_I_mean = result["avg_I_mean"]
    avg_I_std = result["avg_I_std"]

    avg_R_mean = result["avg_R_mean"]
    avg_R_std = result["avg_R_std"]

    plt.figure(figsize=(7, 5))

    plt.semilogx(iterations, avg_I_mean,
                 label="Fermi baseline: Avg I",
                 linewidth=2)

    plt.fill_between(iterations,
                     avg_I_mean - avg_I_std,
                     avg_I_mean + avg_I_std,
                     alpha=0.2)

    plt.semilogx(iterations, avg_R_mean,
                 label="Fermi baseline: Avg R",
                 linewidth=2,
                 linestyle="--")

    plt.fill_between(iterations,
                     avg_R_mean - avg_R_std,
                     avg_R_mean + avg_R_std,
                     alpha=0.15)

    plt.xlabel("Iteration")
    plt.ylabel("Average value")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()