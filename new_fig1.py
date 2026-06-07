import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math
import numba
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float64

from AIS import (
    SPGG_ImmuneSystem_GPU,
    calc_instant_reputations_kernel,
    update_U_kernel,
    calc_payoffs_kernel,
    calc_concentrations_kernel,
    immune_sweep_kernel
)

# ==============================================================================
# Figure 5: Time evolution of average investment
# Fermi baseline + AIS mechanism comparison
# ==============================================================================

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# PART 1: Fermi baseline kernel
# ==============================================================================

@cuda.jit
def fermi_sweep_kernel(rng_states,
                       I_grid, R_grid, payoffs,
                       new_I, new_R,
                       grid_size, K_fermi):
    """
    Fermi pairwise comparison update.

    焦点个体 i 随机选择一个 von Neumann 邻居 j，
    以 Fermi 概率复制邻居的完整二维策略 (I_j, R_j)。

    W_{i<-j} = 1 / (1 + exp((P_i - P_j) / K))
    """

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

    # 随机选择一个邻居
    u = xoroshiro128p_uniform_float64(rng_states, flat_idx)
    nb_idx = int(u * 4.0)

    if nb_idx >= 4:
        nb_idx = 3

    nr = (r + offsets_r[nb_idx] + grid_size) % grid_size
    nc = (c + offsets_c[nb_idx] + grid_size) % grid_size

    p_self = payoffs[r, c]
    p_nb = payoffs[nr, nc]

    # Fermi adoption probability
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

    new_I[r, c] = child_I
    new_R[r, c] = child_R


# ==============================================================================
# PART 2: 构造记录时间点
# ==============================================================================

def build_log_record_times(max_iter):
    """
    构造适合 symlog 横坐标的记录时间点。

    - t=0 在 run_one_time_series() 中单独记录；
    - t=1~100 每代记录；
    - t=110~1000 每 10 代记录；
    - t=1100~10000 每 100 代记录。
    """

    record_times = sorted(set(
        list(range(1, min(100, max_iter) + 1, 1)) +
        list(range(110, min(1000, max_iter) + 1, 10)) +
        list(range(1100, max_iter + 1, 100)) +
        [max_iter]
    ))

    return set(record_times)


# ==============================================================================
# PART 3: 单次 AIS 时间序列
# ==============================================================================

def run_one_ais_time_series(params, seed_val, max_iter=10000):
    """
    单次 AIS 模型模拟，返回平均投资水平 I_bar(t)。
    """

    sim = SPGG_ImmuneSystem_GPU(**params, seed=seed_val)

    times = []
    avg_I_series = []

    record_times = build_log_record_times(max_iter)

    # 记录初始状态 t=0
    h_I0 = sim.d_I.copy_to_host()
    times.append(0)
    avg_I_series.append(np.mean(h_I0))

    for iteration in range(1, max_iter + 1):

        # 1. 更新瞬时免疫记忆
        calc_instant_reputations_kernel[sim.blockspergrid, sim.threadsperblock](
            sim.d_I, sim.d_R, sim.d_inst_U,
            sim.omega_I, sim.omega_R, sim.grid_size
        )

        # 2. 更新累积免疫记忆
        update_U_kernel[sim.blockspergrid, sim.threadsperblock](
            sim.d_U, sim.d_inst_U,
            sim.alpha_U, sim.max_U, sim.grid_size
        )

        # 3. 计算收益
        calc_payoffs_kernel[sim.blockspergrid, sim.threadsperblock](
            sim.d_I, sim.d_R, sim.d_U, sim.d_P,
            sim.grid_size, sim.r,
            sim.reward_cost_factor, sim.reward_multiplier,
            sim.beta, sim.max_U
        )

        # 4. 全局平均免疫记忆
        h_U = sim.d_U.copy_to_host()
        global_avg_U = np.mean(h_U)

        # 5. 计算局部浓度
        calc_concentrations_kernel[sim.blockspergrid, sim.threadsperblock](
            sim.d_I, sim.d_R, sim.d_C,
            sim.grid_size, sim.delta
        )

        # 6. AIS 免疫选择与亲和度成熟变异
        immune_sweep_kernel[sim.blockspergrid, sim.threadsperblock](
            sim.rng_states,
            sim.d_I, sim.d_R,
            sim.d_P, sim.d_U, sim.d_C,
            sim.d_new_I, sim.d_new_R,
            sim.grid_size,
            sim.lambda_param,
            sim.alpha_conc,
            sim.gamma,
            sim.sigma_min,
            sim.sigma_max,
            global_avg_U,
            sim.K_P,
            sim.K_U
        )

        # 7. 同步更新
        sim.d_I, sim.d_new_I = sim.d_new_I, sim.d_I
        sim.d_R, sim.d_new_R = sim.d_new_R, sim.d_R

        # 8. 记录平均投资
        if iteration in record_times:
            h_I = sim.d_I.copy_to_host()
            times.append(iteration)
            avg_I_series.append(np.mean(h_I))

    return np.array(times), np.array(avg_I_series)


# ==============================================================================
# PART 4: 单次 Fermi baseline 时间序列
# ==============================================================================

def run_one_fermi_time_series(params, seed_val, max_iter=10000, K_fermi=0.1):
    """
    单次 Fermi baseline 模拟。

    注意：
    - 收益函数、奖励机制、免疫记忆 U 更新与 AIS 模型保持一致；
    - 只把策略更新规则替换为 Fermi pairwise comparison；
    - Fermi 更新只根据收益差异决定是否复制邻居策略。
    """

    sim = SPGG_ImmuneSystem_GPU(**params, seed=seed_val)

    times = []
    avg_I_series = []

    record_times = build_log_record_times(max_iter)

    # 记录初始状态 t=0
    h_I0 = sim.d_I.copy_to_host()
    times.append(0)
    avg_I_series.append(np.mean(h_I0))

    for iteration in range(1, max_iter + 1):

        # 1. 更新瞬时免疫记忆
        calc_instant_reputations_kernel[sim.blockspergrid, sim.threadsperblock](
            sim.d_I, sim.d_R, sim.d_inst_U,
            sim.omega_I, sim.omega_R, sim.grid_size
        )

        # 2. 更新累积免疫记忆
        update_U_kernel[sim.blockspergrid, sim.threadsperblock](
            sim.d_U, sim.d_inst_U,
            sim.alpha_U, sim.max_U, sim.grid_size
        )

        # 3. 计算收益
        calc_payoffs_kernel[sim.blockspergrid, sim.threadsperblock](
            sim.d_I, sim.d_R, sim.d_U, sim.d_P,
            sim.grid_size, sim.r,
            sim.reward_cost_factor, sim.reward_multiplier,
            sim.beta, sim.max_U
        )

        # 4. Fermi 更新
        fermi_sweep_kernel[sim.blockspergrid, sim.threadsperblock](
            sim.rng_states,
            sim.d_I, sim.d_R,
            sim.d_P,
            sim.d_new_I, sim.d_new_R,
            sim.grid_size,
            K_fermi
        )

        # 5. 同步更新
        sim.d_I, sim.d_new_I = sim.d_new_I, sim.d_I
        sim.d_R, sim.d_new_R = sim.d_new_R, sim.d_R

        # 6. 记录平均投资
        if iteration in record_times:
            h_I = sim.d_I.copy_to_host()
            times.append(iteration)
            avg_I_series.append(np.mean(h_I))

    return np.array(times), np.array(avg_I_series)


# ==============================================================================
# PART 5: 主实验函数
# ==============================================================================

def run_mechanism_time_evolution_experiment():
    """
    Fermi baseline 与三种 AIS 机制配置下的平均投资水平时间演化对照实验。
    """

    # ==========================================================================
    # 1. 基础参数设置
    # ==========================================================================

    L = 100
    max_iter = 10000
    num_runs = 5
    base_seed = 42

    test_r = 3.5
    K_fermi = 0.1

    output_folder = "results/fig5_fermi_ais_mechanism_comparison"
    os.makedirs(output_folder, exist_ok=True)

    # ==========================================================================
    # 2. 对照场景
    # ==========================================================================

    scenarios = [
        {
            "type": "fermi",
            "label": "Fermi baseline",
            "short_label": "Fermi",
            "alpha_conc": 0.0,
            "gamma": 0.0,
        },
        {
            "type": "ais",
            "label": r"$(\alpha=0,\gamma=0)$",
            "short_label": "NoSupp_NoMat",
            "alpha_conc": 0.0,
            "gamma": 0.0,
        },
        {
            "type": "ais",
            "label": r"$(\alpha=3,\gamma=0)$",
            "short_label": "SuppOnly",
            "alpha_conc": 3.0,
            "gamma": 0.0,
        },
        {
            "type": "ais",
            "label": r"$(\alpha=3,\gamma=1)$",
            "short_label": "FullAIS",
            "alpha_conc": 3.0,
            "gamma": 1.0,
        }
    ]

    # 其他参数与原图保持一致
    base_params = {
        "r": test_r,
        "grid_size": L,
        "max_iterations": max_iter,

        "delta": 1.0,
        "sigma_min": 0.001,
        "sigma_max": 0.03,

        "reward_cost_factor": 0.3,
        "reward_multiplier": 0.7,
        "beta": 0.5,
        "omega_I": 1.0,
        "omega_R": 0.5,
        "alpha_U": 0.5,

        "lambda_param": 0.9,
        "K_P": 0.2,
        "K_U": 0.2,
    }

    # ==========================================================================
    # 3. 多次独立运行
    # ==========================================================================

    all_results = {}

    print("=" * 90)
    print("Start Fig. 5 Fermi baseline + AIS mechanism comparison experiment")
    print(f"L={L}, r={test_r}, max_iter={max_iter}, runs={num_runs}, K_fermi={K_fermi}")
    print("Initial state t=0 will be recorded.")
    print("=" * 90)

    for sc in scenarios:
        label = sc["label"]
        short_label = sc["short_label"]

        print(f"\n>>> Scenario: {label}")

        scenario_series = []
        common_times = None

        for run_idx in range(num_runs):
            seed_val = base_seed + run_idx * 1000

            params = base_params.copy()
            params["alpha_conc"] = sc["alpha_conc"]
            params["gamma"] = sc["gamma"]

            start_time = time.time()

            if sc["type"] == "fermi":
                times, avg_I_series = run_one_fermi_time_series(
                    params=params,
                    seed_val=seed_val,
                    max_iter=max_iter,
                    K_fermi=K_fermi
                )
            else:
                times, avg_I_series = run_one_ais_time_series(
                    params=params,
                    seed_val=seed_val,
                    max_iter=max_iter
                )

            cuda.synchronize()

            if common_times is None:
                common_times = times
            else:
                if not np.array_equal(common_times, times):
                    raise ValueError("Recorded time points are inconsistent across runs.")

            scenario_series.append(avg_I_series)

            print(
                f"    Run {run_idx + 1:02d}/{num_runs} finished. "
                f"Initial Avg I = {avg_I_series[0]:.4f}, "
                f"Final Avg I = {avg_I_series[-1]:.4f}, "
                f"time = {time.time() - start_time:.2f}s"
            )

        scenario_series = np.array(scenario_series)

        mean_series = np.mean(scenario_series, axis=0)
        std_series = np.std(scenario_series, axis=0)

        all_results[short_label] = {
            "type": sc["type"],
            "label": label,
            "times": common_times,
            "all_runs": scenario_series,
            "mean": mean_series,
            "std": std_series,
            "alpha_conc": sc["alpha_conc"],
            "gamma": sc["gamma"],
        }

    # ==========================================================================
    # 4. 保存原始数据
    # ==========================================================================

    npz_path = os.path.join(
        output_folder,
        "Fig5_Fermi_AIS_Mechanism_Time_Evolution_Data.npz"
    )

    save_dict = {
        "test_r": test_r,
        "K_fermi": K_fermi,
        "num_runs": num_runs,
        "max_iter": max_iter,
        "grid_size": L,
    }

    for key, res in all_results.items():
        save_dict[f"{key}_times"] = res["times"]
        save_dict[f"{key}_all_runs"] = res["all_runs"]
        save_dict[f"{key}_mean"] = res["mean"]
        save_dict[f"{key}_std"] = res["std"]
        save_dict[f"{key}_alpha_conc"] = res["alpha_conc"]
        save_dict[f"{key}_gamma"] = res["gamma"]

    np.savez(npz_path, **save_dict)

    print(f"\n[Data saved] {npz_path}")

    # ==========================================================================
    # 5. 绘图：平均投资水平时间演化，symlog 横坐标，包含 t=0
    # ==========================================================================

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=300)

    colors = {
        "Fermi": "#4D4D4D",
        "NoSupp_NoMat": "#355C7D",
        "SuppOnly": "#F28E2B",
        "FullAIS": "#3AA17E",
    }

    line_styles = {
        "Fermi": "--",
        "NoSupp_NoMat": "-",
        "SuppOnly": "-",
        "FullAIS": "-",
    }

    line_widths = {
        "Fermi": 2.6,
        "NoSupp_NoMat": 2.8,
        "SuppOnly": 2.8,
        "FullAIS": 3.0,
    }

    plot_order = ["Fermi", "NoSupp_NoMat", "SuppOnly", "FullAIS"]

    for key in plot_order:
        res = all_results[key]

        times = res["times"]
        mean_series = res["mean"]
        std_series = res["std"]

        ax.plot(
            times,
            mean_series,
            color=colors[key],
            linestyle=line_styles[key],
            linewidth=line_widths[key],
            label=res["label"]
        )

        ax.fill_between(
            times,
            mean_series - std_series,
            mean_series + std_series,
            color=colors[key],
            alpha=0.10
        )

    # symlog 横坐标，可以显示 t=0
    ax.set_xscale("symlog", linthresh=1, linscale=0.12)
    ax.set_xlim(0, max_iter)

    ax.set_xticks([0, 1, 10, 100, 1000, 10000])
    ax.set_xticklabels([
        "0",
        r"$10^0$",
        r"$10^1$",
        r"$10^2$",
        r"$10^3$",
        r"$10^4$"
    ])

    ax.set_xlabel("Iteration", fontsize=14, fontweight="bold")
    ax.set_ylabel(r"Avg I", fontsize=14, fontweight="bold")

    ax.set_ylim(0, 1.05)

    ax.tick_params(axis="both", labelsize=15)

    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)

    legend = ax.legend(
        fontsize=14,
        loc="lower right",
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        borderpad=0.6,
        labelspacing=0.6,
        handlelength=2.4
    )

    legend.get_frame().set_edgecolor("#BFBFBF")

    plt.tight_layout()

    fig_path_png = os.path.join(
        output_folder,
        "Fig5_Fermi_AIS_Mechanism_Comparison_AvgI_SymLogX_WithT0.png"
    )

    fig_path_pdf = os.path.join(
        output_folder,
        "Fig5_Fermi_AIS_Mechanism_Comparison_AvgI_SymLogX_WithT0.pdf"
    )

    plt.savefig(fig_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig_path_pdf, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Figure saved] {fig_path_png}")
    print(f"[Figure saved] {fig_path_pdf}")

    # ==========================================================================
    # 6. 输出初始状态和最终稳态结果
    # ==========================================================================

    print("\n" + "=" * 90)
    print("Initial and final statistics of average investment")
    print("=" * 90)

    for key in plot_order:
        res = all_results[key]

        initial_values = res["all_runs"][:, 0]
        final_values = res["all_runs"][:, -1]

        print(
            f"{res['label']}: "
            f"Initial Avg I = {np.mean(initial_values):.4f} ± {np.std(initial_values):.4f}, "
            f"Final Avg I = {np.mean(final_values):.4f} ± {np.std(final_values):.4f}"
        )


if __name__ == "__main__":
    run_mechanism_time_evolution_experiment()