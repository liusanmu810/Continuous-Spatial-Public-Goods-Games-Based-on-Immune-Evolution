import numpy as np
import matplotlib.pyplot as plt
import os
import time
from numba import cuda
from scipy.ndimage import convolve

from AIS import (
    SPGG_ImmuneSystem_GPU,
    calc_instant_reputations_kernel,
    update_U_kernel,
    calc_payoffs_kernel,
    calc_concentrations_kernel,
    immune_sweep_kernel
)

# ==============================================================================
# Fig. 3: Time evolution of average local strategy diversity under different AIS mechanisms
# Include t=0 and keep logarithmic-style horizontal axis
# ==============================================================================

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def build_log_record_times(max_iter):
    """
    构造适合 symlog 横坐标的记录时间点。

    说明：
    - t=0 在 run_one_lsd_time_series() 中单独记录；
    - t=1~100 每代记录，用于捕捉早期变化；
    - t=110~1000 每 10 代记录；
    - t=1100~max_iter 每 100 代记录。
    """
    record_times = sorted(set(
        list(range(1, min(100, max_iter) + 1, 1)) +
        list(range(110, min(1000, max_iter) + 1, 10)) +
        list(range(1100, max_iter + 1, 100)) +
        [max_iter]
    ))
    return set(record_times)


def calculate_local_strategy_diversity(grid_I, grid_R):
    """
    基于冯·诺依曼邻域的二维局部策略多样性 LSD。

    LSD_i = sqrt( Var_local(I) + Var_local(R) )

    局部邻域：
    自身 + 上下左右，共 5 个节点。

    边界条件：
    使用 mode='wrap' 实现周期边界。
    """

    von_neumann_kernel = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=np.float64) / 5.0

    mean_I = convolve(grid_I, von_neumann_kernel, mode='wrap')
    mean_I2 = convolve(grid_I * grid_I, von_neumann_kernel, mode='wrap')
    var_I = np.clip(mean_I2 - mean_I * mean_I, 0.0, None)

    mean_R = convolve(grid_R, von_neumann_kernel, mode='wrap')
    mean_R2 = convolve(grid_R * grid_R, von_neumann_kernel, mode='wrap')
    var_R = np.clip(mean_R2 - mean_R * mean_R, 0.0, None)

    lsd_grid = np.sqrt(var_I + var_R)

    return lsd_grid


def run_one_lsd_time_series(params, seed_val, max_iter=10000):
    """
    单次运行，返回：
    - times: 记录的时间点；
    - avg_lsd_series: 每个时间点的全局平均 LSD。

    修改点：
    - 先记录初始状态 t=0；
    - 再从 t=1 开始演化；
    - 配合 symlog 横坐标，可以在图中显示初始平均 LSD。
    """

    sim = SPGG_ImmuneSystem_GPU(**params, seed=seed_val)

    times = []
    avg_lsd_series = []

    record_times = build_log_record_times(max_iter)

    # --------------------------------------------------------------------------
    # 0. 记录初始状态 t=0 的平均 LSD
    # --------------------------------------------------------------------------
    initial_I = sim.d_I.copy_to_host()
    initial_R = sim.d_R.copy_to_host()

    initial_lsd_grid = calculate_local_strategy_diversity(initial_I, initial_R)
    initial_avg_lsd = np.mean(initial_lsd_grid)

    times.append(0)
    avg_lsd_series.append(initial_avg_lsd)

    # --------------------------------------------------------------------------
    # 1. 从 t=1 开始正式演化
    # --------------------------------------------------------------------------
    for iteration in range(1, max_iter + 1):

        # ----------------------------------------------------------------------
        # 1. 更新瞬时声誉 / 免疫记忆
        # ----------------------------------------------------------------------
        calc_instant_reputations_kernel[sim.blockspergrid, sim.threadsperblock](
            sim.d_I,
            sim.d_R,
            sim.d_inst_U,
            sim.omega_I,
            sim.omega_R,
            sim.grid_size
        )

        update_U_kernel[sim.blockspergrid, sim.threadsperblock](
            sim.d_U,
            sim.d_inst_U,
            sim.alpha_U,
            sim.max_U,
            sim.grid_size
        )

        # ----------------------------------------------------------------------
        # 2. 计算收益
        # ----------------------------------------------------------------------
        calc_payoffs_kernel[sim.blockspergrid, sim.threadsperblock](
            sim.d_I,
            sim.d_R,
            sim.d_U,
            sim.d_P,
            sim.grid_size,
            sim.r,
            sim.reward_cost_factor,
            sim.reward_multiplier,
            sim.beta,
            sim.max_U
        )

        # ----------------------------------------------------------------------
        # 3. 计算全局平均免疫记忆
        # ----------------------------------------------------------------------
        h_U = sim.d_U.copy_to_host()
        global_avg_U = np.mean(h_U)

        # ----------------------------------------------------------------------
        # 4. 计算局部浓度
        # ----------------------------------------------------------------------
        calc_concentrations_kernel[sim.blockspergrid, sim.threadsperblock](
            sim.d_I,
            sim.d_R,
            sim.d_C,
            sim.grid_size,
            sim.delta
        )

        # ----------------------------------------------------------------------
        # 5. 免疫选择与亲和度成熟变异
        # ----------------------------------------------------------------------
        immune_sweep_kernel[sim.blockspergrid, sim.threadsperblock](
            sim.rng_states,
            sim.d_I,
            sim.d_R,
            sim.d_P,
            sim.d_U,
            sim.d_C,
            sim.d_new_I,
            sim.d_new_R,
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

        # 同步更新
        sim.d_I, sim.d_new_I = sim.d_new_I, sim.d_I
        sim.d_R, sim.d_new_R = sim.d_new_R, sim.d_R

        # ----------------------------------------------------------------------
        # 6. 记录平均 LSD
        # ----------------------------------------------------------------------
        if iteration in record_times:
            current_I = sim.d_I.copy_to_host()
            current_R = sim.d_R.copy_to_host()

            lsd_grid = calculate_local_strategy_diversity(current_I, current_R)
            avg_lsd = np.mean(lsd_grid)

            times.append(iteration)
            avg_lsd_series.append(avg_lsd)

    return np.array(times), np.array(avg_lsd_series)


def run_fig3_lsd_experiment():
    """
    三种机制配置下的平均局部策略多样性 overline{LSD}(t) 实验。
    """

    # ==========================================================================
    # 1. 基础参数设置
    # ==========================================================================
    L = 100
    max_iter = 10000
    num_runs = 5
    base_seed = 42

    test_r = 3.5

    output_folder = "results/fig3_mechanism_lsd_evolution"
    os.makedirs(output_folder, exist_ok=True)

    # ==========================================================================
    # 2. 三组机制配置
    # ==========================================================================
    scenarios = [
        {
            "label": r"$(\alpha=0,\gamma=0)$",
            "short_label": "alpha0_gamma0",
            "alpha_conc": 0.0,
            "gamma": 0.0,
        },
        {
            "label": r"$(\alpha=3,\gamma=0)$",
            "short_label": "alpha3_gamma0",
            "alpha_conc": 3.0,
            "gamma": 0.0,
        },
        {
            "label": r"$(\alpha=3,\gamma=1)$",
            "short_label": "alpha3_gamma1",
            "alpha_conc": 3.0,
            "gamma": 1.0,
        }
    ]

    # 与平均投资时间演化图保持一致
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
    print("Start Fig. 3 experiment: time evolution of average LSD")
    print(f"L={L}, r={test_r}, max_iter={max_iter}, runs={num_runs}")
    print("Initial state t=0 will be recorded.")
    print("=" * 90)

    for sc in scenarios:
        print(f"\n>>> Scenario: {sc['label']}")

        scenario_series = []
        common_times = None

        for run_idx in range(num_runs):
            seed_val = base_seed + run_idx * 1000

            params = base_params.copy()
            params["alpha_conc"] = sc["alpha_conc"]
            params["gamma"] = sc["gamma"]

            start_time = time.time()

            times, avg_lsd_series = run_one_lsd_time_series(
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

            scenario_series.append(avg_lsd_series)

            print(
                f"    Run {run_idx + 1:02d}/{num_runs} finished. "
                f"Initial Avg LSD = {avg_lsd_series[0]:.6f}, "
                f"Final Avg LSD = {avg_lsd_series[-1]:.6f}, "
                f"time = {time.time() - start_time:.2f}s"
            )

        scenario_series = np.array(scenario_series)

        mean_series = np.mean(scenario_series, axis=0)
        std_series = np.std(scenario_series, axis=0)

        all_results[sc["short_label"]] = {
            "label": sc["label"],
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
        "Fig3_AvgLSD_Time_Evolution_Data_SymLogX_WithT0.npz"
    )

    save_dict = {}
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
    # 5. 绘图：平均 LSD 时间演化，包含 t=0
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=300)

    colors = ["#355C7D", "#F28E2B", "#3AA17E"]
    line_widths = [2.8, 2.8, 3.0]

    for idx, sc in enumerate(scenarios):
        res = all_results[sc["short_label"]]

        times = res["times"]
        mean_series = res["mean"]
        std_series = res["std"]

        ax.plot(
            times,
            mean_series,
            color=colors[idx],
            linestyle="-",
            linewidth=line_widths[idx],
            label=res["label"]
        )

        ax.fill_between(
            times,
            mean_series - std_series,
            mean_series + std_series,
            color=colors[idx],
            alpha=0.12
        )

    # --------------------------------------------------------------------------
    # symlog 横坐标：
    # - 可以显示 t=0；
    # - linscale=0.15 用于压缩 0 到 10^0 之间的视觉间距；
    # - t>1 后仍然保持近似对数展开。
    # --------------------------------------------------------------------------
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
    ax.set_ylabel(r"Avg LSD", fontsize=14, fontweight="bold")

    ax.tick_params(axis="both", labelsize=15)

    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)

    legend = ax.legend(
        fontsize=15,
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        borderpad=0.6,
        labelspacing=0.6,
        handlelength=2.2
    )
    legend.get_frame().set_edgecolor("#BFBFBF")

    plt.tight_layout()

    fig_path_png = os.path.join(
        output_folder,
        "Fig3_AvgLSD_Time_Evolution_SymLogX_WithT0.png"
    )
    fig_path_pdf = os.path.join(
        output_folder,
        "Fig3_AvgLSD_Time_Evolution_SymLogX_WithT0.pdf"
    )

    plt.savefig(fig_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig_path_pdf, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Figure saved] {fig_path_png}")
    print(f"[Figure saved] {fig_path_pdf}")

    # ==========================================================================
    # 6. 输出初始状态和最终统计
    # ==========================================================================
    print("\n" + "=" * 90)
    print("Initial and final statistics of average LSD")
    print("=" * 90)

    for sc in scenarios:
        res = all_results[sc["short_label"]]

        initial_values = res["all_runs"][:, 0]
        final_values = res["all_runs"][:, -1]

        print(
            f"{res['label']}: "
            f"Initial Avg LSD = {np.mean(initial_values):.6f} ± {np.std(initial_values):.6f}, "
            f"Final Avg LSD = {np.mean(final_values):.6f} ± {np.std(final_values):.6f}"
        )


if __name__ == "__main__":
    run_fig3_lsd_experiment()