import numpy as np
import matplotlib.pyplot as plt
import os
import time

from AIS import (
    SPGG_ImmuneSystem_GPU,
    calc_instant_reputations_kernel,
    update_U_kernel,
    calc_payoffs_kernel,
    calc_concentrations_kernel,
    immune_sweep_kernel
)

# 设置中文字体 (兼容科研论文对字体的要求)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run_single_scenario(sc_config, params, snapshots_indices, seed_val):
    """
    运行单个对照组的模拟，并返回截取的投资网格(I)和奖励网格(R)快照。
    """
    model_type = sc_config.get("type", "AIS")
    print(f"\n>>> 正在模拟: [{sc_config['name'].replace('\n', ' ')}] (Model: {model_type})")

    history_I = []
    history_R = []
    start_time = time.time()
    max_iter = snapshots_indices[-1] + 1  # 迭代目标设定为最大快照代数

    # ==========================================
    # IA-SPGG 实验组 (调用 GPU CUDA 运算)
    # ==========================================
    print(f"    参数: Alpha={sc_config['alpha']}, Gamma={sc_config['gamma']}")

    # 剥离 GPU 模型不需要的冗余参数
    gpu_params = params.copy()
    gpu_params.pop('visualize_interval', None)
    gpu_params.pop('output_folder', None)

    sim = SPGG_ImmuneSystem_GPU(**gpu_params, seed=seed_val)

    for iteration in range(max_iter):
        # --- 抓取快照 ---
        if iteration in snapshots_indices:

            history_I.append(sim.d_I.copy_to_host())
            history_R.append(sim.d_R.copy_to_host())
            print(f"    [+] 捕获第 {iteration} 代快照...")

        # --- 物理层更新 (GPU Kernels) ---
        calc_instant_reputations_kernel[sim.blockspergrid, sim.threadsperblock](
            sim.d_I, sim.d_R, sim.d_inst_U, sim.omega_I, sim.omega_R, sim.grid_size
        )
        update_U_kernel[sim.blockspergrid, sim.threadsperblock](
            sim.d_U, sim.d_inst_U, sim.alpha_U, sim.max_U, sim.grid_size
        )
        calc_payoffs_kernel[sim.blockspergrid, sim.threadsperblock](
            sim.d_I, sim.d_R, sim.d_U, sim.d_P,
            sim.grid_size, sim.r, sim.reward_cost_factor, sim.reward_multiplier,
            sim.beta, sim.max_U
        )

        # 提取全局声誉 (拷贝回 CPU 以计算均值，开销极小)
        h_U = sim.d_U.copy_to_host()
        global_avg_U = np.mean(h_U)

        # --- 进化层更新 (GPU Kernels) ---
        calc_concentrations_kernel[sim.blockspergrid, sim.threadsperblock](
            sim.d_I, sim.d_R, sim.d_C, sim.grid_size, sim.delta
        )
        immune_sweep_kernel[sim.blockspergrid, sim.threadsperblock](
            sim.rng_states, sim.d_I, sim.d_R, sim.d_P, sim.d_U, sim.d_C,
            sim.d_new_I, sim.d_new_R, sim.grid_size,
            sim.lambda_param, sim.alpha_conc, sim.gamma,
            sim.sigma_min, sim.sigma_max, global_avg_U, sim.K_P, sim.K_U
        )


        sim.d_I, sim.d_new_I = sim.d_new_I, sim.d_I
        sim.d_R, sim.d_new_R = sim.d_new_R, sim.d_R

    print(f"    耗时: {time.time() - start_time:.2f}s")
    return history_I, history_R


def run_spatial_pattern_comparison():
    # 1. 通用实验设置
    L = 100  # 网格大小
    test_r = 3.5
    snapshots_indices = [0, 5, 10, 500, 10000]
    base_seed = 42  # 保证各组的初始“雪花图”完全一致


    scenarios = [
        {"name": r" ", "type": "AIS", "alpha": 0.0, "gamma": 0.0},
        {"name": r" ", "type": "AIS", "alpha": 3.0, "gamma": 0.0},
        {"name": r" ", "type": "AIS", "alpha": 3.0, "gamma": 1.0}
    ]

    all_scenarios_history_I = []
    all_scenarios_history_R = []

    print(f"========== 启动空间演化对照实验 (L={L}, r={test_r}) ==========")

    # 3. 依次运行所有对照组
    for sc in scenarios:
        # 直接使用 AIS 的参数配置
        params = {
            'r': test_r, 'grid_size': L, 'max_iterations': snapshots_indices[-1] + 1,
            'delta': 1, 'sigma_min': 0.001, 'sigma_max': 0.03,
            'beta': 0.5, 'lambda_param': 0.9,
            'alpha_conc': sc["alpha"], 'gamma': sc["gamma"],
            'K_P': 0.2, 'K_U': 0.2,
            'visualize_interval': 0, 'output_folder': 'temp_dump'
        }

        history_I, history_R = run_single_scenario(sc, params, snapshots_indices, base_seed)
        all_scenarios_history_I.append(history_I)
        all_scenarios_history_R.append(history_R)

    # 4. 分别绘制 投资(I) 和 奖励(R) 的最终大图
    output_folder = 'results/exp17_spatial_patterns'
    os.makedirs(output_folder, exist_ok=True)

    # 画投资意愿斑图 (使用 plasma，对比极其鲜明)
    plot_comparison_snapshots(
        all_scenarios_history_I, scenarios, snapshots_indices, output_folder,
        cmap_name='plasma', var_label='Investment ($I$)', filename='Fig_Spatial_Ablation_Investment.png'
    )

    # 画奖励意愿斑图 (使用 viridis)
    plot_comparison_snapshots(
        all_scenarios_history_R, scenarios, snapshots_indices, output_folder,
        cmap_name='viridis', var_label='Reward ($R$)', filename='Fig_Spatial_Ablation_Reward.png'
    )

    plot_strategy_distribution(
        all_history_I=all_scenarios_history_I,
        scenarios=scenarios,
        times=snapshots_indices,
        output_folder=output_folder
    )


def plot_comparison_snapshots(all_history_data, scenarios, times, output_folder, cmap_name, var_label, filename):
    rows = len(scenarios)
    cols = len(times)
    # 因为少了一行对照组，自动调整一下画布高度
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows), dpi=300)
    font_title = {'fontsize': 18, 'fontweight': 'bold', 'family': 'sans-serif'}
    font_ylabel = {'fontsize': 16, 'fontweight': 'bold', 'family': 'sans-serif'}

    for r_idx in range(rows):
        for c_idx, t in enumerate(times):
            ax = axes[r_idx, c_idx]
            grid_data = all_history_data[r_idx][c_idx]

            im = ax.imshow(grid_data, cmap=cmap_name, vmin=0, vmax=1, origin='lower')
            ax.set_xticks([])
            ax.set_yticks([])

            if c_idx == 0:
                ax.set_ylabel(scenarios[r_idx]["name"], **font_ylabel, labelpad=25, rotation=90, va='center')
            if r_idx == 0:
                ax.set_title(f"Iteration $t={t}$", **font_title, pad=15)

    plt.subplots_adjust(left=0.08, right=0.90, bottom=0.05, top=0.95, wspace=0.05, hspace=0.08)
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(var_label, fontsize=18, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=14)

    save_path = os.path.join(output_folder, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\n[大功告成] {rows}x{cols} 空间斑图 [{var_label}] 已保存至: {save_path}")


def plot_strategy_distribution(all_history_I, scenarios, times, output_folder,
                               filename='Fig_Strategy_Distribution_Evolution.png'):
    rows = len(scenarios)
    cols = len(times)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), dpi=300)
    bins = 20

    for r_idx in range(rows):
        sc_name = scenarios[r_idx]["name"].replace('\n', ' ')
        for c_idx, t in enumerate(times):
            ax = axes[r_idx, c_idx]
            grid_data = all_history_I[r_idx][c_idx]
            data = grid_data.flatten()
            total_count = len(data)

            counts, bin_edges, _ = ax.hist(data, bins=bins, range=(0.0, 1.0), color='orange', edgecolor='black',
                                           alpha=0.9)
            ax.set_xlim(-0.05, 1.05)
            max_freq = np.max(counts) if np.max(counts) > 0 else 1
            ax.set_ylim(0, max_freq * 1.2)

            percentages = (counts / total_count) * 100
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            ax2 = ax.twinx()
            ax2.plot(bin_centers, percentages, color='red', marker='o', linestyle='--', markersize=5)
            ax2.set_ylim(0, 100)

            if r_idx == rows - 1:
                ax.set_xlabel('Investment Level ($I$)', fontsize=14)
            else:
                ax.set_xticklabels([])

            if c_idx == 0:
                ax.set_ylabel(f"{sc_name}\n\nFrequency", fontsize=14, fontweight='bold')
            else:
                ax.set_ylabel('')

            if c_idx == cols - 1:
                ax2.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold', color='red')
                ax2.tick_params(axis='y', colors='red')
            else:
                ax2.set_ylabel('')
                ax2.set_yticks([])

            if r_idx == 0:
                ax.set_title(f"Iteration $t={t}$", fontsize=16, fontweight='bold', pad=15)

            ax.text(0.5, 0.9, f"$t={t}$", transform=ax.transAxes, ha='center', va='top',
                    fontsize=12, fontweight='bold', fontfamily='sans-serif',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    plt.subplots_adjust(wspace=0.25, hspace=0.15)
    save_path = os.path.join(output_folder, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\n[大功告成] 多代策略分布演化矩阵 已保存至: {save_path}")


if __name__ == "__main__":
    run_spatial_pattern_comparison()