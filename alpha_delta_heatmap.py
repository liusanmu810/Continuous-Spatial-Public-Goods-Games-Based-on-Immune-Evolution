import numpy as np
import time
import os
import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
from numba.core.errors import NumbaPerformanceWarning
import traceback

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

from AIS import SPGG_ImmuneSystem_GPU


def run_single_sim(params):
    """
    独立运算函数（自带防崩溃护甲）
    """
    r_val, alpha_val, gamma_val, delta_val, sigma_val = params
    num_runs = 5

    # 同时准备两个列表，分别记录 I (投资) 和 R (奖励意愿)
    avg_inv_list = []
    avg_rew_list = []

    # 使用所有参数组合生成唯一的 base_seed
    base_seed = 42

    try:
        for run_idx in range(num_runs):
            current_seed = base_seed + run_idx * 999

            # 调用最新版的 GPU 模型
            sim = SPGG_ImmuneSystem_GPU(
                r=r_val,
                grid_size=100,  # 100x100 的大网格，结果更稳定
                max_iterations=5000,  # 5000 代确保系统完全进入稳态
                alpha_conc=alpha_val,
                gamma=gamma_val,
                delta=delta_val,
                sigma_min=0.001,
                sigma_max=sigma_val,
                seed=current_seed
            )

            # 精准解包，完美适配最新模型返回的两个值
            steady_I, steady_R = sim.run_simulation()

            avg_inv_list.append(steady_I)
            avg_rew_list.append(steady_R)

        # 分别求 5 次独立运行的平均值
        final_avg_investment = np.mean(avg_inv_list)
        final_avg_reward = np.mean(avg_rew_list)

        return (r_val, alpha_val, gamma_val, delta_val, sigma_val, final_avg_investment, final_avg_reward)

    except Exception as e:
        err_msg = traceback.format_exc()
        # 将报错信息精简，防止刷屏导致看不清进度条
        print(f"\n❌ [GPU 错误] 参数 {params} 崩了: {str(e)[:100]}...")
        # 出错时返回 -1.0 占位，保证 CSV 数据对齐
        return (r_val, alpha_val, gamma_val, delta_val, sigma_val, -1.0, -1.0)


def get_all_params():
    """
    直接生成参数列表
    """
    r_list = [1.8,2.0]
    alpha_list = np.round(np.arange(0, 5.1, 0.1), 2)
    gamma_list = [1.0]  # 当前设定的 Gamma 值
    delta_list = np.round(np.linspace(0, 3.0, 50), 4)
    sigma_list = [0.03]

    all_params = []
    for r in r_list:
        for a in alpha_list:
            for g in gamma_list:
                for d in delta_list:
                    for s in sigma_list:
                        all_params.append((r, a, g, d, s))
    return all_params, gamma_list[0]


if __name__ == "__main__":
    # 强制使用 spawn，防止 CUDA 上下文在多进程中死锁
    mp.set_start_method('spawn')

    print("-> 正在生成参数列表...")
    params_list, current_gamma = get_all_params()
    total_experiments = len(params_list)

    print(f"=== 🚀 绝对防卡死版 GPU 扫描启动 (当前测试 γ={current_gamma}) ===")
    print(f"-> 总计参数点数: {total_experiments:,} 个 (底层运算 {total_experiments * 5:,} 次)")

    # 动态分配 Worker 数量，兼顾速度与显存安全
    max_workers = min(6, max(1, os.cpu_count() // 2))
    print(f"-> 启用 {max_workers} 个并行 Worker...\n")

    # 动态命名保存文件，防止覆盖
    save_dir = "results/exp10_Heatmaps_Alpha_Delta"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"gpu_sweep_results_gamma_{current_gamma}.csv")

    start_time = time.time()

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头（包含新加的 Avg_Reward）
        writer.writerow(['r', 'alpha', 'gamma', 'delta', 'sigma', 'Avg_Investment', 'Avg_Reward'])

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(run_single_sim, p) for p in params_list]

            # as_completed 实时更新进度条
            with tqdm(total=total_experiments, desc="极速扫描中", unit="点") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    writer.writerow(res)
                    pbar.update(1)

                    # 每 500 个点强制刷新缓冲区，落盘保存
                    if pbar.n % 500 == 0:
                        f.flush()

    elapsed = (time.time() - start_time)
    print("\n=== 实验圆满结束 ===")
    print(f"✅ {total_experiments:,} 组数据已安全保存至: '{save_path}'")
    if elapsed > 3600:
        print(f"⏱️ 总耗时: {elapsed / 3600:.2f} 小时")
    else:
        print(f"⏱️ 总耗时: {elapsed / 60:.2f} 分钟")