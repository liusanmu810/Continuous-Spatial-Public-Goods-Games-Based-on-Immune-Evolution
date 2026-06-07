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
    独立运算函数（自带防崩溃护甲，每个点跑 5 次取平均）
    """
    r_val, alpha_val, gamma_val, delta_val, sigma_val = params
    num_runs =5
    avg_inv_list = []

    # 生成唯一的 Base Seed
    base_seed = int((r_val + alpha_val + gamma_val + delta_val + sigma_val) * 1000)

    try:
        for run_idx in range(num_runs):
            current_seed = base_seed + run_idx * 999

            sim = SPGG_ImmuneSystem_GPU(
                r=r_val,
                grid_size=100,
                max_iterations=5000,
                alpha_conc=alpha_val,
                gamma=gamma_val,
                delta=delta_val,
                sigma_min=0.001,
                sigma_max=sigma_val,
                seed=current_seed
            )

            single_run_avg = sim.run_simulation()
            avg_inv_list.append(single_run_avg)

        # 返回 5 次独立实验的平滑均值
        final_avg_investment = np.mean(avg_inv_list)
        return (r_val, alpha_val, gamma_val, delta_val, sigma_val, final_avg_investment)

    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"\n❌ [GPU 错误] 参数 {params} 崩了: {str(e)[:100]}...")
        return (r_val, alpha_val, gamma_val, delta_val, sigma_val, -1.0)


def get_all_params():
    """
    生成 Gamma vs Sigma 的高精度扫描网格
    """

    r_list = [1.5, 1.8, 2.0, 3.0]


    alpha_list = [3.0]
    delta_list = [1]


    gamma_list = np.round(np.arange(0, 5.1, 0.1), 2)  # 51 个点
    sigma_list = np.round(np.linspace(0.001, 1.0, 50), 4)  # 50 个点

    all_params = []
    for r in r_list:
        for a in alpha_list:
            for g in gamma_list:
                for d in delta_list:
                    for s in sigma_list:
                        all_params.append((r, a, g, d, s))
    return all_params


if __name__ == "__main__":
    # 必须使用 spawn，防止 CUDA 冲突
    mp.set_start_method('spawn')

    print("=== Gamma vs Sigma 高精度热力图扫描启动 ===")

    print("-> 正在生成参数列表...")
    params_list = get_all_params()
    total_experiments = len(params_list)
    print(f"-> 总计参数点数: {total_experiments:,} 个 (底层运算 {total_experiments * 5:,} 次)")

    # 启用 Worker 数量，保守使用一半核心数，确保稳定高速
    max_workers = min(6, max(1, os.cpu_count() // 2))
    print(f"-> 启用 {max_workers} 个并行 Worker 喂饱 GPU...\n")

    # 独立的保存文件名
    save_path = "results/gpu_heatmap_gamma_sigma_results.csv"
    start_time = time.time()

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['r', 'alpha', 'gamma', 'delta', 'sigma', 'Avg_Investment'])

        # 核心防卡死：ProcessPoolExecutor + as_completed
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_single_sim, p) for p in params_list]

            with tqdm(total=total_experiments, desc="热力图网格扫描进度", unit="点") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    writer.writerow(res)
                    pbar.update(1)

                    # 每 500 个点强行落盘，防止意外断电
                    if pbar.n % 500 == 0:
                        f.flush()

    elapsed = (time.time() - start_time)
    print("\n=== 实验圆满结束 ===")
    print(f"✅ {total_experiments:,} 组数据已安全保存至: '{save_path}'")
    if elapsed > 3600:
        print(f"⏱️ 总耗时: {elapsed / 3600:.2f} 小时")
    else:
        print(f"⏱️ 总耗时: {elapsed / 60:.2f} 分钟")