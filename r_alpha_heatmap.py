import numpy as np
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
from numba.core.errors import NumbaPerformanceWarning
import traceback

# 屏蔽 Numba 的底层性能警告
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

from AIS import SPGG_ImmuneSystem_GPU


def run_heatmap_point(args):
    """
    GPU 版本的单点仿真函数
    """

    a_idx, alpha_val, r_idx, r_val, delta_val, gamma_val, sigma_max, max_iter, grid_size, base_seed, num_runs = args

    avg_inv_list = []

    # 保持随机种子的唯一性
    base_seed = base_seed + int(r_val * 100) + int(alpha_val * 100) + int(delta_val * 1000)

    try:
        for run_idx in range(num_runs):
            current_seed = base_seed + run_idx * 10000


            sim = SPGG_ImmuneSystem_GPU(
                r=r_val,
                grid_size=grid_size,
                max_iterations=max_iter,
                alpha_conc=alpha_val,
                gamma=gamma_val,
                delta=delta_val,
                sigma_min=0.001,
                sigma_max=sigma_max,
                seed=current_seed
            )

            # GPU 模型内部已经处理好了稳态平均值的计算，直接拿结果
            steady_I, steady_R = sim.run_simulation()
            avg_inv_list.append(steady_I)

        # 返回索引和这 num_runs 次运行的整体平均值
        return a_idx, r_idx, np.mean(avg_inv_list)

    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"\n [GPU 错误] r={r_val}, alpha={alpha_val} 崩溃: {str(e)[:100]}...")
        # 报错时返回 -1 填入矩阵
        return a_idx, r_idx, -1.0


def generate_and_save_heatmap_data():
    # 基础参数设置
    grid_size = 100
    max_iter = 5000
    base_seed = 42
    num_runs = 5

    # 固定的演化参数
    gamma_val = 1.0
    sigma_max = 0.03

    res_x, res_y = 50, 50
    r_values = np.linspace(1.0, 4.0, res_x)
    alpha_values = np.linspace(0.0, 5.0, res_y)

    delta_list = [0.2,0.4,0.6,0.8,1.0]

    # 保守设置多进程数量，防止显存 OOM
    max_workers = min(6, max(1, os.cpu_count() // 2))

    for delta_val in delta_list:
        scenario_name = f'delta_{delta_val}'
        output_dir = f'heatmap_results_r_alpha_{scenario_name}'
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n========== 开始生成场景: {scenario_name} ==========")
        print(f"网格: {res_x}x{res_y}, 每个点独立运行 {num_runs} 次取平均...")

        tasks = []

        for r_idx, r_val in enumerate(r_values):
            for a_idx, a_val in enumerate(alpha_values):
                # 将所需的所有参数打包传给子进程
                tasks.append((
                    a_idx, a_val, r_idx, r_val, delta_val,
                    gamma_val, sigma_max, max_iter, grid_size, base_seed, num_runs
                ))

        # 初始化用来保存结果的二维矩阵
        avg_I_matrix = np.zeros((res_y, res_x))

        # 使用我们之前验证过绝对防卡死的 ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_heatmap_point, task) for task in tasks]

            with tqdm(total=len(tasks), desc=f"扫描 delta={delta_val}") as pbar:
                for future in as_completed(futures):
                    # 获取计算结果（直接拿到正确的行号 a_idx 和列号 r_idx）
                    ret_a_idx, ret_r_idx, steady_I = future.result()


                    avg_I_matrix[ret_a_idx, ret_r_idx] = steady_I

                    pbar.update(1)

        # 保存为 .npz 格式，完美兼容你的画图脚本
        data_path = os.path.join(output_dir, 'heatmap_data_avg_I_gpu.npz')
        np.savez(data_path,
                 avg_I=avg_I_matrix,
                 r_values=r_values,
                 alpha_values=alpha_values)

        print(f"场景 delta={delta_val} 数据已保存至: {data_path}")

    print("\n所有 delta 场景的 GPU 数据生成完毕！")


if __name__ == "__main__":
    # 必须加上 spawn 启动方法，防止 Windows 下 CUDA 进程冲突
    mp.set_start_method('spawn')
    generate_and_save_heatmap_data()