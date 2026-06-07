import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


from AIS import SPGG_ImmuneSystem_GPU


def run_heatmap_point_r_gamma(args):
    """单点仿真函数，多次运行取平均"""
    # 注意：稳态窗口 (steady_window) 在 GPU 类中已硬编码为 50，所以这里不再需要传入
    sigma_val, r_val, gamma_val, params, base_seed, num_runs = args

    steady_I_runs = []
    steady_R_runs = []

    for run_idx in range(num_runs):
        # 确保随机种子独立
        current_seed = base_seed + int(r_val * 100) + int(gamma_val * 100) + int(sigma_val * 1000) + run_idx * 10000

        point_params = params.copy()
        point_params['sigma_max'] = sigma_val
        point_params['r'] = r_val
        point_params['gamma'] = gamma_val
        point_params['seed'] = current_seed

        sim = SPGG_ImmuneSystem_GPU(**point_params)


        steady_I, steady_R = sim.run_simulation()

        # 记录本次运行的稳态平均值
        steady_I_runs.append(steady_I)
        steady_R_runs.append(steady_R)

    # 返回 r_val, gamma_val 及其平均投资 (I) 和 平均奖励倾向 (R)
    return r_val, gamma_val, np.mean(steady_I_runs), np.mean(steady_R_runs)


def generate_and_save_heatmap_data_r_gamma():
    L = 100
    max_iter = 5000
    base_seed = 42
    num_runs = 5  # 跑5次求平均

    res_x, res_y = 50, 50
    r_values = np.linspace(1.0, 4.0, res_x)
    gamma_values = np.linspace(0.0, 5.0, res_y)  # 设定 gamma 范围 0 到 5


    sigma_list = [0.01, 0.05, 0.2, 0.5]

    # 固定免疫系统的核心参数，必须与 GPU 类的参数名完全匹配
    base_params = {
        'grid_size': L,
        'max_iterations': max_iter,
        'lambda_param': 0.9,
        'alpha_conc': 3.0,
        'delta': 1.0,
        'sigma_min': 0.001,
        'K_P': 0.2,
        'K_U': 0.2
        # 其他未填写的参数（如 beta, reward_multiplier）将自动使用 GPU 类的默认值
    }


    # 请根据你的显存大小调整，普通显卡建议 2-4，24G 显卡可以尝试 8-10
    max_workers = min(6, os.cpu_count() or 1)

    for sigma_val in sigma_list:
        scenario_name = f'sigma_{sigma_val}'
        output_dir = f'heatmap_results_r_gamma_{scenario_name}'
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n========== 开始生成场景: {scenario_name} ==========")
        print(f"网格: {res_x}x{res_y}, 每个点独立运行 {num_runs} 次取平均...")
        print(f"并发进程数: {max_workers} (防止 GPU 爆显存)")

        tasks = []
        for r_val in r_values:
            for g_val in gamma_values:
                # 移除了 steady_window 参数
                tasks.append((sigma_val, r_val, g_val, base_params, base_seed, num_runs))

        avg_I_matrix = np.zeros((res_y, res_x))
        avg_R_matrix = np.zeros((res_y, res_x))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_heatmap_point_r_gamma, task): task for task in tasks}
            with tqdm(total=len(tasks), desc=f"扫描 sigma_max={sigma_val}") as pbar:
                for future in as_completed(futures):

                    ret_r, ret_gamma, steady_I, steady_R = future.result()

                    r_idx = np.where(r_values == ret_r)[0][0]
                    g_idx = np.where(gamma_values == ret_gamma)[0][0]
                    avg_I_matrix[g_idx, r_idx] = steady_I
                    avg_R_matrix[g_idx, r_idx] = steady_R  # 存入 R 数据

                    pbar.update(1)

        data_path = os.path.join(output_dir, 'heatmap_data_avg_I.npz')

        # 保存时，同时存入 avg_I 和 avg_R
        np.savez(data_path,
                 avg_I=avg_I_matrix,
                 avg_R=avg_R_matrix,
                 r_values=r_values,
                 gamma_values=gamma_values)

        print(f"✅ 数据已保存至: {data_path} (包含 I 和 R 矩阵)")


if __name__ == "__main__":
    generate_and_save_heatmap_data_r_gamma()