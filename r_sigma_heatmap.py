import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm



from AIS import SPGG_ImmuneSystem_GPU

def run_heatmap_point_r_sigma(args):
    """单点仿真：扫描 r 和 sigma_max 并返回 I 和 R 的稳态均值"""
    gamma_val, r_val, sigma_val, params, base_seed, num_runs = args

    steady_I_runs = []
    steady_R_runs = []

    for run_idx in range(num_runs):
        # 确保独立随机种子，加上各参数特征值避免重复
        current_seed = base_seed + int(r_val * 100) + int(sigma_val * 10000) + int(gamma_val * 10) + run_idx * 10000

        point_params = params.copy()
        point_params['gamma'] = gamma_val
        point_params['r'] = r_val
        point_params['sigma_max'] = sigma_val  # 动态注入 sigma_max
        point_params['seed'] = current_seed

        #  实例化 GPU 模型
        sim = SPGG_ImmuneSystem_GPU(**point_params)

        # 一键运行并获取双稳态结果
        steady_I, steady_R = sim.run_simulation()

        steady_I_runs.append(steady_I)
        steady_R_runs.append(steady_R)

    return r_val, sigma_val, np.mean(steady_I_runs), np.mean(steady_R_runs)


def generate_and_save_heatmap_data_r_sigma():
    L = 100  # 网格大小
    max_iter = 5000
    base_seed = 42
    num_runs = 5

    res_x, res_y = 50, 50
    r_values = np.linspace(1.0, 4.0, res_x)
    # 扫描 sigma_max，通常变异步长在 [0, 1] 空间中不需要太大，0.001 到 0.1 足够观察演化差异
    sigma_values = np.linspace(0.001, 1, res_y)

    # 设定的两个 gamma (亲和力敏感度) 断面
    # gamma 越大，只有亲和力极低的个体才会发生大步长变异
    gamma_list = [1.0,3.0,5.0]

    # 基础参数
    base_params = {
        'grid_size': L,
        'max_iterations': max_iter,
        'lambda_param': 0.9,
        'alpha_conc': 3.0,
        'delta': 1.0,
        'sigma_min': 0.001,  # 固定最小变异步长
        'K_P': 0.2,
        'K_U': 0.2
    }

    #  防爆显存：控制多进程并发数
    max_workers = min(6, os.cpu_count() or 1)

    for gamma_val in gamma_list:
        scenario_name = f'gamma_{gamma_val}'
        output_dir = f'heatmap_results_r_sigma_{scenario_name}'
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n========== 开始生成场景: {scenario_name} ==========")
        print(f"网格: {res_x}x{res_y}, 扫描 r 和 sigma_max...")
        print(f"并发进程数: {max_workers}")

        tasks = []
        for r_val in r_values:
            for s_val in sigma_values:
                tasks.append((gamma_val, r_val, s_val, base_params, base_seed, num_runs))

        avg_I_matrix = np.zeros((res_y, res_x))
        avg_R_matrix = np.zeros((res_y, res_x))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_heatmap_point_r_sigma, task): task for task in tasks}
            with tqdm(total=len(tasks), desc=f"扫描 gamma={gamma_val}") as pbar:
                for future in as_completed(futures):
                    # 接收解包结果
                    ret_r, ret_sigma, steady_I, steady_R = future.result()

                    r_idx = np.where(r_values == ret_r)[0][0]
                    s_idx = np.where(sigma_values == ret_sigma)[0][0]
                    avg_I_matrix[s_idx, r_idx] = steady_I
                    avg_R_matrix[s_idx, r_idx] = steady_R

                    pbar.update(1)

        data_path = os.path.join(output_dir, 'heatmap_data_avg_I_R.npz')

        # 保存所有相关数据
        np.savez(data_path,
                 avg_I=avg_I_matrix,
                 avg_R=avg_R_matrix,
                 r_values=r_values,
                 sigma_values=sigma_values)

        print(f"✅ 数据已保存至: {data_path} (包含 I 和 R 矩阵)")


if __name__ == "__main__":
    generate_and_save_heatmap_data_r_sigma()