import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm



from AIS import SPGG_ImmuneSystem_GPU

def run_heatmap_point_r_delta(args):
    """单点仿真：跑有门阀的经典模型并返回 I 和 R 的稳态均值"""
    # 移除了 steady_window，因为 GPU 类里已经自带了最后 50 代平均
    alpha_val, r_val, delta_val, params, base_seed, num_runs = args

    steady_I_runs = []
    steady_R_runs = []

    for run_idx in range(num_runs):
        # 确保独立随机种子
        current_seed = base_seed + int(r_val * 100) + int(delta_val * 1000) + int(alpha_val * 10) + run_idx * 10000

        point_params = params.copy()
        point_params['alpha_conc'] = alpha_val
        point_params['r'] = r_val
        point_params['delta'] = delta_val
        point_params['seed'] = current_seed


        sim = SPGG_ImmuneSystem_GPU(**point_params)


        steady_I, steady_R = sim.run_simulation()

        steady_I_runs.append(steady_I)
        steady_R_runs.append(steady_R)

    return r_val, delta_val, np.mean(steady_I_runs), np.mean(steady_R_runs)


def generate_and_save_heatmap_data_r_delta():
    L = 100
    max_iter = 5000
    base_seed = 42
    num_runs = 5

    res_x, res_y = 50, 50
    r_values = np.linspace(1.0, 4.0, res_x)
    delta_values = np.linspace(0.05, 3.0, res_y)


    alpha_list = [ 1.0,3.0,5.0]

    # 基础黄金参数：K_P 和 K_U 已按您的要求回调为 0.2
    # 删除了不需要传给 GPU 的 visualize_interval 和 output_folder
    base_params = {
        'grid_size': L,
        'max_iterations': max_iter,
        'lambda_param': 0.9,
        'gamma': 1.0,
        'sigma_max': 0.03,
        'sigma_min': 0.001,
        'K_P': 0.2,
        'K_U': 0.2
    }


    max_workers = min(6, os.cpu_count() or 1)

    for alpha_val in alpha_list:
        scenario_name = f'alpha_{alpha_val}'
        output_dir = f'heatmap_results_r_delta_{scenario_name}'
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n========== 开始生成场景: {scenario_name} ==========")
        print(f"网格: {res_x}x{res_y}, 扫描 r 和 delta...")
        print(f"并发进程数: {max_workers}")

        tasks = []
        for r_val in r_values:
            for d_val in delta_values:
                tasks.append((alpha_val, r_val, d_val, base_params, base_seed, num_runs))

        avg_I_matrix = np.zeros((res_y, res_x))
        avg_R_matrix = np.zeros((res_y, res_x))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_heatmap_point_r_delta, task): task for task in tasks}
            with tqdm(total=len(tasks), desc=f"扫描 alpha={alpha_val}") as pbar:
                for future in as_completed(futures):

                    ret_r, ret_delta, steady_I, steady_R = future.result()

                    r_idx = np.where(r_values == ret_r)[0][0]
                    d_idx = np.where(delta_values == ret_delta)[0][0]
                    avg_I_matrix[d_idx, r_idx] = steady_I
                    avg_R_matrix[d_idx, r_idx] = steady_R

                    pbar.update(1)

        data_path = os.path.join(output_dir, 'heatmap_data_avg_I.npz')


        np.savez(data_path,
                 avg_I=avg_I_matrix,
                 avg_R=avg_R_matrix,
                 r_values=r_values,
                 delta_values=delta_values)

        print(f"✅ 数据已保存至: {data_path} (包含 I 和 R 矩阵)")


if __name__ == "__main__":
    generate_and_save_heatmap_data_r_delta()