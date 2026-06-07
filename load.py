import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter, zoom
import matplotlib.ticker as ticker


plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_ultimate_heatmap_from_csv(csv_filepath, zoom_factor=8, smoothing_sigma=6):
    """
    终极版平滑热力图 (基于 CSV)：动态 Sigma + Bicubic 插值 + 完美正方形比例 + 600 DPI
    X轴: Concentration Inhibition (alpha), Y轴: Concentration Tolerance Radius (delta)
    """
    print(f"\n========== 正在准备渲染极致高清图 ==========")
    if not os.path.exists(csv_filepath):
        print(f"❌ 错误：未找到数据文件 '{csv_filepath}'。请检查路径。")
        return

    # 建立统一的高清输出文件夹
    output_dir = 'results/Ultimate_Heatmaps_Alpha_Delta'
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. 读取 CSV 数据
        df = pd.read_csv(csv_filepath)
        print(f"✅ 数据读取成功！共计 {len(df)} 行。")

        # 预设的 r 值断面
        r_list = [1.5, 1.8, 2.0, 3.0]
        gamma_val = df['gamma'].iloc[0]  # 自动获取当前的 gamma 值

        for r_val in r_list:
            print(f"\n-> 正在处理 r = {r_val} 的相图...")

            # 2. 提取当前 r 值的数据子集
            subset = df[df['r'] == r_val]
            if subset.empty:
                print(f"   [跳过] r={r_val} 无数据。")
                continue


            pivot_table = subset.pivot(index='delta', columns='alpha', values='Avg_Investment')

            # 提取坐标轴和值矩阵
            loaded_alpha_values = pivot_table.columns.values
            loaded_delta_values = pivot_table.index.values
            loaded_avg_I = pivot_table.values

            # 清理可能的 -1.0 (实验报错占位符)，防止污染高斯平滑
            loaded_avg_I[loaded_avg_I < 0] = 0.0


            adapted_sigma = smoothing_sigma * (zoom_factor / 4.0)
            print(f"   [设置] 上采样倍率 (zoom): {zoom_factor}x, 高斯模糊 (sigma): {adapted_sigma}")

            upsampled_avg_I = zoom(loaded_avg_I, zoom=zoom_factor, order=3)
            smoothed_avg_I = gaussian_filter(upsampled_avg_I, sigma=adapted_sigma)

            # 强制截断防溢出
            smoothed_avg_I = np.clip(smoothed_avg_I, 0.0, 1.0)

            # 物理坐标映射: X轴(alpha), Y轴(delta)
            common_extent = [loaded_alpha_values.min(), loaded_alpha_values.max(),
                             loaded_delta_values.min(), loaded_delta_values.max()]

            # 5. 绘图 (调整 figsize 为适应正方形和 Colorbar 的 7x6)
            fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)


            im = ax.imshow(smoothed_avg_I, origin='lower', cmap='jet',
                           extent=common_extent, aspect='auto', vmin=0, vmax=1,
                           interpolation='bicubic')


            ax.set_box_aspect(1)

            # --- 标题与坐标轴设置 (英文国际化标准) ---
            ax.set_title('Avg I', fontsize=20, pad=4)
            ax.set_xlabel(r'$\alpha$', fontsize=16)
            ax.set_ylabel(r'$\delta$', fontsize=16)

            # X轴刻度 (alpha: 0 ~ 5.0)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

            # Y轴刻度 (delta: 0 ~ 3.0)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=10)

            # 6. Colorbar 设置
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


            # 7. 🌟 顶刊标准 600 DPI 输出
            replot_output_filename = f'Ultimate_Heatmap_Avg_I_r_{r_val}_gamma_{gamma_val}.png'
            replot_output_path = os.path.join(output_dir, replot_output_filename)
            plt.savefig(replot_output_path, dpi=600, bbox_inches='tight')
            plt.close(fig)  # 静默关闭

            print(f"    渲染完成！已保存至: {replot_output_path}")

    except Exception as e:
        print(f"读取或绘图过程中发生错误: {e}")


if __name__ == "__main__":
    # 确保这里的路径与您实验代码保存的路径完全一致
    csv_file = "results/exp10_Heatmaps_Alpha_Delta/gpu_sweep_results_gamma_1.0.csv"

    # 使用极致画质参数：8倍上采样，基础平滑为 6
    plot_ultimate_heatmap_from_csv(csv_file, zoom_factor=4, smoothing_sigma=8)