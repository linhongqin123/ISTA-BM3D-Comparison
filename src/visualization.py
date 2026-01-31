"""
可视化工具模块
用于结果展示和图表生成
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cv2
from pathlib import Path




def setup_plot_style():
    """设置绘图样式（安全版本）"""
    try:
        # 获取所有可用样式
        available_styles = plt.style.available
        
        # 尝试不同的样式名称
        style_candidates = [
            'seaborn-v0_8-darkgrid',
            'seaborn-darkgrid', 
            'seaborn',
            'ggplot',
            'classic',
            'default'
        ]
        
        selected_style = 'default'
        for candidate in style_candidates:
            if candidate in available_styles:
                selected_style = candidate
                break
        
        plt.style.use(selected_style)
        
        # 设置seaborn调色板（如果可用）
        try:
            sns.set_palette("husl")
        except:
            pass  # 如果seaborn不可用，跳过
        
        # 设置matplotlib参数
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
        
        return selected_style
        
    except Exception as e:
        print(f"警告: 设置绘图样式时出错: {e}")
        print("使用默认样式设置")
        # 设置最基本的参数
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12
        })
        return 'default'

def plot_image_comparison(original, noisy, denoised_list, 
                         titles_list=None, 
                         figsize=(20, 12)):
    """
    绘制图像对比图
    
    参数:
        original: 原始图像
        noisy: 含噪图像
        denoised_list: 去噪图像列表
        titles_list: 标题列表
        figsize: 图形大小
    """
    setup_plot_style()
    
    n_images = 2 + len(denoised_list)  # 原始 + 含噪 + 多个去噪结果
    
    # 创建子图
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    if n_images == 1:
        axes = [axes]
    
    # 确保标题列表长度正确
    if titles_list is None:
        titles_list = ['Original', 'Noisy'] + [f'Denoised {i+1}' for i in range(len(denoised_list))]
    elif len(titles_list) != n_images:
        titles_list = titles_list[:n_images] + [f'Denoised {i}' for i in range(len(titles_list), n_images)]
    
    # 显示原始图像
    if len(original.shape) == 2:  # 灰度图
        axes[0].imshow(original, cmap='gray', vmin=0, vmax=255)
    else:  # 彩色图
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title(titles_list[0])
    axes[0].axis('off')
    
    # 显示含噪图像
    if len(noisy.shape) == 2:  # 灰度图
        axes[1].imshow(noisy, cmap='gray', vmin=0, vmax=255)
    else:  # 彩色图
        axes[1].imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
    axes[1].set_title(titles_list[1])
    axes[1].axis('off')
    
    # 显示去噪结果
    for i, denoised in enumerate(denoised_list):
        if len(denoised.shape) == 2:  # 灰度图
            axes[i+2].imshow(denoised, cmap='gray', vmin=0, vmax=255)
        else:  # 彩色图
            axes[i+2].imshow(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
        axes[i+2].set_title(titles_list[i+2])
        axes[i+2].axis('off')
    
    plt.tight_layout()
    return fig

def plot_metrics_comparison(metrics_dict, noise_levels, 
                           metric_type='PSNR',
                           figsize=(10, 6)):
    """
    绘制指标对比图
    
    参数:
        metrics_dict: 指标字典 {算法名: [指标值列表]}
        noise_levels: 噪声水平列表
        metric_type: 指标类型 ('PSNR' 或 'SSIM')
        figsize: 图形大小
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_dict)))
    
    for i, (algo_name, metrics) in enumerate(metrics_dict.items()):
        if len(metrics) != len(noise_levels):
            print(f"警告: {algo_name}的数据长度与噪声水平数不匹配")
            continue
        
        ax.plot(noise_levels, metrics, 
                marker=markers[i % len(markers)],
                color=colors[i],
                linewidth=2,
                markersize=8,
                label=algo_name)
    
    ax.set_xlabel('Noise Level', fontweight='bold')
    ax.set_ylabel(f'{metric_type}', fontweight='bold')
    ax.set_title(f'{metric_type} Comparison Across Noise Levels', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_convergence_curve(convergence_data, 
                          title='ISTA Convergence',
                          figsize=(12, 8)):
    """
    绘制收敛曲线
    
    参数:
        convergence_data: 收敛数据字典
        title: 标题
        figsize: 图形大小
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # 1. PSNR收敛曲线
    if 'psnr_history' in convergence_data:
        axes[0].plot(convergence_data['psnr_history'], linewidth=2)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('PSNR (dB)')
        axes[0].set_title('PSNR Convergence')
        axes[0].grid(True, alpha=0.3)
    
    # 2. 目标函数值收敛曲线
    if 'objective_history' in convergence_data:
        axes[1].plot(convergence_data['objective_history'], linewidth=2)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Objective Value')
        axes[1].set_title('Objective Function Convergence')
        axes[1].grid(True, alpha=0.3)
    
    # 3. 残差收敛曲线
    if 'residual_history' in convergence_data:
        axes[2].plot(convergence_data['residual_history'], linewidth=2)
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Residual')
        axes[2].set_title('Residual Convergence')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale('log')
    
    # 4. 步长变化曲线
    if 'step_size_history' in convergence_data:
        axes[3].plot(convergence_data['step_size_history'], linewidth=2)
        axes[3].set_xlabel('Iteration')
        axes[3].set_ylabel('Step Size')
        axes[3].set_title('Step Size Adaptation')
        axes[3].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_computation_time(time_dict, 
                         algorithms=None,
                         figsize=(10, 6)):
    """
    绘制计算时间对比图
    
    参数:
        time_dict: 时间字典 {图像名: {算法: 时间}}
        algorithms: 算法列表
        figsize: 图形大小
    """
    setup_plot_style()
    
    if algorithms is None:
        # 从字典中提取所有算法
        algorithms = set()
        for image_times in time_dict.values():
            algorithms.update(image_times.keys())
        algorithms = list(algorithms)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 准备数据
    image_names = list(time_dict.keys())
    n_images = len(image_names)
    n_algorithms = len(algorithms)
    
    # 柱状图数据
    bar_width = 0.2
    index = np.arange(n_images)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_algorithms))
    
    # 绘制每个算法的柱状图
    for i, algo in enumerate(algorithms):
        times = [time_dict[img].get(algo, 0) for img in image_names]
        ax1.bar(index + i * bar_width, times, bar_width,
               label=algo, color=colors[i])
    
    ax1.set_xlabel('Image')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Computation Time per Image')
    ax1.set_xticks(index + bar_width * (n_algorithms - 1) / 2)
    ax1.set_xticklabels(image_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 绘制平均时间饼图
    avg_times = []
    for algo in algorithms:
        times = [time_dict[img].get(algo, 0) for img in image_names]
        avg_times.append(np.mean(times))
    
    ax2.pie(avg_times, labels=algorithms, autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax2.set_title('Average Computation Time Distribution')
    
    plt.tight_layout()
    return fig

def save_figure(fig, filename, dpi=300, transparent=False):
    """
    保存图形到文件
    
    参数:
        fig: matplotlib图形对象
        filename: 输出文件名
        dpi: 分辨率
        transparent: 是否透明背景
    """
    # 创建目录
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    # 保存图形
    fig.savefig(filename, dpi=dpi, transparent=transparent,
               bbox_inches='tight', pad_inches=0.1)
    print(f"图形已保存到: {filename}")
    
    # 关闭图形以释放内存
    plt.close(fig)

def visualize_noise_distribution(original, noisy_images, 
                                noise_names=None,
                                figsize=(15, 10)):
    """
    可视化噪声分布
    
    参数:
        original: 原始图像
        noisy_images: 含噪图像列表
        noise_names: 噪声名称列表
        figsize: 图形大小
    """
    setup_plot_style()
    
    n_noise = len(noisy_images)
    
    fig, axes = plt.subplots(n_noise, 3, figsize=figsize)
    if n_noise == 1:
        axes = axes.reshape(1, -1)
    
    for i, (noisy, name) in enumerate(zip(noisy_images, noise_names or [])):
        # 计算噪声
        noise = noisy.astype(np.float32) - original.astype(np.float32)
        
        # 显示噪声图像
        axes[i, 0].imshow(noisy, cmap='gray')
        axes[i, 0].set_title(f'{name}\nNoisy Image')
        axes[i, 0].axis('off')
        
        # 显示噪声分布
        axes[i, 1].imshow(noise, cmap='seismic', vmin=-50, vmax=50)
        axes[i, 1].set_title('Noise Distribution')
        axes[i, 1].axis('off')
        
        # 显示噪声直方图
        axes[i, 2].hist(noise.ravel(), bins=100, alpha=0.7)
        axes[i, 2].set_title('Noise Histogram')
        axes[i, 2].set_xlabel('Noise Value')
        axes[i, 2].set_ylabel('Frequency')
        axes[i, 2].grid(True, alpha=0.3)
        
        # 添加统计信息
        noise_std = np.std(noise)
        axes[i, 2].text(0.05, 0.95, f'Std: {noise_std:.2f}',
                       transform=axes[i, 2].transAxes,
                       verticalalignment='top')
    
    plt.suptitle('Noise Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # 测试代码
    # 创建测试数据
    original = np.random.rand(256, 256) * 255
    noisy1 = original + np.random.randn(256, 256) * 25
    noisy2 = original + np.random.randn(256, 256) * 50
    
    # 测试图像对比
    fig1 = plot_image_comparison(
        original, noisy1, [noisy2],
        ['Original', 'Noisy (σ=25)', 'Noisy (σ=50)']
    )
    plt.show()
    
    # 测试指标对比
    metrics_data = {
        'ISTA': [28.5, 26.8, 24.2],
        'BM3D': [30.2, 28.5, 26.0]
    }
    noise_levels = [10, 25, 50]
    
    fig2 = plot_metrics_comparison(metrics_data, noise_levels, 'PSNR')
    plt.show()
