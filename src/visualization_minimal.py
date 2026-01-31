"""
最小化可视化模块 - 避免样式导入问题
"""

import numpy as np
import matplotlib
# 必须在使用pyplot之前设置后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def simple_plot_comparison(original, noisy, denoised, title="Image Comparison"):
    """
    简单的图像对比图
    
    参数:
        original: 原始图像
        noisy: 含噪图像
        denoised: 去噪图像
        title: 标题
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # 设置基本样式参数
    plt.rcParams.update({
        'figure.figsize': (12, 4),
        'font.size': 10
    })
    
    # 显示图像
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(noisy, cmap='gray')
    axes[1].set_title('Noisy')
    axes[1].axis('off')
    
    axes[2].imshow(denoised, cmap='gray')
    axes[2].set_title('Denoised')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig

def simple_plot_convergence(iterations, residuals, title="Convergence"):
    """
    简单的收敛曲线图
    
    参数:
        iterations: 迭代次数列表
        residuals: 残差列表
        title: 标题
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 设置基本样式
    plt.rcParams.update({
        'figure.figsize': (8, 5),
        'font.size': 12
    })
    
    ax.plot(iterations, residuals, 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    return fig

def save_figure_simple(fig, filename, dpi=300):
    """
    保存图形
    
    参数:
        fig: matplotlib图形对象
        filename: 输出文件名
        dpi: 分辨率
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"图形已保存到: {filename}")