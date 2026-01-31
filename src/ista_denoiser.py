"""
ISTA (Iterative Shrinkage-Thresholding Algorithm) 图像去噪实现
包含图像域和小波域两种实现方式
"""

import numpy as np
import pywt
import cv2
from scipy.fftpack import dct, idct
import time
from typing import Tuple, List, Dict, Optional, Union

class ISTADenoiser:
    """
    ISTA去噪器类
    实现基于小波变换和图像域的ISTA算法
    """
    
    def __init__(self, 
                 wavelet_type: str = 'db4',
                 max_iter: int = 100,
                 lambda_: float = 0.1,
                 step_size: float = 1.0,
                 tol: float = 1e-6,
                 verbose: bool = False):
        """
        初始化ISTA去噪器
        
        参数:
            wavelet_type: 小波类型 ('db4', 'haar', 'sym8'等)
            max_iter: 最大迭代次数
            lambda_: 正则化参数
            step_size: 步长（学习率）
            tol: 收敛容差
            verbose: 是否显示迭代信息
        """
        self.wavelet_type = wavelet_type
        self.max_iter = max_iter
        self.lambda_ = lambda_
        self.step_size = step_size
        self.tol = tol
        self.verbose = verbose
        
        # 收敛历史记录
        self.convergence_history = {
            'psnr': [],
            'objective': [],
            'residual': [],
            'iterations': []
        }
    
    @staticmethod
    def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
        """
        软阈值算子
        
        参数:
            x: 输入数组
            threshold: 阈值
            
        返回:
            阈值处理后的数组
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    @staticmethod
    def compute_gradient_image_domain(current_estimate: np.ndarray, 
                                    noisy_image: np.ndarray) -> np.ndarray:
        """
        计算图像域的梯度（用于图像域ISTA）
        
        参数:
            current_estimate: 当前估计值
            noisy_image: 含噪图像
            
        返回:
            梯度
        """
        # 梯度: ∇f(x) = (x - y)
        return current_estimate - noisy_image
    
    @staticmethod
    def wavelet_transform(image: np.ndarray, 
                         wavelet: str = 'db4') -> Tuple[List[np.ndarray], tuple]:
        """
        小波变换
        
        参数:
            image: 输入图像
            wavelet: 小波类型
            
        返回:
            coeffs: 小波系数列表
            shape: 原始图像形状
        """
        # 转换为float32
        image_float = image.astype(np.float32)
        
        # 执行小波变换（2层分解）
        coeffs = pywt.wavedec2(image_float, wavelet, level=2)
        
        return coeffs
    
    @staticmethod
    def inverse_wavelet_transform(coeffs: List[np.ndarray], 
                                wavelet: str = 'db4') -> np.ndarray:
        """
        逆小波变换
        
        参数:
            coeffs: 小波系数列表
            wavelet: 小波类型
            
        返回:
            重建图像
        """
        # 执行逆小波变换
        reconstructed = pywt.waverec2(coeffs, wavelet)
        
        return reconstructed.astype(np.float32)
    
    def ista_image_domain(self, 
                         noisy_image: np.ndarray,
                         lambda_: Optional[float] = None,
                         max_iter: Optional[int] = None,
                         step_size: Optional[float] = None) -> np.ndarray:
        """
        图像域ISTA去噪（恒等变换）
        
        参数:
            noisy_image: 含噪图像
            lambda_: 正则化参数（如果为None则使用self.lambda_）
            max_iter: 最大迭代次数
            step_size: 步长
            
        返回:
            去噪图像
        """
        # 参数设置
        lambda_ = lambda_ if lambda_ is not None else self.lambda_
        max_iter = max_iter if max_iter is not None else self.max_iter
        step_size = step_size if step_size is not None else self.step_size
        
        # 初始化
        x = noisy_image.astype(np.float32).copy()
        y = noisy_image.astype(np.float32).copy()
        
        # 归一化到[0, 1]范围
        if y.max() > 1.0:
            y = y / 255.0
            x = x / 255.0
        
        # 清空历史记录
        self.convergence_history = {
            'psnr': [], 'objective': [], 'residual': [], 'iterations': []
        }
        
        # ISTA迭代
        for k in range(max_iter):
            # 保存上一次迭代结果
            x_old = x.copy()
            
            # 梯度下降步骤
            gradient = self.compute_gradient_image_domain(x, y)
            x = x - step_size * gradient
            
            # 软阈值收缩步骤
            x = self.soft_threshold(x, lambda_ * step_size)
            
            # 计算收敛指标
            residual = np.linalg.norm(x - x_old) / (np.linalg.norm(x_old) + 1e-10)
            
            # 计算目标函数值（仅用于监控）
            data_fidelity = 0.5 * np.linalg.norm(x - y)**2
            regularization = lambda_ * np.linalg.norm(x, ord=1)
            objective_value = data_fidelity + regularization
            
            # 记录历史
            self.convergence_history['iterations'].append(k + 1)
            self.convergence_history['residual'].append(residual)
            self.convergence_history['objective'].append(objective_value)
            
            # 显示进度
            if self.verbose and (k % 10 == 0 or k == max_iter - 1):
                print(f"Iteration {k+1}/{max_iter}: "
                      f"Residual = {residual:.6f}, "
                      f"Objective = {objective_value:.6f}")
            
            # 检查收敛
            if residual < self.tol:
                if self.verbose:
                    print(f"收敛于迭代 {k+1}, 残差: {residual:.6f}")
                break
        
        # 恢复到[0, 255]范围
        x = np.clip(x * 255, 0, 255).astype(np.uint8)
        
        return x
    
    def ista_wavelet_domain(self, 
                           noisy_image: np.ndarray,
                           lambda_: Optional[float] = None,
                           max_iter: Optional[int] = None,
                           step_size: Optional[float] = None,
                           threshold_type: str = 'soft') -> np.ndarray:
        """
        小波域ISTA去噪（推荐使用）
        
        参数:
            noisy_image: 含噪图像
            lambda_: 正则化参数
            max_iter: 最大迭代次数
            step_size: 步长
            threshold_type: 阈值类型 ('soft' 或 'hard')
            
        返回:
            去噪图像
        """
        # 参数设置
        lambda_ = lambda_ if lambda_ is not None else self.lambda_
        max_iter = max_iter if max_iter is not None else self.max_iter
        step_size = step_size if step_size is not None else self.step_size
        
        # 转换为float32
        noisy_float = noisy_image.astype(np.float32)
        if noisy_float.max() > 1.0:
            noisy_float = noisy_float / 255.0
        
        # 小波变换
        coeffs_y = self.wavelet_transform(noisy_float, self.wavelet_type)
        
        # 初始化系数 - 正确处理小波系数结构
        coeffs_x = []
        for c in coeffs_y:
            if isinstance(c, tuple):
                # 对于元组（细节系数），复制每个方向的系数
                coeffs_x.append(tuple(sub_c.copy() for sub_c in c))
            else:
                # 对于数组（近似系数），直接复制
                coeffs_x.append(c.copy())
        
        # 清空历史记录
        self.convergence_history = {
            'psnr': [], 'objective': [], 'residual': [], 'iterations': []
        }
        
        # ISTA迭代
        for k in range(max_iter):
            # 保存上一次系数 - 深拷贝
            coeffs_x_old = []
            for c in coeffs_x:
                if isinstance(c, tuple):
                    coeffs_x_old.append(tuple(sub_c.copy() for sub_c in c))
                else:
                    coeffs_x_old.append(c.copy())
            
            # 对每个子带进行处理
            for i in range(len(coeffs_x)):
                if i == 0:  # 近似系数（低频），通常不进行阈值处理或使用较小的lambda
                    # 梯度下降步骤
                    gradient = coeffs_x[i] - coeffs_y[i]
                    coeffs_x[i] = coeffs_x[i] - step_size * gradient
                else:  # 细节系数（高频）
                    # 对每个方向的细节系数进行梯度下降
                    for j in range(3):  # 水平、垂直、对角线
                        if isinstance(coeffs_x[i], tuple):
                            # 梯度下降步骤
                            gradient = coeffs_x[i][j] - coeffs_y[i][j]
                            coeffs_x[i][j] = coeffs_x[i][j] - step_size * gradient
                            
                            # 软阈值收缩
                            if threshold_type == 'soft':
                                coeffs_x[i][j] = self.soft_threshold(
                                    coeffs_x[i][j], lambda_ * step_size
                                )
                            else:  # 硬阈值
                                coeffs_x[i][j] = coeffs_x[i][j] * (
                                    np.abs(coeffs_x[i][j]) > lambda_ * step_size
                                ).astype(np.float32)
            
            # 计算残差（系数变化）
            residual = 0
            for i in range(len(coeffs_x)):
                if i == 0:
                    residual += np.linalg.norm(coeffs_x[i] - coeffs_x_old[i])
                else:
                    for j in range(3):
                        residual += np.linalg.norm(coeffs_x[i][j] - coeffs_x_old[i][j])
            
            # 记录历史
            self.convergence_history['iterations'].append(k + 1)
            self.convergence_history['residual'].append(residual)
            
            # 显示进度
            if self.verbose and (k % 10 == 0 or k == max_iter - 1):
                print(f"Iteration {k+1}/{max_iter}: Residual = {residual:.6f}")
            
            # 检查收敛
            if residual < self.tol:
                if self.verbose:
                    print(f"收敛于迭代 {k+1}, 残差: {residual:.6f}")
                break
        
        # 逆小波变换
        denoised_float = self.inverse_wavelet_transform(coeffs_x, self.wavelet_type)
        
        # 恢复到[0, 255]范围并裁剪
        denoised_float = np.clip(denoised_float, 0, 1)
        denoised = (denoised_float * 255).astype(np.uint8)
        
        return denoised
    
    def ista_dct_domain(self, 
                       noisy_image: np.ndarray,
                       lambda_: Optional[float] = None,
                       max_iter: Optional[int] = None,
                       step_size: Optional[float] = None,
                       block_size: int = 8) -> np.ndarray:
        """
        DCT域ISTA去噪（块处理）
        
        参数:
            noisy_image: 含噪图像
            lambda_: 正则化参数
            max_iter: 最大迭代次数
            step_size: 步长
            block_size: DCT块大小
            
        返回:
            去噪图像
        """
        # 参数设置
        lambda_ = lambda_ if lambda_ is not None else self.lambda_
        max_iter = max_iter if max_iter is not None else self.max_iter
        step_size = step_size if step_size is not None else self.step_size
        
        # 转换为float32
        noisy_float = noisy_image.astype(np.float32)
        if noisy_float.max() > 1.0:
            noisy_float = noisy_float / 255.0
        
        # 获取图像尺寸
        h, w = noisy_float.shape
        
        # 初始化
        x = noisy_float.copy()
        y = noisy_float.copy()
        
        # 分块处理
        denoised_blocks = []
        
        for i in range(0, h, block_size):
            row_blocks = []
            for j in range(0, w, block_size):
                # 提取块
                block = y[i:i+block_size, j:j+block_size]
                if block.shape != (block_size, block_size):
                    # 边缘填充
                    pad_h = block_size - block.shape[0]
                    pad_w = block_size - block.shape[1]
                    block = np.pad(block, ((0, pad_h), (0, pad_w)), mode='edge')
                
                # DCT变换
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                
                # ISTA迭代处理该块
                for _ in range(max_iter):
                    # 梯度下降（在DCT域）
                    dct_block = dct_block - step_size * (dct_block - dct_block)
                    
                    # 软阈值
                    dct_block = self.soft_threshold(dct_block, lambda_ * step_size)
                
                # 逆DCT
                denoised_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
                
                # 裁剪回原始大小
                if block.shape != (block_size, block_size):
                    denoised_block = denoised_block[:block_size-pad_h, :block_size-pad_w]
                
                row_blocks.append(denoised_block)
            
            # 拼接行
            denoised_row = np.concatenate(row_blocks, axis=1)
            denoised_blocks.append(denoised_row)
        
        # 拼接所有行
        denoised_float = np.concatenate(denoised_blocks, axis=0)
        
        # 裁剪到原始大小
        denoised_float = denoised_float[:h, :w]
        
        # 恢复到[0, 255]范围
        denoised_float = np.clip(denoised_float, 0, 1)
        denoised = (denoised_float * 255).astype(np.uint8)
        
        return denoised
    
    def denoise(self, 
               noisy_image: np.ndarray,
               method: str = 'wavelet',
               **kwargs) -> np.ndarray:
        """
        主去噪函数
        
        参数:
            noisy_image: 含噪图像
            method: 方法类型 ('wavelet', 'image', 'dct')
            **kwargs: 传递给具体方法的参数
            
        返回:
            去噪图像
        """
        start_time = time.time()
        
        if method == 'wavelet':
            denoised = self.ista_wavelet_domain(noisy_image, **kwargs)
        elif method == 'image':
            denoised = self.ista_image_domain(noisy_image, **kwargs)
        elif method == 'dct':
            denoised = self.ista_dct_domain(noisy_image, **kwargs)
        else:
            raise ValueError(f"未知的方法: {method}")
        
        elapsed = time.time() - start_time
        
        if self.verbose:
            print(f"去噪完成，耗时: {elapsed:.2f} 秒")
            print(f"最终迭代次数: {len(self.convergence_history['iterations'])}")
        
        return denoised
    
    def get_convergence_history(self) -> Dict:
        """
        获取收敛历史
        
        返回:
            收敛历史字典
        """
        return self.convergence_history
    
    def plot_convergence(self, save_path: Optional[str] = None):
        """
        绘制收敛曲线
        
        参数:
            save_path: 保存路径（如果为None则显示）
        """
        if not self.convergence_history['iterations']:
            print("没有收敛历史数据")
            return
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            
            # 设置非交互式后端
            matplotlib.use('Agg')
            
            # 动态导入我们的visualization模块
            from .visualization import setup_plot_style
            setup_plot_style()
            
        except ImportError as e:
            print(f"无法导入matplotlib: {e}")
            print("跳过收敛曲线绘制")
            return
        
        iterations = self.convergence_history['iterations']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 1. 残差收敛
        if self.convergence_history['residual']:
            axes[0].plot(iterations, self.convergence_history['residual'], 'b-', linewidth=2)
            axes[0].set_xlabel('迭代次数')
            axes[0].set_ylabel('残差')
            axes[0].set_title('残差收敛曲线')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_yscale('log')
        
        # 2. 目标函数收敛
        if self.convergence_history['objective']:
            axes[1].plot(iterations, self.convergence_history['objective'], 'r-', linewidth=2)
            axes[1].set_xlabel('迭代次数')
            axes[1].set_ylabel('目标函数值')
            axes[1].set_title('目标函数收敛曲线')
            axes[1].grid(True, alpha=0.3)
        
        # 3. PSNR收敛
        if self.convergence_history['psnr']:
            axes[2].plot(iterations, self.convergence_history['psnr'], 'g-', linewidth=2)
            axes[2].set_xlabel('迭代次数')
            axes[2].set_ylabel('PSNR (dB)')
            axes[2].set_title('PSNR收敛曲线')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"收敛曲线已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close(fig)

def ista_denoise(noisy_img: np.ndarray,
                lambda_: float = 0.1,
                n_iter: int = 50,
                step_size: float = 1.0,
                method: str = 'wavelet',
                verbose: bool = True) -> np.ndarray:
    """
    简化的ISTA去噪函数（方便调用）
    
    参数:
        noisy_img: 含噪图像
        lambda_: 正则化参数
        n_iter: 迭代次数
        step_size: 步长
        method: 方法 ('wavelet', 'image', 'dct')
        verbose: 是否显示信息
        
    返回:
        去噪图像
    """
    denoiser = ISTADenoiser(
        max_iter=n_iter,
        lambda_=lambda_,
        step_size=step_size,
        verbose=verbose
    )
    
    return denoiser.denoise(noisy_img, method=method)

def parameter_sweep(original_image: np.ndarray,
                   noisy_image: np.ndarray,
                   lambda_range: List[float] = None,
                   step_size_range: List[float] = None,
                   method: str = 'wavelet') -> Dict:
    """
    参数扫描实验
    
    参数:
        original_image: 原始图像
        noisy_image: 含噪图像
        lambda_range: lambda参数范围
        step_size_range: 步长范围
        method: 方法类型
        
    返回:
        最佳参数和结果
    """
    if lambda_range is None:
        lambda_range = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    if step_size_range is None:
        step_size_range = [0.1, 0.5, 1.0, 1.5, 2.0]
    
    results = []
    
    from src.evaluation import calculate_psnr, calculate_ssim
    
    print("开始参数扫描实验...")
    print(f"lambda范围: {lambda_range}")
    print(f"步长范围: {step_size_range}")
    
    best_psnr = 0
    best_params = None
    best_denoised = None
    
    for lambda_ in lambda_range:
        for step_size in step_size_range:
            print(f"测试 lambda={lambda_:.3f}, step_size={step_size:.3f}...")
            
            # 使用ISTA去噪
            denoiser = ISTADenoiser(
                lambda_=lambda_,
                step_size=step_size,
                max_iter=50,
                verbose=False
            )
            
            denoised = denoiser.denoise(noisy_image, method=method)
            
            # 计算指标
            psnr = calculate_psnr(original_image, denoised)
            ssim = calculate_ssim(original_image, denoised)
            
            # 记录结果
            result = {
                'lambda': lambda_,
                'step_size': step_size,
                'psnr': psnr,
                'ssim': ssim,
                'iterations': len(denoiser.convergence_history['iterations'])
            }
            results.append(result)
            
            print(f"  PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
            
            # 更新最佳结果
            if psnr > best_psnr:
                best_psnr = psnr
                best_params = {'lambda': lambda_, 'step_size': step_size}
                best_denoised = denoised
    
    # 打印最佳结果
    print(f"\n最佳参数:")
    print(f"  lambda: {best_params['lambda']:.3f}")
    print(f"  步长: {best_params['step_size']:.3f}")
    print(f"  PSNR: {best_psnr:.2f} dB")
    
    return {
        'best_params': best_params,
        'best_psnr': best_psnr,
        'best_denoised': best_denoised,
        'all_results': results
    }

if __name__ == "__main__":
    # 测试代码
    print("测试ISTA去噪算法...")
    
    # 创建测试图像
    test_image = np.random.rand(128, 128) * 255
    test_image = test_image.astype(np.uint8)
    
    # 添加高斯噪声
    noise = np.random.randn(128, 128) * 25
    noisy = test_image + noise.astype(np.uint8)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    print(f"原始图像形状: {test_image.shape}")
    print(f"含噪图像形状: {noisy.shape}")
    
    # 测试小波域ISTA
    print("\n测试小波域ISTA...")
    denoiser = ISTADenoiser(
        wavelet_type='db4',
        max_iter=30,
        lambda_=0.1,
        step_size=1.0,
        verbose=True
    )
    
    denoised = denoiser.denoise(noisy, method='wavelet')
    print(f"去噪图像形状: {denoised.shape}")
    
    # 绘制收敛曲线
    denoiser.plot_convergence()