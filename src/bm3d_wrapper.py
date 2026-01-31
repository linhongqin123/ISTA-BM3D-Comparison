"""
修复的BM3D包装器
"""

import numpy as np
import bm3d
import cv2
import time
from typing import Optional

class BM3DDenoiserFixed:
    """
    修复的BM3D去噪器
    正确处理图像范围和噪声水平
    """
    
    def __init__(self, 
                 sigma_psd: float = 25,
                 stage_arg: bool = True,
                 profile: str = 'np',
                 verbose: bool = False):
        """
        初始化BM3D去噪器
        
        参数:
            sigma_psd: 噪声标准差（注意：这个值应该是相对于255的，比如25表示sigma=25）
            stage_arg: 是否使用两阶段
            profile: 配置类型
            verbose: 是否显示信息
        """
        # 注意：BM3D库期望sigma在[0,1]范围内，所以我们需要将sigma_psd除以255
        self.sigma_input = sigma_psd  # 保存输入值用于显示
        self.sigma_normalized = sigma_psd / 255.0  # 归一化的sigma
        self.stage_arg = stage_arg
        self.profile = profile
        self.verbose = verbose
        self.processing_time = 0.0
    
    def denoise(self, 
               noisy_image: np.ndarray,
               sigma_psd: Optional[float] = None) -> np.ndarray:
        """
        BM3D去噪（修复版本）
        
        参数:
            noisy_image: 含噪图像 (0-255范围)
            sigma_psd: 噪声标准差（如果为None则使用self.sigma_input）
            
        返回:
            去噪图像 (0-255范围)
        """
        if sigma_psd is not None:
            sigma_normalized = sigma_psd / 255.0
        else:
            sigma_normalized = self.sigma_normalized
        
        # 确保图像是float32类型且在[0, 1]范围
        if noisy_image.dtype != np.float32:
            noisy_image_float = noisy_image.astype(np.float32)
        else:
            noisy_image_float = noisy_image.copy()
        
        # 归一化到[0,1]
        if noisy_image_float.max() > 1.0:
            noisy_image_float = noisy_image_float / 255.0
        
        # 检查图像范围
        if self.verbose:
            print(f"输入图像范围: [{noisy_image_float.min():.3f}, {noisy_image_float.max():.3f}]")
            print(f"BM3D sigma: {sigma_normalized:.4f} (对应原始sigma={sigma_normalized*255:.1f})")
        
        # 记录开始时间
        start_time = time.time()
        
        # 调用BM3D
        try:
            if self.stage_arg:
                denoised_float = bm3d.bm3d(noisy_image_float, sigma_normalized, 
                                         stage_arg=bm3d.BM3DStages.ALL_STAGES,
                                         profile=self.profile)
            else:
                denoised_float = bm3d.bm3d(noisy_image_float, sigma_normalized,
                                         stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING,
                                         profile=self.profile)
        except Exception as e:
            if self.verbose:
                print(f"BM3D调用失败: {e}")
            # 尝试使用更简单的profile
            try:
                denoised_float = bm3d.bm3d(noisy_image_float, sigma_normalized, profile='np')
            except Exception as e2:
                if self.verbose:
                    print(f"BM3D再次失败: {e2}")
                # 返回原始图像作为降级方案
                denoised_float = noisy_image_float.copy()
        
        # 记录处理时间
        self.processing_time = time.time() - start_time
        
        if self.verbose:
            print(f"BM3D输出范围: [{denoised_float.min():.3f}, {denoised_float.max():.3f}]")
            print(f"去噪完成，耗时: {self.processing_time:.2f} 秒")
        
        # 转换回[0, 255]范围
        denoised_float = np.clip(denoised_float, 0, 1)
        denoised = (denoised_float * 255).astype(np.uint8)
        
        return denoised
    
    def denoise_with_auto_sigma(self, noisy_image: np.ndarray) -> np.ndarray:
        """
        自动估计噪声水平的BM3D去噪
        
        参数:
            noisy_image: 含噪图像
            
        返回:
            去噪图像
        """
        # 简单估计噪声水平
        # 将图像转换为float32
        noisy_float = noisy_image.astype(np.float32) / 255.0
        
        # 使用图像块估计噪声水平
        # 简单方法：计算高频成分的标准差
        from scipy import ndimage
        
        # 应用高通滤波器
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]]) / 8.0
        
        high_freq = ndimage.convolve(noisy_float, kernel)
        
        # 估计噪声标准差
        estimated_sigma = np.std(high_freq)
        
        if self.verbose:
            print(f"自动估计的sigma: {estimated_sigma:.4f} (×255={estimated_sigma*255:.1f})")
        
        # 使用估计的sigma进行去噪
        return self.denoise(noisy_image, sigma_psd=estimated_sigma * 255)
    
    def get_processing_time(self) -> float:
        """获取处理时间"""
        return self.processing_time

def bm3d_denoise_fixed(noisy_image: np.ndarray,
                      sigma_psd: float = 25,
                      verbose: bool = True) -> np.ndarray:
    """
    修复的BM3D去噪函数
    """
    denoiser = BM3DDenoiserFixed(sigma_psd=sigma_psd, verbose=verbose)
    return denoiser.denoise(noisy_image)

def test_fixed_bm3d():
    """测试修复的BM3D"""
    print("测试修复的BM3D...")
    
    # 创建测试图像
    np.random.seed(42)
    original = np.random.rand(256, 256) * 255
    original = original.astype(np.uint8)
    
    # 添加高斯噪声
    noise = np.random.randn(256, 256) * 25
    noisy = original + noise.astype(np.uint8)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    print(f"原始图像: {original.shape}")
    print(f"含噪图像: {noisy.shape}")
    
    # 测试修复的BM3D
    denoiser = BM3DDenoiserFixed(sigma_psd=25, verbose=True)
    denoised = denoiser.denoise(noisy)
    
    from src.evaluation import calculate_psnr
    psnr = calculate_psnr(original, denoised)
    print(f"修复BM3D PSNR: {psnr:.2f} dB")
    
    return original, noisy, denoised

if __name__ == "__main__":
    original, noisy, denoised = test_fixed_bm3d()
    print("\n修复BM3D测试完成!")