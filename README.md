# **ISTA vs BM3D 图像去噪对比实验**

## 项目概述

本实验实现了ISTA（Iterative Shrinkage-Thresholding Algorithm）算法，并与当前先进的BM3D（Block-Matching and 3D Filtering）算法进行图像去噪性能对比。实验使用Set14数据集，在多种噪声条件下（高斯噪声和椒盐噪声）评估两种算法的PSNR、SSIM指标和计算时间，并对ISTA的收敛行为进行深入分析。

## 实验目标

1. **理论理解**：深入理解ISTA算法的数学原理、迭代过程和收敛性
2. **算法实现**：实现ISTA算法并进行参数优化
3. **性能对比**：与BM3D算法进行全面的定量和定性对比
4. **分析研究**：分析算法在不同噪声条件下的表现差异和收敛行为

## 项目结构

```
ISTA-BM3D-Comparison/
├── data/                          # 数据集
│   ├── Set14/                    # Set14数据集
│   └── ppt3.png                  # 重点测试图像
├── src/                          # 源代码
│   ├── ista_denoiser.py          # ISTA算法实现
│   ├── bm3d_wrapper.py           # BM3D算法包装器
│   ├── noise_generation.py       # 噪声生成模块
│   ├── evaluation.py             # 评估指标计算
│   ├── visualization.py          # 可视化工具
│   └── main.py                   # 主实验脚本
├── results/                      # 实验结果
│   ├── experiment_fixed/         # 修正实验的完整结果
│   ├── convergence_analysis/     # ISTA收敛性分析
│   └── ppt3_fixed/              # ppt3.png专门处理结果
│   └── full_experiment/
├── scripts/                      # 运行脚本
│   ├── run_full_experiment.py    # 完整实验脚本
│   ├── run_experiment_fixed.py   # 修正实验脚本
│   ├── convergence_analysis.py   # 收敛性分析
│   ├── process_ppt3_fixed.py     # ppt3处理脚本
├── requirements.txt              # Python依赖包
└── README.md                     # 项目说明
```

##  环境要求

### Python版本
- Python 3.7

### 依赖包
```
numpy>=1.21.0
opencv-python>=4.5.0
scikit-image>=0.18.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tqdm>=4.62.0
pandas>=1.3.0
PyWavelets>=1.1.1
scipy>=1.7.0
bm3d>=1.0
```

### 快速安装
```bash
# 克隆项目
git clone https://github.com/linhongqin123/ISTA-BM3D-Comparison.git
cd ISTA-BM3D-Comparison

# 创建虚拟环境（可选）
python -m venv venv

venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

##  数据集准备

### Set14数据集
1. 从Kaggle下载Set14数据集：https://www.kaggle.com/datasets/101dm/set-5-14-super-resolution-dataset/data
1. 将数据集放在 `data/Set14/` 目录下

### 测试图像
将 `ppt3.png` 放在 `data/` 目录下作为重点测试图像

##  快速开始

### 1. 基本功能测试
```bash
# 测试ISTA基本功能
python src/test_ista.py

# 测试BM3D基本功能
python src/test_bm3d_fixed.py
```

### 2. 运行完整实验
```bash
# 运行ISTA和BM3D对比实验
python run_full_experiment.py

# 运行修正版本（推荐）
python run_experiment_fixed.py

# 运行收敛性分析
python convergence_analysis.py

# 处理ppt3.png图像
python process_ppt3_fixed.py

# 生成总结报告
python generate_summary.py
```

### 3. 参数配置
所有实验参数可在相应脚本的 `config` 部分进行调整：
- 噪声水平：高斯噪声σ=[10, 25, 50]，椒盐噪声=[4%, 10%, 20%]
- ISTA参数：λ=[0.01, 0.1, 0.2]，步长=[0.5, 1.0, 1.5]，迭代次数=50
- BM3D参数：使用自动σ估计或手动设置

## 实验结果

### 主要发现

#### 1. 算法性能对比
| 算法 | 平均PSNR (dB) | 平均SSIM      | 平均时间 (秒) |
| ---- | ------------- | ------------- | ------------- |
| ISTA | 15.87 ± 2.27  | 0.438 ± 0.192 | 0.01 ± 0.00   |
| BM3D | 23.51 ± 6.22  | 0.623 ± 0.236 | 2.64 ± 0.54   |

#### 2. 噪声类型影响
- **高斯噪声**：BM3D平均比ISTA高 **11.05 dB**
- **椒盐噪声**：BM3D平均比ISTA高 **4.23 dB**

#### 3. 计算效率
- ISTA比BM3D快约 **260倍**
- ISTA适合实时处理，BM3D适合高质量离线处理

#### 4. ISTA收敛行为
- 最佳步长：**1.0**
- 最佳λ：**0.1**
- 收敛迭代：**20-30次**
- 收敛速度与步长选择密切相关

### 可视化结果
实验结果保存在 `results/` 目录下，包括：
- `experiment_fixed/`：完整的定量结果和对比图表
- `convergence_analysis/`：ISTA收敛曲线分析
- `ppt3_fixed/`：重点图像的视觉对比结果

## 核心算法

### 1. ISTA算法
**数学原理**：
\[
x^{(k+1)} = S_{\lambda t}\left(x^{(k)} - t\nabla f(x^{(k)})\right)
\]
其中 \(S_\tau\) 是软阈值算子：
\[
S_\tau(z) = \text{sign}(z) \cdot \max(|z| - \tau, 0)
\]

**实现特点**：
- 支持图像域和小波域两种实现
- 可调节的正则化参数λ和步长t
- 包含收敛监控和自动停止机制

### 2. BM3D算法
**核心思想**：
1. **块匹配**：查找相似的图像块
2. **3D变换**：将相似块堆叠并进行3D变换
3. **协同滤波**：在变换域进行硬阈值或维纳滤波
4. **逆变换聚合**：恢复图像并进行加权平均

**实现特点**：
- 使用Python bm3d库包装
- 自动噪声水平估计
- 支持多种配置模式

## 实验设置

### 噪声类型
1. **高斯噪声**：标准差σ = {10, 25, 50}
2. **椒盐噪声**：噪声密度 = {4%, 10%, 20%}

### 评估指标
1. **PSNR**（峰值信噪比）：衡量去噪图像的保真度
2. **SSIM**（结构相似性）：衡量结构信息的保持程度
3. **计算时间**：算法运行时间（秒）

### 实验流程
1. 加载Set14数据集图像
2. 为每张图像添加不同噪声
3. 分别用ISTA和BM3D进行去噪
4. 计算PSNR、SSIM和运行时间
5. 统计分析和可视化

## 结果分析与讨论

### 1. 算法优缺点对比

| 特性               | ISTA                | BM3D                       |
| ------------------ | ------------------- | -------------------------- |
| **理论保证**       | 有严格的收敛证明    | 基于经验观察，理论保证有限 |
| **计算效率**       | 高（迭代简单）      | 低（块匹配和3D变换复杂）   |
| **内存需求**       | 低                  | 高（需要存储3D块组）       |
| **参数敏感性**     | 敏感（需调λ和步长） | 相对鲁棒                   |
| **纹理保持**       | 中等                | 优秀                       |
| **边缘保持**       | 好                  | 优秀                       |
| **噪声类型适应性** | 对脉冲噪声较鲁棒    | 对高斯噪声效果最佳         |

### 2. 参数影响分析
- **ISTA步长**：过大导致振荡，过小收敛慢，最佳值≈1.0
- **正则化参数λ**：控制稀疏性强度，过大导致过平滑
- **迭代次数**：20-30次通常足够收敛

### 3. 应用场景建议
- **选择ISTA当**：需要实时处理、计算资源有限、对理论保证有要求
- **选择BM3D当**：追求最佳去噪质量、有充足计算资源、纹理保持重要

