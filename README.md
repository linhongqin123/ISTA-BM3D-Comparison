# ISTA-BM3D-Comparison
Comparison study between ISTA and BM3D algorithms for image denoising

```
# ISTA vs BM3D 图像去噪对比实验

## 项目概述
本项目实现ISTA（Iterative Shrinkage-Thresholding Algorithm）算法，并与BM3D算法进行图像去噪性能对比。实验使用Set14数据集，评估两种算法在不同噪声条件下的表现。

## 项目结构
```



ISTA-BM3D-Comparison/
├── data/
│ ├── Set14/ # 下载的Set14数据集
│ └── ppt3.png # 需要重点处理的图像
├── src/
│ ├── ista_denoiser.py # ISTA算法实现
│ ├── bm3d_wrapper.py # BM3D算法包装器
│ ├── noise_generation.py # 噪声生成模块
│ ├── evaluation.py # 评估指标计算
│ ├── visualization.py # 可视化工具
│ └── main.py # 主实验脚本
├── results/
│ ├── quantitative/ # 量化结果表格
│ ├── images/ # 可视化图像
│ └── convergence/ # 收敛分析图
├── report/
│ ├── report.tex # LaTeX双栏报告
│ └── figures/ # 报告图表
├── requirements.txt # Python依赖
├── README.md # 项目说明
└── run_experiments.py # 一键运行脚本

text

```
## 安装依赖
```bash
pip install -r requirements.txt
```



## 数据准备

1. 从Kaggle下载Set14数据集
2. 将数据集放在 `data/Set14/` 目录下
3. 确保包含 `ppt3.png` 图像

## 运行实验

bash

```
# 运行完整实验（所有算法）
python run_experiments.py

# 只运行ISTA算法
python src/main.py --run_ista --max_images 3

# 只运行BM3D算法
python src/main.py --run_bm3d --max_images 3

# 指定数据路径
python src/main.py --data_dir /path/to/data --output_dir /path/to/results
```



## 实验结果

实验结果将保存在 `results/` 目录下：

- `results/images/`: 可视化图像结果
- `results/tables/`: 量化指标表格（CSV格式）
- `results/plots/`: 性能对比图表
- `results/analysis.json`: 完整的分析结果

## 核心算法

### ISTA算法

- 迭代收缩阈值算法
- 用于稀疏信号恢复和图像去噪
- 可调整的正则化参数和迭代次数

### BM3D算法

- 块匹配与3D滤波算法
- 当前最先进的图像去噪方法之一
- 利用图像的非局部自相似性

## 评估指标

- PSNR（峰值信噪比）
- SSIM（结构相似性指数）
- 计算时间

## 作者
