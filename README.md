# 🎬 MovieLens 20M 矩阵补全：凸优化 vs 非凸优化

## 📊 项目简介

本项目基于 **MovieLens 20M 数据集**（2000万条电影评分），系统对比了**凸优化方法（核范数正则化）**与**非凸优化方法（扩展矩阵分解）**在低秩矩阵补全任务上的性能差异。实验涵盖完整的数据分析、预处理、模型实现与5折交叉验证流程，为推荐系统研究提供可复现的基准对比。

## 📥 数据下载

本项目使用 **MovieLens 20M** 数据集：
- **官方地址**: https://grouplens.org/datasets/movielens/20m/
- **文件大小**: 约 200 MB（解压后）
- **包含文件**: 
  - `ratings.csv` - 20,000,263 条评分
  - `movies.csv` - 27,278 部电影信息
  - `tags.csv` - 465,564 条标签
  - `genome-scores.csv` - 基因组合分
  - `genome-tags.csv` - 基因组标签

### 快速获取数据
```bash
# 下载数据集（推荐直接到官网下载）
wget http://files.grouplens.org/datasets/movielens/ml-20m.zip

# 解压
unzip ml-20m.zip
mv ml-20m data/
```

## 🎯 核心对比

### ⚖️ 方法对比
| 维度 | 凸优化 (Soft-Impute) | 非凸优化 (Biased MF + ALS) |
|------|-------------------|--------------------------|
| **正则化方式** | 核范数正则化 | Frobenius范数 + 标签敏感度正则 |
| **秩控制** | 自动软阈值收缩 | 固定隐因子维度(k=20) |
| **时间建模** | 时间权重(预留) | 显式时间动态偏置(5年份桶) |
| **语义建模** | 无 | 可学习标签热度偏置 |
| **优化保证** | 全局最优收敛 | 局部最优，实用性能强 |

## 🔬 实验流程

### 📋 完整实验流水线
```
1. 数据下载 → 2. EDA分析 → 3. 预处理 → 4. 凸优化实验 → 5. 非凸优化实验 → 6. 结果对比
```

### 1. 📊 探索性数据分析 (`EDA.py`)
- 数据统计：20,000,263条评分，138,493用户，27,278电影，稀疏度99.47%
- 评分分布可视化与特征分析
- 基因组标签相关性分析

### 2. 🔧 数据预处理
- 连续ID映射（user_idx, movie_idx）
- 偏差中心化：`centered_rating = rating - μ - b_u - b_i`
- 时间特征工程：时间权重 `exp(-0.001 × 时间差)`
- 标签热度矩阵构建（Top-100高频标签）

### 3. ⚙️ 凸优化实现 (`convex.py`)
```python
# 核心优化目标
min_Z 0.5 * Σ w_ui (r_ui_centered - Z_ui)² + λ ||Z||_*
```
- 加权Soft-Impute算法
- 线性算子优化，避免显式稠密矩阵
- 部分SVD加速（k≤50）
- 自动秩选择与软阈值处理

### 4. 🧠 非凸优化实现 (`nonconvex.py`)
```python
# 完整预测公式
r̂_ui = μ + b_u + b_i + b_{i,t} + p_u^T q_i + γ·(H_year · s_i)
```
- 扩展偏置矩阵分解
- ALS交替最小二乘法求解
- 时间动态偏置（5个年份桶）
- 可学习标签敏感度向量

## ⚡ 性能优化

### 🚀 工程亮点
- **内存管理**：稀疏存储 + LinearOperator + 主动垃圾回收
- **计算加速**：向量化einsum、CSR行列快速访问、预计算标签偏置表
- **数值稳定性**：SVD退化回退、预测值裁剪[0.5, 5.0]、正则化防过拟合
- **监控调试**：多层tqdm进度条、详细迭代日志、提前停止机制

### 📈 实验设置
- **评估指标**：5折交叉验证RMSE
- **超参数搜索**：
  - 凸方法：λ ∈ [100, 50, 10]
  - 非凸方法：λ ∈ [10.0, 1.0]，标签正则λ_s=1.0
- **收敛标准**：拟合损失变化<1e-4或patience=3

## 🛠️ 快速开始

### 环境要求
```bash
Python 3.8+
# 推荐使用虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行完整实验
```bash
# 1. 下载数据（手动下载并解压到data/文件夹）
# 2. 运行EDA分析
python EDA.py

# 3. 运行凸优化实验
python convex.py

# 4. 运行非凸优化实验
python nonconvex.py

# 5. 查看结果（自动保存在results/文件夹）
```

## 📁 项目结构
```
movie-matrix-completion/
├── EDA.py              # 探索性数据分析与预处理
├── convex.py           # 凸优化方法实现
├── nonconvex.py        # 非凸优化方法实现
├── data/               # MovieLens 20M数据集（需自行下载）
│   ├── ml-20m/        # 解压后的原始数据
│   └── processed/     # 预处理后的数据
├── results/            # 实验结果与图表
│   ├── figures/       # 可视化图表
│   └── logs/          # 实验日志
├── requirements.txt    # Python依赖包
├── .gitignore         # Git忽略配置
├── LICENSE            # MIT许可证
└── README.md          # 项目说明
```

## 📊 预期结果

### 性能对比
| 方法 | 验证RMSE | 训练时间 | 内存占用 | 可解释性 |
|------|----------|----------|----------|----------|
| 凸优化 | ~0.85-0.90 | 较长 | 中等 | 中等 |
| 非凸优化 | ~0.80-0.85 | 较短 | 较低 | 高 |

### 核心发现
1. **凸方法优势**：理论保证强，适合作为鲁棒性基线
2. **非凸方法优势**：多层次偏置建模能更好捕捉时间动态与语义信息
3. **实践建议**：实际推荐系统中，非凸方法在合理计算成本下通常表现更优

## 🔮 扩展方向

### 短期改进
1. ✅ 在凸方法中显式启用时间加权损失
2. ✅ 引入更多侧信息（用户画像、电影内容特征）
3. ✅ 扩展为深度矩阵分解模型

### 长期规划
1. 🔄 实时推荐服务API部署
2. 🔄 分布式计算支持（Spark/Dask）
3. 🔄 在线学习与增量更新

## 👥 适用场景

- 🎓 **学术研究**：矩阵补全算法对比
- 🏢 **工业实践**：推荐系统基线模型
- 📚 **教学案例**：凸优化与非凸优化实践
- 🔬 **算法竞赛**：个性化推荐解决方案

## 📄 引用与参考

### 数据集引用
```bibtex
@article{harper2015movielens,
  title={The MovieLens Datasets: History and Context},
  author={Harper, F. Maxwell and Konstan, Joseph A.},
  journal={ACM Transactions on Interactive Intelligent Systems},
  year={2015}
}
```

### 核心算法参考
1. Mazumder et al. "Spectral Regularization Algorithms for Learning Large Incomplete Matrices" (Soft-Impute)
2. Koren et al. "Matrix Factorization Techniques for Recommender Systems"
3. Koren "Collaborative Filtering with Temporal Dynamics"

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：
1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📧 联系与支持

- 🐛 **问题反馈**：[GitHub Issues](https://github.com/yourusername/repo/issues)
- 💡 **功能建议**：欢迎提交Issue讨论
- 📚 **使用疑问**：查看Wiki或提交Question

---

**⭐ 如果本项目对你有帮助，请点个Star支持！**

**📅 最后更新**：2024年1月  
**🔖 版本**：1.0.0  
**📄 许可证**：MIT License - 详见 [LICENSE](LICENSE) 文件

> 💡 **提示**：运行前请确保已下载MovieLens 20M数据集到data/文件夹！
