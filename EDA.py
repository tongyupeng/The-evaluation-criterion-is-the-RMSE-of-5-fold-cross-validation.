import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import coo_matrix
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# 加载数据集
ratings = pd.read_csv('./ml-20m/ratings.csv')
movies = pd.read_csv('./ml-20m/movies.csv')
tags = pd.read_csv('./ml-20m/tags.csv')
genome_scores = pd.read_csv('./ml-20m/genome-scores.csv')

# 基本统计
print("Ratings shape:", ratings.shape)
print("Unique users:", ratings['userId'].nunique())
print("Unique movies:", ratings['movieId'].nunique())
print("Total ratings:", len(ratings))
print("Sparsity:", 1 - len(ratings) / (ratings['userId'].nunique() * ratings['movieId'].nunique()))

# ID 映射：将userId和movieId映射到连续整数
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()
ratings['user_idx'] = user_encoder.fit_transform(ratings['userId'])
ratings['movie_idx'] = movie_encoder.fit_transform(ratings['movieId'])

# 构建稀疏矩阵
sparse_ratings = coo_matrix((ratings['rating'], (ratings['user_idx'], ratings['movie_idx'])))
print("Sparse matrix shape:", sparse_ratings.shape)

# 评分分布
plt.figure(figsize=(8, 4))
sns.histplot(ratings['rating'], bins=10, kde=True)
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# 时间戳处理：转换为日期，并计算相对天数
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
min_time = ratings['timestamp'].min()
ratings['days_since_min'] = (ratings['timestamp'] - min_time).dt.days

# 时间趋势：按年分组评分数量
ratings['year'] = ratings['timestamp'].dt.year
yearly_ratings = ratings.groupby('year').size()
plt.figure(figsize=(10, 4))
yearly_ratings.plot(kind='bar')
plt.title('Ratings per Year')
plt.xlabel('Year')
plt.ylabel('Number of Ratings')
plt.show()


# Genome scores 统计：movie与tag相关性分布
plt.figure(figsize=(8, 4))
sns.histplot(genome_scores['relevance'], bins=20, kde=True)
plt.title('Genome Relevance Distribution')
plt.xlabel('Relevance')
plt.ylabel('Count')
plt.show()

# 偏差估计预处理（全局均值、用户/电影偏差）
global_mu = ratings['rating'].mean()
user_bias = ratings.groupby('userId')['rating'].mean() - global_mu
movie_bias = ratings.groupby('movieId')['rating'].mean() - global_mu
ratings['centered_rating'] = ratings['rating'] - global_mu - ratings['userId'].map(user_bias) - ratings['movieId'].map(movie_bias)

# 时间权重示例：alpha=0.001，按天衰减
alpha = 0.001
t_max = ratings['days_since_min'].max()
ratings['time_weight'] = np.exp(-alpha * (t_max - ratings['days_since_min']))

# 保存预处理数据（可选）
ratings.to_csv('preprocessed_ratings.csv', index=False)