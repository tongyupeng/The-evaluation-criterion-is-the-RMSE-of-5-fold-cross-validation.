import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.model_selection import KFold
from math import sqrt
import time
import warnings
import gc
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==============================================================================
# 全局常量配置
# ==============================================================================
K_LATENT = 20          # 隐因子维度
MAX_ITER = 10          # ALS 最大迭代次数
N_BINS = 5             # 时间分桶数量（用于动态偏置）
GAMMA = 0.05           # 标签热度偏置权重 γ
LAMBDA_S = 1.0          # S_matrix 正则化强度
TOP_N_TAGS = 100       # 保留最高频的标签数量（降维）
RATING_FILE = 'preprocessed_ratings.csv'
TAGS_FILE = './ml-20m/tags.csv'

# ==============================================================================
# 数据加载与预处理
# ==============================================================================
print("正在加载预处理后的评分数据...")
ratings = pd.read_csv(
    RATING_FILE,
    usecols=['user_idx', 'movie_idx', 'centered_rating', 'time_weight', 'rating', 'year'],
    dtype={
        'user_idx': 'int32',
        'movie_idx': 'int32',
        'centered_rating': 'float32',
        'time_weight': 'float32',
        'rating': 'float32',
        'year': 'int32'
    }
)

rows = ratings['user_idx'].values
cols = ratings['movie_idx'].values
num_users = int(rows.max()) + 1
num_movies = int(cols.max()) + 1
n_samples = len(rows)
global_mean = ratings['rating'].mean()

print(f"用户数: {num_users:,} | 电影数: {num_movies:,} | 评分总数: {n_samples:,} | 全局均值: {global_mean:.4f}")

ratings_raw = ratings['rating'].values.astype(np.float32)
years = ratings['year'].values.astype(np.int32)

# 将评分年份分为 N_BINS 个时间桶（用于时间动态偏置）
year_bins = np.digitize(years, bins=np.linspace(years.min(), years.max(), N_BINS + 1)) - 1
year_bins = np.clip(year_bins, 0, N_BINS - 1).astype(np.int8)

# ==============================================================================
# 标签热度特征构建（Tag Heat + 电影-标签敏感度 S）
# ==============================================================================
print(f"\n处理标签热度特征 - 降维至 Top {TOP_N_TAGS} 个最频繁标签...")

# 构建 movieId → movie_idx 的映射（基于原始 ratings.csv 中的电影顺序）
original_ratings = pd.read_csv('./ml-20m/ratings.csv', usecols=['movieId'])
unique_movie_ids = np.sort(original_ratings['movieId'].unique())
movie_id_to_idx = {mid: idx for idx, mid in enumerate(unique_movie_ids)}

# 加载标签数据并提取年份
tags_df = pd.read_csv(TAGS_FILE, usecols=['movieId', 'tag', 'timestamp'])
tags_df['year'] = pd.to_datetime(tags_df['timestamp'], unit='s').dt.year
tags_df = tags_df.drop(columns=['timestamp'])

# 只保留出现在评分记录中的电影
tags_df = tags_df[tags_df['movieId'].isin(movie_id_to_idx)]
tags_df['movie_idx'] = tags_df['movieId'].map(movie_id_to_idx)

# 选取出现频率最高的前 TOP_N_TAGS 个标签
top_tags = tags_df['tag'].value_counts().nlargest(TOP_N_TAGS).index
tags_df = tags_df[tags_df['tag'].isin(top_tags)]

# 构建年份-标签热度矩阵 H（每行归一化，表示该年份标签分布）
tag_heat_matrix = (
    tags_df.groupby(['year', 'tag']).size()
    .unstack(fill_value=0)
    .astype('float32')
)
tag_heat_matrix = tag_heat_matrix.div(tag_heat_matrix.sum(axis=1), axis=0).fillna(0)
num_tags = tag_heat_matrix.shape[1]
print(f"年份-标签热度矩阵 H 形状: {tag_heat_matrix.shape} (年份 × 标签)")

if num_tags == 0:
    warnings.warn("未找到有效标签，标签热度偏置项将被忽略。")

# 构建电影-标签相关度矩阵 S（每行归一化，表示该电影的标签倾向）
tag_to_idx = {tag: i for i, tag in enumerate(tag_heat_matrix.columns)}
tags_df['tag_idx'] = tags_df['tag'].map(tag_to_idx)

movie_tag_counts = tags_df.groupby(['movie_idx', 'tag_idx']).size().reset_index(name='count')

S_sparse = csr_matrix(
    (movie_tag_counts['count'].astype('float32'),
     (movie_tag_counts['movie_idx'], movie_tag_counts['tag_idx'])),
    shape=(num_movies, num_tags)
)

S_matrix = S_sparse.toarray().astype('float32')
S_matrix = np.log1p(S_matrix)                    # log平滑，避免极端值主导
row_sums = S_matrix.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
S_matrix = S_matrix / row_sums                   # 归一化为概率分布

print(f"电影-标签敏感度矩阵 S 形状: {S_matrix.shape} (电影 × 标签)")

# ==============================================================================
# 核心函数：更新电影对标签的敏感度向量 S_matrix
# ==============================================================================
def update_S_matrix(R_T, orig_order_idx, train_years, train_year_bins,
                    P, Q, user_bias, movie_bias, movie_time_bias,
                    tag_heat_matrix, S_matrix, gamma, lambda_s, active_movies,
                    progress_callback=None):
    """
    交替更新 S_matrix：每部电影对各个标签的敏感度向量
    支持可选的进度回调（用于 tqdm 进度条）
    """
    reg_s = lambda_s * np.eye(num_tags, dtype=np.float32)
    updates = []

    for i in active_movies:
        idx_start = R_T.indptr[i]
        idx_end = R_T.indptr[i + 1]

        # 空电影（当前 fold 无评分）：跳过更新，保持原值
        if idx_start == idx_end:
            if progress_callback:
                progress_callback()
            continue

        users = R_T.indices[idx_start:idx_end]
        r_ui = R_T.data[idx_start:idx_end]

        # 通过原始顺序索引准确获取对应的年份和时间桶
        data_indices = orig_order_idx[idx_start:idx_end]
        rating_years = train_years[data_indices]
        rating_bins = train_year_bins[data_indices]

        # 计算不含标签偏置的基础预测
        base_pred = (global_mean +
                     user_bias[users] +
                     movie_bias[i] +
                     movie_time_bias[i, rating_bins] +
                     np.sum(P[users] * Q[i], axis=1))

        # 残差（移除其他项，留作求解 S[i]）
        y = r_ui - base_pred

        # 构建该电影所有评分的年份标签热度矩阵 H_y (n_obs × num_tags)
        H_y = np.array([
            tag_heat_matrix.loc[year].values if year in tag_heat_matrix.index
            else np.zeros(num_tags, dtype=np.float32)
            for year in rating_years
        ], dtype=np.float32)

        # 最小二乘求解：S[i] = (γ H_y^T H_y + λ_s I)^(-1) (γ H_y^T y)
        XtX = gamma * (H_y.T @ H_y) + reg_s
        Xty = gamma * (H_y.T @ y)
        try:
            new_s = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            new_s = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

        updates.append((i, new_s))

        if progress_callback:
            progress_callback()

    # 批量应用所有更新
    for i, new_s in updates:
        S_matrix[i] = new_s

    return S_matrix


# ==============================================================================
# 用户因子与偏置更新
# ==============================================================================
def update_user_factor(u, R, year_indices, year_bins, orig_order_idx,
                       P, Q, user_bias, movie_bias, movie_time_bias,
                       precomputed_tag_bias, gamma, reg_matrix, k):
    """更新单个用户的隐因子 P[u] 和偏置 b_u（使用预计算的标签偏置）"""
    idx_start = R.indptr[u]
    idx_end = R.indptr[u + 1]
    if idx_start == idx_end:
        return None

    items = R.indices[idx_start:idx_end]
    r_ui = R.data[idx_start:idx_end]

    data_indices = orig_order_idx[idx_start:idx_end]
    curr_year_idx = year_indices[data_indices]
    rating_bins = year_bins[data_indices]

    # 直接查表获取标签热度偏置（已乘 γ）
    tag_bias = precomputed_tag_bias[curr_year_idx, items]

    base_pred = (global_mean +
                 movie_bias[items] +
                 movie_time_bias[items, rating_bins] +
                 tag_bias)

    y = r_ui - base_pred

    Q_i = Q[items]
    XtX = Q_i.T @ Q_i + reg_matrix
    Xty = Q_i.T @ y
    try:
        new_p = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        new_p = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

    pred_mf = np.sum(new_p * Q_i, axis=1)
    pred_full = base_pred + pred_mf
    new_user_bias = np.mean(r_ui - pred_full)

    return u, new_p, new_user_bias


# ==============================================================================
# 单折 ALS 训练主函数
# ==============================================================================
def als_train_fold(train_rows, train_cols, train_ratings, train_year_bins, train_years,
                   S_matrix_init, k=20, lambda_reg=10.0, max_iter=10, verbose=True):
    """执行单折 ALS 训练，返回训练好的模型参数"""
    if verbose:
        print(f"开始 ALS 训练 | k={k}, λ={lambda_reg}, max_iter={max_iter}")

    # 年份映射与标签偏置预计算
    all_years = np.sort(np.unique(np.concatenate([train_years, tag_heat_matrix.index.values])))
    year_to_idx = {y: i for i, y in enumerate(all_years)}
    train_year_indices = np.array([year_to_idx[y] for y in train_years], dtype=np.int32)

    H_full = np.zeros((len(all_years), num_tags), dtype=np.float32)
    for year, idx in year_to_idx.items():
        if year in tag_heat_matrix.index:
            H_full[idx] = tag_heat_matrix.loc[year].values

    precomputed_tag_bias = GAMMA * np.dot(H_full, S_matrix_init.T)

    # 构建稀疏矩阵（COO 保留原始顺序，确保年份对齐准确）
    R_coo = coo_matrix((train_ratings, (train_rows, train_cols)), shape=(num_users, num_movies))
    R = R_coo.tocsr()
    R_T = R.transpose().tocsr()
    orig_order_idx = np.arange(len(train_ratings), dtype=np.int64)

    active_users = np.where(np.diff(R.indptr) > 0)[0]
    active_movies = np.where(np.diff(R_T.indptr) > 0)[0]

    # 参数初始化
    P = np.random.randn(num_users, k).astype(np.float32) * 0.01
    Q = np.random.randn(num_movies, k).astype(np.float32) * 0.01
    user_bias = np.zeros(num_users, dtype=np.float32)
    movie_bias = np.zeros(num_movies, dtype=np.float32)
    movie_time_bias = np.zeros((num_movies, N_BINS), dtype=np.float32)
    reg_matrix = lambda_reg * np.eye(k, dtype=np.float32)
    S_matrix = S_matrix_init.copy()

    # 交替最小二乘迭代
    for iteration in range(max_iter):
        iter_start = time.time()
        tqdm.write(f"\n迭代 {iteration + 1}/{max_iter} 开始")

        # 1. 更新用户因子与偏置
        tqdm.write(f"  [1/5] 更新用户因子 (活跃用户: {len(active_users):,} 个)")
        for u in tqdm(active_users, desc="  User Update  ", leave=False, position=0):
            res = update_user_factor(u, R, train_year_indices, train_year_bins, orig_order_idx,
                                     P, Q, user_bias, movie_bias, movie_time_bias,
                                     precomputed_tag_bias, GAMMA, reg_matrix, k)
            if res:
                _, new_p, new_bu = res
                P[u] = new_p
                user_bias[u] = new_bu
        tqdm.write("  → 用户因子更新完成\n")

        # 2. 更新电影因子与偏置
        tqdm.write(f"  [2/5] 更新电影因子 (活跃电影: {len(active_movies):,} 个)")
        for i in tqdm(active_movies, desc="  Movie Update ", leave=False, position=0):
            idx_start, idx_end = R_T.indptr[i], R_T.indptr[i + 1]
            if idx_start == idx_end:
                continue

            users = R_T.indices[idx_start:idx_end]
            r_ui = R_T.data[idx_start:idx_end]
            data_indices = orig_order_idx[idx_start:idx_end]
            curr_year_idx = train_year_indices[data_indices]
            rating_bins = train_year_bins[data_indices]

            tag_bias = precomputed_tag_bias[curr_year_idx, i]
            y = r_ui - global_mean - user_bias[users] - movie_time_bias[i, rating_bins] - tag_bias

            P_u = P[users]
            XtX = P_u.T @ P_u + reg_matrix
            Xty = P_u.T @ y
            try:
                Q[i] = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                Q[i] = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

            pred_mf = np.sum(P_u * Q[i], axis=1)
            pred_full = global_mean + user_bias[users] + movie_time_bias[i, rating_bins] + tag_bias + pred_mf
            movie_bias[i] = np.mean(r_ui - pred_full)
        tqdm.write("  → 电影因子更新完成\n")

        # 3. 更新时间动态偏置 b_{i,t}
        tqdm.write(f"  [3/5] 更新时间动态偏置 ({N_BINS} 个时间桶)")
        for bin_idx in tqdm(range(N_BINS), desc="  Time Bias   ", leave=False, position=0):
            mask = (train_year_bins == bin_idx)
            if mask.any():
                group_mean = np.bincount(train_cols[mask], weights=train_ratings[mask], minlength=num_movies)
                group_count = np.bincount(train_cols[mask], minlength=num_movies)
                valid = group_count > 0
                movie_time_bias[valid, bin_idx] = (group_mean[valid] / group_count[valid]) - global_mean
        tqdm.write("  → 时间偏置更新完成\n")

        # 4. 更新 S_matrix（带进度条）
        tqdm.write(f"  [4/5] 更新标签敏感度矩阵 S ({num_tags} 维，活跃电影: {len(active_movies):,} 个)")
        with tqdm(total=len(active_movies), desc="  S_matrix Upd", leave=False, position=0) as pbar:
            S_matrix = update_S_matrix(
                R_T, orig_order_idx, train_years, train_year_bins,
                P, Q, user_bias, movie_bias, movie_time_bias,
                tag_heat_matrix, S_matrix, GAMMA, LAMBDA_S, active_movies,
                progress_callback=pbar.update
            )
            precomputed_tag_bias = GAMMA * np.dot(H_full, S_matrix.T)
        tqdm.write("  → S_matrix 更新完成\n")

        # 5. 训练集近似 RMSE（监控收敛）
        tqdm.write(f"  [5/5] 计算训练集近似 RMSE (采样 {min(20000, R.nnz):,} 条)")
        sample_size = min(20000, R.nnz)
        sample_idx = np.random.choice(R.nnz, size=sample_size, replace=False)
        coo = R.tocoo()
        u_s, i_s, r_s = coo.row[sample_idx], coo.col[sample_idx], coo.data[sample_idx]
        pred_s = global_mean + user_bias[u_s] + movie_bias[i_s] + np.sum(P[u_s] * Q[i_s], axis=1)
        rmse = sqrt(np.mean((r_s - pred_s) ** 2))
        tqdm.write(f"  → 简略训练 RMSE: {rmse:.4f} | 本轮耗时: {time.time() - iter_start:.1f}s")

    return P, Q, user_bias, movie_bias, movie_time_bias, S_matrix


# ==============================================================================
# 验证集完整 RMSE 计算（包含所有偏置项）
# ==============================================================================
def calculate_rmse(val_rows, val_cols, val_ratings, val_year_bins, val_years,
                   P, Q, user_bias, movie_bias, movie_time_bias,
                   tag_heat_matrix, S_matrix, gamma, global_mean):
    """计算验证集上的完整 RMSE（含标签热度偏置）"""
    if len(val_ratings) == 0:
        return 0.0

    mf_term = np.sum(P[val_rows] * Q[val_cols], axis=1)
    pred = (global_mean +
            user_bias[val_rows] +
            movie_bias[val_cols] +
            mf_term +
            movie_time_bias[val_cols, val_year_bins])

    # 添加标签热度偏置（仅对验证集中出现的年份计算）
    if not tag_heat_matrix.empty and S_matrix.size > 0:
        present_years = tag_heat_matrix.index.intersection(val_years)
        if len(present_years) > 0:
            H_active = tag_heat_matrix.loc[present_years].values
            tag_bias_mat = gamma * (H_active @ S_matrix.T)
            year_to_idx = {y: i for i, y in enumerate(present_years)}
            val_year_idx = np.array([year_to_idx.get(y, -1) for y in val_years])
            valid = val_year_idx != -1
            if valid.any():
                pred[valid] += tag_bias_mat[val_year_idx[valid], val_cols[valid]]

    pred = np.clip(pred, 0.5, 5.0)
    return sqrt(np.mean((val_ratings - pred) ** 2))


# ==============================================================================
# 5 折交叉验证主函数
# ==============================================================================
def cross_validate_als(lambdas, n_splits=5, k=K_LATENT, max_iter=MAX_ITER, verbose=True):
    """执行 KFold 交叉验证，评估不同正则化参数 λ 的性能"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}

    for lambda_reg in lambdas:
        fold_rmses = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(n_samples)), 1):
            fold_start = time.time()

            # 划分训练/验证集
            tr_rows, tr_cols = rows[train_idx], cols[train_idx]
            tr_ratings = ratings_raw[train_idx]
            tr_year_bins, tr_years = year_bins[train_idx], years[train_idx]

            val_rows, val_cols = rows[val_idx], cols[val_idx]
            val_ratings = ratings_raw[val_idx]
            val_year_bins, val_years = year_bins[val_idx], years[val_idx]

            if verbose:
                print(f"\n=== Fold {fold}/{n_splits} | λ = {lambda_reg} ===")
                print(f"训练样本: {len(train_idx):,} | 验证样本: {len(val_idx):,}")

            # 训练模型
            train_start = time.time()
            P, Q, ub, mb, mtb, S_trained = als_train_fold(
                tr_rows, tr_cols, tr_ratings, tr_year_bins, tr_years,
                S_matrix_init=S_matrix.copy(),
                k=k, lambda_reg=lambda_reg, max_iter=max_iter, verbose=verbose
            )

            # 评估
            rmse = calculate_rmse(val_rows, val_cols, val_ratings, val_year_bins, val_years,
                                  P, Q, ub, mb, mtb, tag_heat_matrix, S_trained, GAMMA, global_mean)
            fold_rmses.append(rmse)

            if verbose:
                print(f"Fold {fold} RMSE: {rmse:.4f} | 训练时间: {time.time() - train_start:.1f}s | "
                      f"总时间: {time.time() - fold_start:.1f}s")

            # 清理内存
            del P, Q, ub, mb, mtb, S_trained
            gc.collect()

        mean_rmse = np.mean(fold_rmses)
        std_rmse = np.std(fold_rmses)
        results[lambda_reg] = {'mean': mean_rmse, 'std': std_rmse, 'folds': fold_rmses}

        if verbose:
            print(f"\nλ = {lambda_reg:.1f} → 平均 RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")

    return results


# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == "__main__":
    lambdas_to_test = [10.0, 1.0]  # 可扩展更多值进行网格搜索
    print("开始 5 折交叉验证（集成时间动态偏置 + 可学习标签热度偏置）\n")

    results = cross_validate_als(
        lambdas=lambdas_to_test,
        n_splits=5,
        k=K_LATENT,
        max_iter=MAX_ITER,
        verbose=True
    )

    # 结果汇总展示
    print("\n" + "=" * 70)
    print("交叉验证最终结果汇总")
    print("=" * 70)
    print(f"{'λ':>8} | {'平均RMSE':>10} | {'标准差':>8} | {'推荐':>8}")
    print("-" * 70)
    best_lambda = None
    best_rmse = float('inf')
    for lam in sorted(results.keys()):
        r = results[lam]
        if r['mean'] < best_rmse:
            best_rmse = r['mean']
            best_lambda = lam
        rec = "★★★★★" if r['std'] < 0.005 else "★★★★" if r['std'] < 0.01 else "★★★"
        print(f"{lam:8.1f} | {r['mean']:10.4f} | {r['std']:8.4f} | {rec:>8}")
    print("-" * 70)
    print(f"\n最佳 λ = {best_lambda:.1f}，平均 RMSE = {best_rmse:.4f}")