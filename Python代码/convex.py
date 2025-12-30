import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, svds
import pandas as pd
import gc
from sklearn.model_selection import KFold
from math import sqrt
import time
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==============================================================================
# 全局常量配置 (针对性能调整)
# ==============================================================================
MAX_ITER = 100  # 最大迭代次数
MAX_RANK = 50   # 每次 svds 的目标秩
TOL = 1e-4      # 收敛容差
EARLY_STOP_PATIENCE = 3  # 提前停止容忍度
RATING_FILE = 'preprocessed_ratings.csv'

print("加载数据...")
try:
    ratings = pd.read_csv(
        RATING_FILE,
        usecols=['user_idx', 'movie_idx', 'centered_rating', 'time_weight', 'rating'],
        dtype={
            'user_idx': 'int32',
            'movie_idx': 'int32',
            'centered_rating': 'float32',
            'time_weight': 'float32',
            'rating': 'float32'
        }
    )
except FileNotFoundError:
    raise FileNotFoundError(f"文件 {RATING_FILE} 不存在，请检查路径。")

# 转换为 numpy 数组并立即释放 DataFrame
rows = ratings['user_idx'].values
cols = ratings['movie_idx'].values
X_centered = ratings['centered_rating'].values
W = ratings['time_weight'].values
X_original = ratings['rating'].values

del ratings
gc.collect()

num_users = int(rows.max()) + 1
num_movies = int(cols.max()) + 1
n_samples = len(rows)

print(f"用户数: {num_users:,} | 电影数: {num_movies:,} | 评分总数: {n_samples:,}")

# ==============================================================================
# 核心辅助函数 (优化版)
# ==============================================================================

def compute_projection(U, s, Vt, rows, cols):
    """
    计算 P_Omega(U * diag(s) * Vt)，即对观测位置的投影值
    返回长度与观测条目数相同的 1D 数组
    """
    if U is None:
        return np.zeros(len(rows), dtype=np.float32)

    U_take = U[rows]
    Vt_take = Vt[:, cols]

    s_v = s[:, None] * Vt_take
    proj = np.einsum('ik,ki->i', U_take, s_v, dtype=np.float32)

    return proj

def get_linear_operator(delta, U, s, Vt, shape):
    """
    构建线性算子：delta + U * diag(s) * Vt
    使用 LinearOperator 避免显式构造巨大矩阵，支持高效的矩阵-向量乘法
    """
    def matvec(v):
        v = v.astype(np.float32)
        delta_v = delta @ v
        if U is None:
            return delta_v
        return delta_v + (U @ (s * (Vt @ v)))

    def matmat(V):
        V = V.astype(np.float32)
        delta_V = delta @ V
        if U is None:
            return delta_V
        tmp = Vt @ V
        tmp = s[:, None] * tmp
        return delta_V + (U @ tmp)

    def rmatvec(v):
        v = v.astype(np.float32)
        delta_v = delta.T @ v
        if U is None:
            return delta_v
        return delta_v + (Vt.T @ (s * (U.T @ v)))

    def rmatmat(V):
        V = V.astype(np.float32)
        delta_V = delta.T @ V
        if U is None:
            return delta_V
        tmp = U.T @ V
        tmp = s[:, None] * tmp
        return delta_V + (Vt.T @ tmp)

    return LinearOperator(
        shape=shape,
        matvec=matvec,
        matmat=matmat,
        rmatvec=rmatvec,
        rmatmat=rmatmat,
        dtype=np.float32
    )


# ==============================================================================
# 核心辅助函数
# ==============================================================================

def soft_impute_fold(
        train_rows, train_cols, train_X, train_W,
        lambda_, shape, max_iter=MAX_ITER, tol=TOL, max_rank=MAX_RANK,
        early_stop_patience=EARLY_STOP_PATIENCE, verbose=True
):
    """
    执行加权 Soft-Impute 算法（核范数正则化低秩矩阵补全）- 交叉验证优化版
    """
    n_obs = len(train_X)
    U, s, Vt = None, None, None

    # 预分配缓冲区 (保持不变)
    delta_buf = np.empty(n_obs, dtype=np.float32)

    best_loss = float('inf')
    patience_counter = 0
    loss_history = []

    if verbose >= 1:
        print(f" Soft-Impute 训练 | λ = {lambda_:.4f} | 最大秩 = {max_rank} | 容差 = {tol}")
        # 新增表头，用于规范化输出
        print("-" * 80)
        print(
            f"{'迭代':>4} | {'秩':>4} | {'拟合损失 (Fit Loss)':>20} | {'目标函数 (Obj Func)':>20} | {'训练RMSE':>10} | {'耗时 (s)':>8}")
        print("-" * 80)

    # 重点优化：移除 bar_format，让 tqdm 使用默认格式
    # 启用 leave=True 或 leave=False 取决于您是否想在迭代完成后保留进度条
    for it in tqdm(range(1, max_iter + 1), leave=False):
        iter_start = time.time()

        # ... (步骤 1-6：计算投影、残差、SVD、软阈值处理，代码保持不变) ...
        # 1. 计算当前 Z 在观测位置的投影 P_Z
        P_Z = compute_projection(U, s, Vt, train_rows, train_cols)
        # 2. 计算加权残差 delta_buf 和拟合损失
        np.subtract(train_X, P_Z, out=delta_buf)
        residual_sq = delta_buf ** 2
        loss = 0.5 * np.sum(train_W * residual_sq)
        np.multiply(train_W, delta_buf, out=delta_buf)
        loss_history.append(loss)
        # 3. 构建稀疏残差矩阵 Delta
        delta = csc_matrix(
            (delta_buf, (train_rows, train_cols)),
            shape=shape, dtype=np.float32
        )
        # 4. 构造线性算子 Y = Delta + Z
        op = get_linear_operator(delta, U, s, Vt, shape)
        # 5. 计算部分 SVD
        k_target = min(max_rank, min(shape) - 1)
        try:
            new_U, new_s, new_Vt = svds(op, k=k_target, which='LM', maxiter=1000, tol=0, return_singular_vectors=True)
            sorted_idx = np.argsort(-new_s)
            new_U = new_U[:, sorted_idx].astype(np.float32)
            new_s = new_s[sorted_idx].astype(np.float32)
            new_Vt = new_Vt[sorted_idx, :].astype(np.float32)
        except Exception as e:
            if verbose >= 1: print(f" SVD 失败: {e}")
            return None, None, None, loss_history
        # 6. 软阈值处理
        new_s_thresh = np.maximum(new_s - lambda_, 0).astype(np.float32)
        mask = new_s_thresh > 1e-8
        new_U = new_U[:, mask]
        new_Vt = new_Vt[mask, :]
        new_s_thresh = new_s_thresh[mask]

        # 7. 收敛检查和提前停止
        converged = False
        current_rank = len(new_s_thresh)

        # --- 计算完整目标函数和近似训练 RMSE ---
        reg_term = lambda_ * np.sum(new_s_thresh)
        obj_func = loss + reg_term
        approx_train_rmse = sqrt(2 * loss / n_obs)
        iter_time = time.time() - iter_start
        # ----------------------------------------

        # 重点优化：在迭代日志输出前，先使用 \r 覆盖 tqdm 进度条，并确保日志打印后换行
        if verbose >= 1:
            # \r 用于回到行首，清除当前的 tqdm 输出
            # 然后打印规范的日志行，并用 end='\n' 确保换行
            print(
                f"\r{it:4d} | {current_rank:4d} | {loss:20.4e} | {obj_func:20.4e} | {approx_train_rmse:10.4f} | {iter_time:8.1f}",
                flush=True)

        if it > 1:
            prev_loss = loss_history[-2]
            loss_change = abs(loss - prev_loss) / (abs(prev_loss) + 1e-12)

            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            if loss_change < tol or patience_counter >= early_stop_patience:
                converged = True

        U, s, Vt = new_U, new_s_thresh, new_Vt
        gc.collect()

        if converged:
            break

    # --- 循环结束后，打印收敛总结 ---
    if verbose >= 1:
        print("-" * 80)
        if converged:
            reason = f"损失变化 < {tol}" if loss_change < tol else f"耐心次数达到 {early_stop_patience}"
            print(f" 第 {it} 轮收敛 ({reason})")
        else:
            print(f" 达到最大迭代次数 {max_iter}，未完全收敛")

    return U, s, Vt, loss_history

def calculate_rmse_centered(U, s, Vt, val_rows, val_cols, val_X_cent):
    """
    计算验证集上的 RMSE（仅中心化评分）
    """
    if U is None:
        pred = np.zeros(len(val_rows), dtype=np.float32)
    else:
        pred = compute_projection(U, s, Vt, val_rows, val_cols)

    errors = val_X_cent - pred
    mse = np.mean(errors ** 2)
    return sqrt(mse)

# ==============================================================================
# 交叉验证主函数 (优化版)
# ==============================================================================

def cross_validate_softimpute(
        lambdas,
        rows, cols, X_centered, W, X_original,
        n_splits=5, max_rank=MAX_RANK, max_iter=MAX_ITER, tol=TOL,
        early_stop_patience=EARLY_STOP_PATIENCE, verbose=True
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    shape = (num_users, num_movies)

    results = {}

    for lambda_idx, lambda_ in enumerate(lambdas):
        print(f"\n{'=' * 50}")
        print(f"λ = {lambda_:.1f} ({lambda_idx + 1}/{len(lambdas)})")
        print('=' * 50)

        fold_rmses = []
        fold_details = []

        for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(np.arange(n_samples)), 1), desc="Fold 进度", total=n_splits, leave=False):
            fold_start = time.time()

            tr_rows = rows[train_idx]
            tr_cols = cols[train_idx]
            tr_X = X_centered[train_idx]
            tr_W = W[train_idx]

            val_rows = rows[val_idx]
            val_cols = cols[val_idx]
            val_X_cent = X_centered[val_idx]

            if verbose:
                print(f"\n Fold {fold}/{n_splits}:")
                print(f"  训练集: {len(train_idx):,} 样本")
                print(f"  验证集: {len(val_idx):,} 样本")

            train_start = time.time()
            U, s, Vt, loss_history = soft_impute_fold(
                tr_rows, tr_cols, tr_X, tr_W,
                lambda_=lambda_,
                shape=shape,
                max_iter=max_iter,
                tol=tol,
                max_rank=max_rank,
                early_stop_patience=early_stop_patience,
                verbose=verbose
            )
            train_time = time.time() - train_start

            if U is not None:
                rmse = calculate_rmse_centered(
                    U, s, Vt, val_rows, val_cols, val_X_cent
                )
                rank = len(s)
                final_loss = loss_history[-1] if loss_history else float('inf')
            else:
                rmse = 1.0
                rank = 0
                final_loss = float('inf')

            fold_rmses.append(rmse)

            fold_info = {
                'lambda': lambda_,
                'fold': fold,
                'rmse': rmse,
                'rank': rank,
                'train_time': train_time,
                'iterations': len(loss_history),
                'final_loss': final_loss
            }
            fold_details.append(fold_info)

            if verbose:
                print()
                print(f" Fold {fold} 结果:")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  秩: {rank}")
                print(f"  迭代次数: {len(loss_history)}")
                print(f"  训练时间: {train_time:.1f}s")
                print(f"  总时间: {time.time() - fold_start:.1f}s")

            del tr_rows, tr_cols, tr_X, tr_W
            del val_rows, val_cols, val_X_cent
            if U is not None:
                del U, s, Vt
            gc.collect()

        mean_rmse = np.mean(fold_rmses)
        std_rmse = np.std(fold_rmses)
        median_rmse = np.median(fold_rmses)

        results[lambda_] = {
            'mean': mean_rmse,
            'std': std_rmse,
            'median': median_rmse,
            'folds': fold_details
        }

        if verbose:
            print(f"\n λ={lambda_:.1f} 汇总:")
            print(f"  平均 RMSE: {mean_rmse:.4f}")
            print(f"  RMSE 标准差: {std_rmse:.4f}")
            print(f"  RMSE 中位数: {median_rmse:.4f}")

    return results

# ==============================================================================
# 执行交叉验证
# ==============================================================================

if __name__ == "__main__":
    lambdas = [100, 50, 10]

    print("开始 5 折交叉验证...")
    print(f"搜索λ值: {lambdas}")
    print(f"用户数: {num_users:,}, 电影数: {num_movies:,}")

    results = cross_validate_softimpute(
        lambdas=lambdas,
        rows=rows, cols=cols,
        X_centered=X_centered, W=W, X_original=X_original,
        n_splits=5,
        max_rank=MAX_RANK,
        max_iter=MAX_ITER,
        tol=TOL,
        early_stop_patience=EARLY_STOP_PATIENCE,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("交叉验证结果汇总")
    print("=" * 60)

    best_lambda = None
    best_rmse = float('inf')
    best_std = float('inf')

    print(f"{'λ':>8} | {'平均RMSE':>10} | {'标准差':>8} | {'中位数':>8} | {'推荐度':>8}")
    print("-" * 60)

    for lam, res in sorted(results.items(), key=lambda x: x[0], reverse=True):
        mean_rmse = res['mean']
        std_rmse = res['std']
        median_rmse = res['median']

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_std = std_rmse
            best_lambda = lam

        recommendation = "★★★★★" if std_rmse < 0.01 else "★★★★" if std_rmse < 0.02 else "★★★" if std_rmse < 0.03 else "★★" if std_rmse < 0.05 else "★"

        print(f"{lam:8.1f} | {mean_rmse:10.4f} | {std_rmse:8.4f} | {median_rmse:8.4f} | {recommendation:>8}")

    print("-" * 60)
    print(f"\n最佳 λ = {best_lambda:.1f}")
    print(f"平均 RMSE = {best_rmse:.4f} ± {best_std:.4f}")