import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN, OPTICS
from collections import defaultdict
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ========== 1. 数据预处理（Entity Embedding 支持） ==========
class ClearingDataPreprocessor:
    def __init__(self):
        self.vocabs = {}
        self.cat_cols = ['Account', 'Profit Center', 'Trading partner', 'Supplier', 'Document currency', 'Text']
        self._amt_mean = 0.0
        self._amt_std = 1.0
        self.cardinalities = {}

    def preprocess(self, df, fit=True):
        df = df.copy()
        df['Text'] = df['Text'].fillna('EMPTY')
        df['Trading partner'] = df['Trading partner'].fillna('0')
        df['Reference'] = df['Reference'].fillna('EMPTY')
        df['Assignment'] = df['Assignment'].fillna('EMPTY')

        if fit:
            for col in self.cat_cols:
                values = df[col].astype(str).values
                unique = sorted(set(v for v in values if pd.notna(v)))
                self.vocabs[col] = {v: i + 1 for i, v in enumerate(unique)}
                self.cardinalities[col] = len(unique) + 1
            amt = df['Amount in doc. curr.'].values
            self._amt_mean = float(amt.mean())
            self._amt_std = float(amt.std())

        cat_list = []
        for col in self.cat_cols:
            vocab = self.vocabs.get(col, {})
            indices = df[col].astype(str).map(lambda x: vocab.get(x, 0)).values.astype(np.int64)
            cat_list.append(indices)
        cat_indices = np.column_stack(cat_list)

        amounts = df['Amount in doc. curr.'].values.astype(np.float64)
        amt_scaled = (amounts - self._amt_mean) / (self._amt_std + 1e-8)

        return cat_indices, amounts, amt_scaled.astype(np.float32)


# ========== 2. 模型（Entity Embedding + 投影） ==========
class ClearingEncoder(nn.Module):
    def __init__(self, cardinalities, embedding_dim=32):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        total_dim = 0
        for col, n_cat in cardinalities.items():
            dim = min(16, max(2, n_cat // 4)) if col == 'Text' else min(24, max(4, n_cat // 4))
            self.embeddings[col] = nn.Embedding(n_cat, dim)
            total_dim += dim
        total_dim += 1

        self.projection = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, cat_values, amt_scaled):
        embs = [self.embeddings[col](cat_values[:, i]) for i, col in enumerate(self.embeddings)]
        embs.append(amt_scaled.unsqueeze(1))
        combined = torch.cat(embs, dim=1)
        emb = self.projection(combined)
        return F.normalize(emb, p=2, dim=1)


def nt_xent_loss(embeddings, labels, temperature=0.5):
    device = embeddings.device
    n = embeddings.shape[0]
    sim = embeddings @ embeddings.T / temperature
    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(n, dtype=torch.bool, device=device)
    pos_mask = label_eq & ~eye
    sim_exp = torch.exp(sim)
    pos_sum = (sim_exp * pos_mask.float()).sum(dim=1)
    all_sum = sim_exp.sum(dim=1) - sim_exp.diag()
    loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))
    valid = pos_mask.sum(dim=1) > 0
    return loss[valid].mean() if valid.any() else torch.tensor(0.0, device=device)


# ========== 3. 模型持久化 ==========
def save_model(model, preprocessor, filepath='model.pt'):
    torch.save({
        'arch_version': 3,
        'model_state_dict': model.state_dict(),
        'vocabs': preprocessor.vocabs,
        'cardinalities': preprocessor.cardinalities,
        'amt_mean': preprocessor._amt_mean,
        'amt_std': preprocessor._amt_std,
    }, filepath)
    print(f"模型已保存到 {filepath}")


def load_model(filepath='model.pt'):
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    preprocessor = ClearingDataPreprocessor()
    preprocessor.vocabs = checkpoint['vocabs']
    preprocessor.cardinalities = checkpoint['cardinalities']
    preprocessor._amt_mean = checkpoint.get('amt_mean', 0.0)
    preprocessor._amt_std = checkpoint.get('amt_std', 1.0)
    model = ClearingEncoder(preprocessor.cardinalities).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型已从 {filepath} 加载")
    return model, preprocessor


# ========== 4. 训练（对比学习） ==========
def train_encoder_model(df_data, epochs=50, batch_group_size=10, lr=0.001, patience=15, val_ratio=0.15):
    print("=" * 70)
    print("训练编码器（对比学习 + Entity Embedding）")
    print("=" * 70)

    preprocessor = ClearingDataPreprocessor()
    cat_indices, amounts, amt_scaled = preprocessor.preprocess(df_data, fit=True)
    group_ids = df_data['Group ID'].values
    accounts_full = df_data['Account'].astype(str).values

    print(f"\n类别特征数: {len(preprocessor.cat_cols)}")
    for col in preprocessor.cat_cols:
        print(f"  {col}: {preprocessor.cardinalities[col]} 个唯一值")

    unique_groups_all = np.unique(group_ids)
    train_groups, val_groups = train_test_split(unique_groups_all, test_size=val_ratio, random_state=42)
    train_mask = np.isin(group_ids, train_groups)

    cat_train = cat_indices[train_mask]
    amt_train = amt_scaled[train_mask]
    gid_train = group_ids[train_mask]
    acct_train = accounts_full[train_mask]

    groups_train = defaultdict(list)
    for idx, gid in enumerate(gid_train):
        groups_train[gid].append(idx)
    group_list_train = list(groups_train.keys())

    account_to_groups = defaultdict(set)
    for idx, gid in enumerate(gid_train):
        account_to_groups[acct_train[idx]].add(gid)

    model = ClearingEncoder(preprocessor.cardinalities).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"训练组数: {len(group_list_train)}, 验证组数: {len(val_groups)}")

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for _ in range(200):
            n_groups = min(batch_group_size, len(group_list_train))
            selected_gids = np.random.choice(group_list_train, n_groups, replace=False)

            batch_indices = []
            batch_gids = []

            for gid in selected_gids:
                members = groups_train[gid]
                k = min(np.random.randint(2, 5), len(members))
                idxs = np.random.choice(members, k, replace=False)
                batch_indices.extend(idxs)
                batch_gids.extend([gid] * k)

            extra = []
            for _ in range(n_groups // 2):
                gid = np.random.choice(group_list_train)
                acct = acct_train[np.random.choice(groups_train[gid])]
                same_acct = [g for g in account_to_groups.get(acct, set()) if g != gid]
                if same_acct:
                    gid2 = np.random.choice(same_acct)
                    extra.append(np.random.choice(groups_train[gid]))
                    extra.append(np.random.choice(groups_train[gid2]))
            if extra:
                batch_indices.extend(extra)
                batch_gids.extend([-1] * len(extra))

            if len(batch_indices) < 4:
                continue

            x_cat = torch.LongTensor(cat_train[batch_indices]).to(device)
            x_amt = torch.FloatTensor(amt_train[batch_indices]).to(device)

            unique_batch = list(set(g for g in batch_gids if g != -1))
            g2i = {g: i for i, g in enumerate(unique_batch)}
            labels = torch.LongTensor([g2i.get(g, -1) for g in batch_gids]).to(device)

            embeddings = model(x_cat, x_amt)
            loss = nt_xent_loss(embeddings, labels, temperature=0.3)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        val_loss = None
        if len(val_groups) >= 2:
            model.eval()
            with torch.no_grad():
                v_idx, v_gid = [], []
                for gid in val_groups:
                    idxs = list(np.where(group_ids == gid)[0])
                    if len(idxs) >= 2:
                        v_idx.extend(idxs)
                        v_gid.extend([gid] * len(idxs))
                if len(v_idx) >= 4:
                    uvg = list(set(v_gid))
                    vg2i = {g: i for i, g in enumerate(uvg)}
                    v_lab = torch.LongTensor([vg2i[g] for g in v_gid]).to(device)
                    v_cat = torch.LongTensor(cat_indices[v_idx]).to(device)
                    v_amt = torch.FloatTensor(amt_scaled[v_idx]).to(device)
                    v_emb = model(v_cat, v_amt)
                    val_loss = nt_xent_loss(v_emb, v_lab, temperature=0.3).item()

        if epoch % 5 == 0:
            msg = f"Epoch {epoch:3d}: train_loss={epoch_loss / max(batch_count, 1):.4f}"
            if val_loss is not None:
                msg += f", val_loss={val_loss:.4f}"
            print(msg)

        monitor = val_loss if val_loss is not None else epoch_loss / max(batch_count, 1)
        if monitor < best_val_loss:
            best_val_loss = monitor
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    print("\n训练完成！")
    return model, preprocessor


# ========== 5. 推理（DBSCAN 聚类） ==========
def find_clearing_groups(model, preprocessor, df_new, max_group_size=10, dbscan_eps=0.5):
    print("\n" + "=" * 70)
    print("寻找清账组（DBSCAN 聚类）")
    print("=" * 70)

    cat_indices, amounts, amt_scaled = preprocessor.preprocess(df_new, fit=False)
    n = len(cat_indices)
    account_values = df_new['Account'].astype(str).values

    account_to_indices = defaultdict(list)
    for idx in range(n):
        account_to_indices[account_values[idx]].append(idx)
    print(f"\n处理 {n} 条凭证, {len(account_to_indices)} 个 Account 分组")

    print("预计算 embeddings...")
    model.eval()
    with torch.no_grad():
        x_cat = torch.LongTensor(cat_indices).to(device)
        x_amt = torch.FloatTensor(amt_scaled).to(device)
        embeddings = model(x_cat, x_amt).cpu().numpy()

    all_clearing_groups = []

    for account, indices_list in account_to_indices.items():
        indices = np.array(indices_list)
        if len(indices) < 2:
            continue
        local_embeds = embeddings[indices]

        clustering = DBSCAN(eps=dbscan_eps, min_samples=2, metric='cosine').fit(local_embeds)

        for cluster_id in set(clustering.labels_):
            if cluster_id < 0:
                continue
            cluster_global = indices[clustering.labels_ == cluster_id]
            if len(cluster_global) < 2:
                continue
            subsets = find_zero_sum_subsets(cluster_global, amounts, max_group_size)
            all_clearing_groups.extend(subsets)

    print(f"\n找到 {len(all_clearing_groups)} 个清账组")
    return all_clearing_groups


# ========== 6. 零和子集搜索 ==========
def find_zero_sum_subsets(indices, amounts, max_group_size=10, tolerance=0.01):
    n = len(indices)
    if n > 25:
        return greedy_zero_sum_search(indices, amounts, tolerance)
    zero_sum_groups = []
    for size in range(2, min(n + 1, max_group_size + 1)):
        for combo in combinations(range(n), size):
            subset = [indices[i] for i in combo]
            if abs(amounts[subset].sum()) < tolerance:
                zero_sum_groups.append(subset)
    return select_non_overlapping_groups(zero_sum_groups)


def greedy_zero_sum_search(indices, amounts, tolerance=0.01):
    clearing_groups = []
    used = set()
    sorted_pairs = sorted([(i, amounts[i]) for i in indices], key=lambda x: abs(x[1]), reverse=True)
    for anchor_idx, anchor_amt in sorted_pairs:
        if anchor_idx in used:
            continue
        group = [anchor_idx]
        cur_sum = anchor_amt
        used.add(anchor_idx)
        remaining = [(i, amounts[i]) for i in indices if i not in used]
        for c_idx, c_amt in remaining:
            new_sum = cur_sum + c_amt
            if abs(new_sum) < abs(cur_sum):
                group.append(c_idx)
                cur_sum = new_sum
                used.add(c_idx)
                if abs(cur_sum) < tolerance:
                    break
        if abs(cur_sum) < tolerance and len(group) > 1:
            clearing_groups.append(group)
    return clearing_groups


def select_non_overlapping_groups(groups):
    if not groups:
        return []
    groups = sorted(groups, key=len, reverse=True)
    selected = []
    used = set()
    for group in groups:
        if not any(idx in used for idx in group):
            selected.append(group)
            used.update(group)
    return selected


# ========== 7. 评估 ==========
def evaluate_clearing_results(df_test, predicted_groups):
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)

    n_total = len(df_test)
    covered_indices = set()
    for group in predicted_groups:
        covered_indices.update(group)
    coverage = len(covered_indices) / n_total

    zero_sum_count = 0
    total_deviation = 0.0
    for group in predicted_groups:
        group_sum = df_test.iloc[group]['Amount in doc. curr.'].sum()
        total_deviation += abs(group_sum)
        if abs(group_sum) < 0.01:
            zero_sum_count += 1

    zsa = zero_sum_count / len(predicted_groups) if predicted_groups else 0
    avg_dev = total_deviation / len(predicted_groups) if predicted_groups else 0
    group_sizes = [len(g) for g in predicted_groups]
    avg_gs = np.mean(group_sizes) if group_sizes else 0

    if 'Group ID' in df_test.columns:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        pred_labels = np.full(n_total, -1)
        true_labels = df_test['Group ID'].values
        for i, group in enumerate(predicted_groups):
            for idx in group:
                pred_labels[idx] = i
        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
    else:
        ari = nmi = None

    print(f"\n覆盖率: {coverage:.2%} ({len(covered_indices)}/{n_total})")
    print(f"零和准确率: {zsa:.2%} ({zero_sum_count}/{len(predicted_groups)})")
    print(f"平均偏差: {avg_dev:.4f}")
    print(f"预测组数: {len(predicted_groups)}")
    print(f"平均组大小: {avg_gs:.2f}")
    print(f"组大小分布: min={min(group_sizes) if group_sizes else 0}, "
          f"max={max(group_sizes) if group_sizes else 0}, "
          f"median={np.median(group_sizes) if group_sizes else 0:.1f}")
    if ari is not None:
        print(f"\nAdjusted Rand Index: {ari:.4f}")
        print(f"Normalized Mutual Info: {nmi:.4f}")

    return {
        'coverage': coverage, 'zero_sum_accuracy': zsa, 'avg_deviation': avg_dev,
        'num_groups': len(predicted_groups), 'avg_group_size': avg_gs, 'ari': ari, 'nmi': nmi
    }


# ========== 8. 主流程 ==========
def main(data_path='clearing_data.csv', max_test_groups=None, model_path='model.pt'):
    print("=" * 70)
    print("自动清账系统 - Entity Embedding + 对比学习")
    print("=" * 70)

    print("\n加载数据...")
    df = pd.read_csv(data_path)
    print(f"数据形状: {df.shape}")
    print(f"唯一组数: {df['Group ID'].nunique()}")
    print(f"平均每组样本数: {len(df) / df['Group ID'].nunique():.2f}")

    print("\n划分训练集和测试集...")
    unique_groups = df['Group ID'].unique()
    train_groups_full, test_groups_full = train_test_split(unique_groups, test_size=0.2, random_state=42)

    if max_test_groups is not None and max_test_groups < len(test_groups_full):
        test_groups = np.random.RandomState(42).choice(test_groups_full, max_test_groups, replace=False)
    else:
        test_groups = test_groups_full

    df_data = df[df['Group ID'].isin(train_groups_full)].reset_index(drop=True)
    df_test = df[df['Group ID'].isin(test_groups)].reset_index(drop=True)

    print(f"训练数据: {len(df_data)} 行, {len(train_groups_full)} 组")
    print(f"测试集: {len(df_test)} 行, {len(test_groups)} 组")

    import os
    need_retrain = False
    if os.path.exists(model_path):
        tmp = torch.load(model_path, map_location='cpu', weights_only=False)
        if tmp.get('arch_version') != 3:
            print(f"\n检测到旧版模型 {model_path}，将重新训练...")
            need_retrain = True
        else:
            print(f"\n发现已有模型 {model_path}，加载中...")
            model, preprocessor = load_model(model_path)
    else:
        need_retrain = True

    if need_retrain:
        print("\n开始训练...")
        model, preprocessor = train_encoder_model(df_data, epochs=50, batch_group_size=10, patience=15)
        save_model(model, preprocessor, model_path)

    df_test_no_label = df_test.drop('Group ID', axis=1)

    predicted_groups = find_clearing_groups(
        model, preprocessor, df_test_no_label,
        max_group_size=10, dbscan_eps=0.3
    )

    metrics = evaluate_clearing_results(df_test, predicted_groups)

    print("\n" + "=" * 70)
    print("示例清账组（前 5 个）")
    print("=" * 70)
    for i, group in enumerate(predicted_groups[:5]):
        print(f"\n{'='*70}")
        print(f"组 {i+1} (共 {len(group)} 条凭证)")
        print(f"{'='*70}")
        group_df = df_test.iloc[group][['Account', 'Company Code', 'Profit Center', 'Supplier', 'Text', 'Amount in doc. curr.']]
        print(group_df.to_string(index=False))
        print(f"\n总金额: {group_df['Amount in doc. curr.'].sum():.4f}")
        if 'Group ID' in df_test.columns:
            true_groups = df_test.iloc[group]['Group ID'].unique()
            print(f"真实 Group ID: {true_groups}")

    print("\n" + "=" * 70)
    print("保存结果...")
    df_test['Predicted_Group'] = -1
    for i, group in enumerate(predicted_groups):
        for idx in group:
            df_test.at[idx, 'Predicted_Group'] = i
    df_test.to_csv('clearing_results.csv', index=False)
    print("结果已保存到 clearing_results.csv")

    return model, preprocessor, predicted_groups, metrics


if __name__ == "__main__":
    model, preprocessor, predicted_groups, metrics = main(
        'clearing_data.csv',
        max_test_groups=50
    )
