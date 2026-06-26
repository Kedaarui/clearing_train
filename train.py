import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from collections import defaultdict
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# 设备检测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ========== 1. 数据预处理 ==========
class ClearingDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def preprocess(self, df, fit=True):
        """
        预处理清账数据
        """
        df = df.copy()
        
        # 填充缺失值
        df['Text'] = df['Text'].fillna('EMPTY')
        df['Trading partner'] = df['Trading partner'].fillna('0')
        df['Reference'] = df['Reference'].fillna('EMPTY')
        df['Assignment'] = df['Assignment'].fillna('EMPTY')
        
        # 选择关键特征
        categorical_features = [
            'Company Code',
            'Account', 
            'Profit Center',
            'Trading partner',
            'Supplier',
            'Document currency',
            'Text'
        ]
        
        # 编码分类特征
        encoded_features = []
        
        for col in categorical_features:
            if fit:
                le = LabelEncoder()
                encoded = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is None:
                    encoded = np.zeros(len(df))
                else:
                    encoded = df[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
            
            encoded_features.append(encoded)
        
        # 合并特征
        X = np.column_stack(encoded_features)
        
        # 金额
        amounts = df['Amount in doc. curr.'].values
        
        # 标准化特征（不包括金额）
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, amounts, categorical_features

# ========== 2. 深度学习模型 ==========
class ClearingCompatibilityNet(nn.Module):
    """
    学习两个凭证是否"兼容"（可能在同一清账组）
    
    输入：两个凭证的特征
    输出：兼容性得分 (0-1)
    """
    def __init__(self, feature_dim):
        super().__init__()
        
        # 单个凭证的编码器
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )
        
        # 成对兼容性判断器
        self.compatibility_net = nn.Sequential(
            nn.Linear(16 * 2 + 2, 32),  # 两个编码 + 金额特征
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x1, x2, amt1, amt2):
        """
        x1, x2: 两个凭证的特征 [batch, feature_dim]
        amt1, amt2: 两个凭证的金额 [batch, 1]
        """
        # 编码
        emb1 = self.encoder(x1)
        emb2 = self.encoder(x2)
        
        # 金额特征
        amt_sum = (amt1 + amt2).unsqueeze(1)
        amt_diff = torch.abs(amt1 - amt2).unsqueeze(1)
        
        # 拼接
        combined = torch.cat([emb1, emb2, amt_sum, amt_diff], dim=1)
        
        # 兼容性得分
        compatibility = self.compatibility_net(combined)
        
        return compatibility

# ========== 3. 模型持久化 ==========
def save_model(model, preprocessor, filepath='model.pt'):
    torch.save({
        'feature_dim': model.encoder[0].in_features,
        'model_state_dict': model.state_dict(),
        'label_encoders': preprocessor.label_encoders,
        'scaler': preprocessor.scaler,
    }, filepath)
    print(f"模型已保存到 {filepath}")

def load_model(model, preprocessor, filepath='model.pt'):
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    preprocessor.label_encoders = checkpoint['label_encoders']
    preprocessor.scaler = checkpoint['scaler']
    feat_dim = checkpoint['feature_dim']
    model = ClearingCompatibilityNet(feature_dim=feat_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型已从 {filepath} 加载 (feature_dim={feat_dim})")
    return model, preprocessor

# ========== 4. 训练函数 ==========
def train_compatibility_model(df_data, epochs=50, batch_size=128, lr=0.001, patience=10, val_ratio=0.15):
    """
    训练兼容性模型（含验证集 + early stopping）
    """
    print("=" * 70)
    print("训练兼容性模型")
    print("=" * 70)
    
    preprocessor = ClearingDataPreprocessor()
    X, amounts, _ = preprocessor.preprocess(df_data, fit=True)
    group_ids = df_data['Group ID'].values
    
    print(f"\n特征维度: {X.shape[1]}")
    print(f"样本数: {len(X)}")
    print(f"组数: {len(np.unique(group_ids))}")
    
    # 按组划分 train / val
    unique_groups_all = np.unique(group_ids)
    train_groups, val_groups = train_test_split(unique_groups_all, test_size=val_ratio, random_state=42)
    train_mask = np.isin(group_ids, train_groups)
    
    X_train = X[train_mask]
    amounts_train = amounts[train_mask]
    group_ids_train = group_ids[train_mask]
    
    groups_train = defaultdict(list)
    for idx, gid in enumerate(group_ids_train):
        groups_train[gid].append(idx)
    group_list_train = list(groups_train.keys())
    
    # 预采样验证 pair（固定，保证跨 epoch 可比）
    val_pairs = []
    val_groups_dict = defaultdict(list)
    for gid in val_groups:
        idxs = list(np.where(group_ids == gid)[0])
        if len(idxs) >= 2:
            val_groups_dict[gid] = idxs
    val_group_list = list(val_groups_dict.keys())
    
    if val_group_list:
        for _ in range(2000):
            if np.random.rand() > 0.5 and len(val_group_list) > 0:
                gid = np.random.choice(val_group_list)
                i1, i2 = np.random.choice(val_groups_dict[gid], 2, replace=False)
                val_pairs.append((i1, i2, 1.0))
            else:
                g1, g2 = np.random.choice(val_group_list, 2, replace=False)
                i1 = np.random.choice(val_groups_dict[g1])
                i2 = np.random.choice(val_groups_dict[g2])
                val_pairs.append((i1, i2, 0.0))
    else:
        val_pairs = None
    
    model = ClearingCompatibilityNet(feature_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"训练组数: {len(group_list_train)}, 验证组数: {len(val_group_list)}")
    
    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        batch_count = 0
        
        for _ in range(200):
            batch_x1, batch_x2, batch_amt1, batch_amt2, batch_labels = [], [], [], [], []
            
            for _ in range(batch_size):
                if np.random.rand() > 0.5:
                    gid = np.random.choice(group_list_train)
                    if len(groups_train[gid]) < 2:
                        continue
                    idx1, idx2 = np.random.choice(groups_train[gid], 2, replace=False)
                    label = 1.0
                else:
                    gid1, gid2 = np.random.choice(group_list_train, 2, replace=False)
                    idx1 = np.random.choice(groups_train[gid1])
                    idx2 = np.random.choice(groups_train[gid2])
                    label = 0.0
                
                batch_x1.append(X_train[idx1])
                batch_x2.append(X_train[idx2])
                batch_amt1.append(amounts_train[idx1])
                batch_amt2.append(amounts_train[idx2])
                batch_labels.append(label)
            
            if not batch_x1:
                continue
            
            x1 = torch.FloatTensor(np.array(batch_x1)).to(device)
            x2 = torch.FloatTensor(np.array(batch_x2)).to(device)
            amt1 = torch.FloatTensor(batch_amt1).to(device)
            amt2 = torch.FloatTensor(batch_amt2).to(device)
            labels = torch.FloatTensor(batch_labels).unsqueeze(1).to(device)
            
            preds = model(x1, x2, amt1, amt2)
            loss = criterion(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += ((preds > 0.5).float() == labels).float().mean().item()
            batch_count += 1
        
        # validation
        val_loss = None
        if val_pairs is not None:
            model.eval()
            with torch.no_grad():
                v_x1 = torch.FloatTensor(np.array([X[p[0]] for p in val_pairs])).to(device)
                v_x2 = torch.FloatTensor(np.array([X[p[1]] for p in val_pairs])).to(device)
                v_amt1 = torch.FloatTensor([amounts[p[0]] for p in val_pairs]).to(device)
                v_amt2 = torch.FloatTensor([amounts[p[1]] for p in val_pairs]).to(device)
                v_labels = torch.FloatTensor([p[2] for p in val_pairs]).unsqueeze(1).to(device)
                v_preds = model(v_x1, v_x2, v_amt1, v_amt2)
                val_loss = criterion(v_preds, v_labels).item()
        
        if epoch % 5 == 0:
            msg = f"Epoch {epoch:3d}: train_loss={epoch_loss/batch_count:.4f}, train_acc={epoch_acc/batch_count:.4f}"
            if val_loss is not None:
                msg += f", val_loss={val_loss:.4f}"
            print(msg)
        
        if val_loss is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
        else:
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
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

# ========== 4. 推理：寻找清账组 ==========
def find_clearing_groups(model, preprocessor, df_new, 
                        compatibility_threshold=0.6,
                        max_group_size=10,
                        block_size=256,
                        pair_batch=8192):
    """
    按 Account 硬过滤后寻找清账组
    """
    print("\n" + "=" * 70)
    print("寻找清账组")
    print("=" * 70)
    
    import networkx as nx
    
    X, amounts, _ = preprocessor.preprocess(df_new, fit=False)
    n = len(X)
    account_values = df_new['Account'].astype(str).values
    
    # 按 Account 分组
    account_to_indices = defaultdict(list)
    for idx in range(n):
        account_to_indices[account_values[idx]].append(idx)
    print(f"\n处理 {n} 条凭证, {len(account_to_indices)} 个 Account 分组")
    
    # 全局预计算 embeddings
    print("预计算 embeddings...")
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        all_embeds = model.encoder(X_tensor)
        all_amt = torch.FloatTensor(amounts).to(device)
    
    all_clearing_groups = []
    
    for acc_idx, (account, indices_list) in enumerate(account_to_indices.items()):
        indices = np.array(indices_list)
        local_n = len(indices)
        if local_n < 2:
            continue
        
        # 当前 Account 组内构建图
        G = _build_account_graph(
            model, all_embeds, all_amt, indices,
            compatibility_threshold, local_n, block_size, pair_batch
        )
        
        # 连通分量 -> 零和子集
        for component in nx.connected_components(G):
            if len(component) < 2:
                continue
            local_comp = list(component)
            global_comp = indices[local_comp]
            clearing_subsets = find_zero_sum_subsets(global_comp, amounts, max_group_size)
            all_clearing_groups.extend(clearing_subsets)
        
        if (acc_idx + 1) % 500 == 0:
            print(f"  处理了 {acc_idx + 1}/{len(account_to_indices)} 个账户")
    
    print(f"\n找到 {len(all_clearing_groups)} 个清账组")
    return all_clearing_groups


def _build_account_graph(model, all_embeds, all_amt, indices,
                         threshold, local_n, block_size, pair_batch):
    """在单个 Account 组内构建兼容图"""
    import networkx as nx
    
    G = nx.Graph()
    G.add_nodes_from(range(local_n))
    
    with torch.no_grad():
        for i in range(0, local_n, block_size):
            end_i = min(i + block_size, local_n)
            # 块内 pair（三角形）
            if end_i - i >= 2:
                inner_a, inner_b = np.meshgrid(
                    np.arange(i, end_i), np.arange(i, end_i), indexing='ij')
                inner_a = inner_a.flatten()
                inner_b = inner_b.flatten()
                keep = inner_a < inner_b
                inner_a = inner_a[keep]
                inner_b = inner_b[keep]
                
                for start in range(0, len(inner_a), pair_batch):
                    end = min(start + pair_batch, len(inner_a))
                    ga = indices[inner_a[start:end]]
                    gb = indices[inner_b[start:end]]
                    
                    combined = torch.cat([
                        all_embeds[torch.from_numpy(ga).to(all_embeds.device)],
                        all_embeds[torch.from_numpy(gb).to(all_embeds.device)],
                        (all_amt[torch.from_numpy(ga).to(all_amt.device)] + all_amt[torch.from_numpy(gb).to(all_amt.device)]).unsqueeze(1),
                        torch.abs(all_amt[torch.from_numpy(ga).to(all_amt.device)] - all_amt[torch.from_numpy(gb).to(all_amt.device)]).unsqueeze(1),
                    ], dim=1)
                    
                    compat = model.compatibility_net(combined).squeeze(-1).cpu().numpy()
                    for k in np.where(compat > threshold)[0]:
                        G.add_edge(inner_a[start+k], inner_b[start+k])
            
            # 跨块 pair
            for j in range(end_i, local_n, block_size):
                end_j = min(j + block_size, local_n)
                ga, gb = np.meshgrid(
                    np.arange(i, end_i), np.arange(j, end_j), indexing='ij')
                ga = ga.flatten()
                gb = gb.flatten()
                
                for start in range(0, len(ga), pair_batch):
                    end = min(start + pair_batch, len(ga))
                    gag = indices[ga[start:end]]
                    gbg = indices[gb[start:end]]
                    
                    combined = torch.cat([
                        all_embeds[torch.from_numpy(gag).to(all_embeds.device)],
                        all_embeds[torch.from_numpy(gbg).to(all_embeds.device)],
                        (all_amt[torch.from_numpy(gag).to(all_amt.device)] + all_amt[torch.from_numpy(gbg).to(all_amt.device)]).unsqueeze(1),
                        torch.abs(all_amt[torch.from_numpy(gag).to(all_amt.device)] - all_amt[torch.from_numpy(gbg).to(all_amt.device)]).unsqueeze(1),
                    ], dim=1)
                    
                    compat = model.compatibility_net(combined).squeeze(-1).cpu().numpy()
                    for k in np.where(compat > threshold)[0]:
                        G.add_edge(ga[start+k], gb[start+k])
    
    return G

def find_zero_sum_subsets(indices, amounts, max_group_size=10, tolerance=0.01):
    """
    在给定索引中寻找金额和为 0 的子集
    """
    n = len(indices)
    zero_sum_groups = []
    
    if n > 25:
        # 使用贪心算法
        return greedy_zero_sum_search(indices, amounts, tolerance)
    
    # 枚举所有可能的子集
    for size in range(2, min(n + 1, max_group_size + 1)):
        for combo in combinations(range(n), size):
            subset_indices = [indices[i] for i in combo]
            subset_amounts = amounts[subset_indices]
            
            if abs(subset_amounts.sum()) < tolerance:
                zero_sum_groups.append(subset_indices)
    
    # 选择不重叠的组
    return select_non_overlapping_groups(zero_sum_groups)

def greedy_zero_sum_search(indices, amounts, tolerance=0.01):
    """
    贪心搜索零和组
    """
    clearing_groups = []
    used = set()
    
    # 按金额绝对值排序
    sorted_pairs = sorted(
        [(i, amounts[i]) for i in indices], 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    
    for anchor_idx, anchor_amt in sorted_pairs:
        if anchor_idx in used:
            continue
        
        current_group = [anchor_idx]
        current_sum = anchor_amt
        used.add(anchor_idx)
        
        # 贪心添加其他凭证
        remaining = [(i, amounts[i]) for i in indices if i not in used]
        
        for candidate_idx, candidate_amt in remaining:
            new_sum = current_sum + candidate_amt
            
            # 如果更接近 0，添加
            if abs(new_sum) < abs(current_sum):
                current_group.append(candidate_idx)
                current_sum = new_sum
                used.add(candidate_idx)
                
                if abs(current_sum) < tolerance:
                    break
        
        # 保存组
        if abs(current_sum) < tolerance and len(current_group) > 1:
            clearing_groups.append(current_group)
    
    return clearing_groups

def select_non_overlapping_groups(groups):
    """
    选择不重叠的组（优先选择大组）
    """
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

# ========== 5. 评估 ==========
def evaluate_clearing_results(df_test, predicted_groups):
    """
    评估清账结果
    """
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    
    n_total = len(df_test)
    
    # 1. 覆盖率
    covered_indices = set()
    for group in predicted_groups:
        covered_indices.update(group)
    coverage = len(covered_indices) / n_total
    
    # 2. 零和准确率
    zero_sum_count = 0
    total_deviation = 0
    
    for group in predicted_groups:
        group_sum = df_test.iloc[group]['Amount in doc. curr.'].sum()
        total_deviation += abs(group_sum)
        
        if abs(group_sum) < 0.01:
            zero_sum_count += 1
    
    zero_sum_accuracy = zero_sum_count / len(predicted_groups) if predicted_groups else 0
    avg_deviation = total_deviation / len(predicted_groups) if predicted_groups else 0
    
    # 3. 组大小统计
    group_sizes = [len(g) for g in predicted_groups]
    avg_group_size = np.mean(group_sizes) if group_sizes else 0
    
    # 4. 与真实标签对比（如果有）
    if 'Group ID' in df_test.columns:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        # 转换为标签
        pred_labels = np.full(n_total, -1)
        true_labels = df_test['Group ID'].values
        
        for i, group in enumerate(predicted_groups):
            for idx in group:
                pred_labels[idx] = i
        
        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
    else:
        ari = None
        nmi = None
    
    # 打印结果
    print(f"\n覆盖率: {coverage:.2%} ({len(covered_indices)}/{n_total})")
    print(f"零和准确率: {zero_sum_accuracy:.2%} ({zero_sum_count}/{len(predicted_groups)})")
    print(f"平均偏差: {avg_deviation:.4f}")
    print(f"预测组数: {len(predicted_groups)}")
    print(f"平均组大小: {avg_group_size:.2f}")
    print(f"组大小分布: min={min(group_sizes) if group_sizes else 0}, "
          f"max={max(group_sizes) if group_sizes else 0}, "
          f"median={np.median(group_sizes) if group_sizes else 0:.1f}")
    
    if ari is not None:
        print(f"\nAdjusted Rand Index: {ari:.4f}")
        print(f"Normalized Mutual Info: {nmi:.4f}")
    
    return {
        'coverage': coverage,
        'zero_sum_accuracy': zero_sum_accuracy,
        'avg_deviation': avg_deviation,
        'num_groups': len(predicted_groups),
        'avg_group_size': avg_group_size,
        'ari': ari,
        'nmi': nmi
    }

# ========== 6. 主流程 ==========
def main(data_path='clearing_data.csv', max_test_groups=None, model_path='model.pt'):
    """
    主流程
    max_test_groups: 限制测试集组数用于快速验证（None=全部）
    """
    print("=" * 70)
    print("自动清账系统 - 深度学习方案")
    print("=" * 70)
    
    print("\n加载数据...")
    df = pd.read_csv(data_path)
    
    print(f"数据形状: {df.shape}")
    print(f"唯一组数: {df['Group ID'].nunique()}")
    print(f"平均每组样本数: {len(df) / df['Group ID'].nunique():.2f}")
    
    print("\n划分训练集和测试集...")
    unique_groups = df['Group ID'].unique()
    train_groups_full, test_groups_full = train_test_split(
        unique_groups, test_size=0.2, random_state=42
    )
    
    if max_test_groups is not None and max_test_groups < len(test_groups_full):
        test_groups = np.random.RandomState(42).choice(test_groups_full, max_test_groups, replace=False)
    else:
        test_groups = test_groups_full
    
    df_data = df[df['Group ID'].isin(train_groups_full)].reset_index(drop=True)
    df_test = df[df['Group ID'].isin(test_groups)].reset_index(drop=True)
    
    print(f"训练数据: {len(df_data)} 行, {len(train_groups_full)} 组")
    print(f"测试集: {len(df_test)} 行, {len(test_groups)} 组")
    
    # 训练（或加载已有模型）
    import os
    if os.path.exists(model_path):
        print(f"\n发现已有模型 {model_path}，加载中...")
        preprocessor = ClearingDataPreprocessor()
        model, preprocessor = load_model(None, preprocessor, model_path)
    else:
        print("\n开始训练...")
        model, preprocessor = train_compatibility_model(
            df_data, epochs=50, batch_size=128, lr=0.001, patience=10
        )
        save_model(model, preprocessor, model_path)
    
    # 在测试集上预测
    df_test_no_label = df_test.drop('Group ID', axis=1)
    
    predicted_groups = find_clearing_groups(
        model, preprocessor, df_test_no_label,
        compatibility_threshold=0.6,
        max_group_size=10
    )
    
    # 评估
    metrics = evaluate_clearing_results(df_test, predicted_groups)
    
    # 展示示例
    print("\n" + "=" * 70)
    print("示例清账组（前 5 个）")
    print("=" * 70)
    
    for i, group in enumerate(predicted_groups[:5]):
        print(f"\n{'='*70}")
        print(f"组 {i+1} (共 {len(group)} 条凭证)")
        print(f"{'='*70}")
        
        group_df = df_test.iloc[group][[
            'Account', 'Company Code', 'Profit Center', 
            'Supplier', 'Text', 'Amount in doc. curr.'
        ]]
        
        print(group_df.to_string(index=False))
        print(f"\n总金额: {group_df['Amount in doc. curr.'].sum():.4f}")
        
        if 'Group ID' in df_test.columns:
            true_groups = df_test.iloc[group]['Group ID'].unique()
            print(f"真实 Group ID: {true_groups}")
    
    # 保存结果
    print("\n" + "=" * 70)
    print("保存结果...")
    
    df_test['Predicted_Group'] = -1
    for i, group in enumerate(predicted_groups):
        for idx in group:
            df_test.at[idx, 'Predicted_Group'] = i
    
    df_test.to_csv('clearing_results.csv', index=False)
    print("结果已保存到 clearing_results.csv")
    
    return model, preprocessor, predicted_groups, metrics

# ========== 运行 ==========
if __name__ == "__main__":
    model, preprocessor, predicted_groups, metrics = main(
        'clearing_data.csv',
        max_test_groups=50  # 先用少量测试集快速验证，设为 None 跑全部
    )