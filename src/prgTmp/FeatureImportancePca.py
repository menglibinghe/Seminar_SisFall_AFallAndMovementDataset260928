import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. 数据读取与预处理
# ==========================================
file_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_csv\temp\SisFall_dataset_filtered\Merged_SisFall_FeaturesReduced103.csv"
df = pd.read_csv(file_path)

# 分离特征(X)和标签(y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 数据清洗：确保特征全是数字
X = X.apply(pd.to_numeric, errors='coerce')
X = X.dropna(axis=1) 

# 标签编码
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==========================================
# 2. 核心步骤：PCA 降维与提取特征来源
# ==========================================

# A. 数据标准化 (PCA 必须步骤！因为不同特征的单位和量级不同)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# B. 执行 PCA
# n_components=20 表示保留前 20 个主成分
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# ==========================================
# 3. 提取原始特征来源（载荷矩阵）
# ==========================================

# 获取载荷矩阵 (Loadings)
# pca.components_ 的形状是 (20, 原始特征数)，每一行代表一个主成分
# 我们需要把它转置 (.T)，变成 (原始特征数, 20)，这样每一列就对应一个新主成分
loadings = pca.components_.T

# 创建包含原始列名的 DataFrame，方便查看
loadings_df = pd.DataFrame(
    loadings, 
    index=X.columns,  # 行索引是原始特征名（如 min_Ax, max_Ax）
    columns=[f'PC{i+1}' for i in range(pca.n_components_)] # 列名是新主成分名
)

# 查看前 5 个原始特征在 20 个新主成分中的权重
print("=== 原始特征在新主成分中的权重（前5行） ===")
print(loadings_df.head())

# ==========================================
# 4. 深度分析：找出每个新主成分的主要来源
# ==========================================
print("\n=== 解析每个新主成分（PC）的核心来源 ===")
for i in range(pca.n_components_):
    pc_name = f'PC{i+1}'
    # 提取当前主成分中，权重绝对值最大的前 5 个原始特征
    top_features = loadings_df[pc_name].abs().sort_values(ascending=False).head(10)
    
    print(f"\n【{pc_name}】 主要由以下原始特征组合而成：")
    for feature_name, weight in top_features.items():
        # 获取原始权重的正负号
        original_weight = loadings_df.loc[feature_name, pc_name]
        print(f"  - {feature_name}: {original_weight:.4f}")

# 打印方差解释比例，看看这20个主成分保留了多少原始信息
print(f"\n前 20 个主成分累计保留了 {pca.explained_variance_ratio_.sum()*100:.2f}% 的原始数据信息。")