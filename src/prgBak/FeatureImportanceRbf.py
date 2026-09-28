import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. 数据读取与预处理
# ==========================================
file_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_csv\temp\SisFall_dataset_filtered\Merged_SisFall_FeaturesReduced103.csv"
df = pd.read_csv(file_path)

# 分离特征(X)和标签(y)
# 假设最后一列是标签，前面全是特征
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 数据清洗：确保特征全是数字（防止之前的 'D01' 报错）
X = X.apply(pd.to_numeric, errors='coerce')
X = X.dropna(axis=1)  # 删除无法转为数字的列

# 标签编码：如果 y 是文字（如 'Fall'），转为数字
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

# 划分训练集和测试集 (RFE 只需要在训练集上跑，或者在全量数据上跑)
# 这里我们在训练集上跑，以模拟真实场景
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==========================================
# 2. 递归特征消除 (RFE)
# ==========================================
print(f"原始特征数量: {X_train.shape[1]}")

# A. 定义基模型 (Estimator)
# RFE 需要一个模型来判断特征的重要性，这里用随机森林
estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

# B. 定义 RFE
# n_features_to_select=20: 目标保留 20 个特征
# step=0.1: 每次迭代淘汰 10% 的特征 (也可以设为整数，如 5，表示每次淘汰5个)
selector = RFE(estimator, n_features_to_select=50, step=0.1)

# C. 拟合数据 (这是最耗时的一步，请耐心等待)
print("正在执行递归特征消除... (可能需要一点时间)")
selector = selector.fit(X_train, y_train)

# ==========================================
# 3. 结果展示
# ==========================================
# 获取被选中的特征名称
selected_features = X_train.columns[selector.support_]

print("\n=== RFE 筛选出的 20 个最佳特征 ===")
for i, feature in enumerate(selected_features):
    print(f"{i+1}. {feature}")

# 查看特征的排名 (1 表示被选中，数字越大表示越早被淘汰)
# ranking_df = pd.DataFrame({'Feature': X_train.columns, 'Ranking': selector.ranking_})
# print(ranking_df.sort_values(by='Ranking').head(20))