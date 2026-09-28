import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. 数据读取与预处理
# ==========================================
file_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_csv\temp\SisFall_dataset_filtered\Merged_SisFall_FeaturesReduced103.csv"
df = pd.read_csv(file_path)

# 分离特征(X)和标签(y)
# 假设最后一列是标签，其余是特征
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 数据清洗：确保特征全是数字
X = X.apply(pd.to_numeric, errors='coerce')
X = X.dropna(axis=1) 

# 标签编码：如果 y 是字符串，转为数字
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

# ==========================================
# 2. 核心步骤：SelectFromModel 特征筛选
# ==========================================

# A. 初始化随机森林模型
# 注意：这里 n_estimators 设大一点（如200），能让重要性评估更稳定
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

# B. 初始化选择器
# prefit=False 表示我们要先训练模型再筛选（这是默认行为，写出来更清晰）
# max_features=20 表示我们强制只保留重要性最高的 50 个特征
selector = SelectFromModel(estimator=rf, max_features=50, threshold=-np.inf)

# C. 拟合并筛选
# 这一步会自动训练随机森林，计算重要性，然后只保留前50个
selector.fit(X, y)

# ==========================================
# 3. 获取结果
# ==========================================

# 获取被选中的特征名称
selected_features = X.columns[selector.get_support()]

print("✅ 筛选完成！被选中的 50 个关键特征是：")
print(selected_features.tolist())

# 如果你想直接得到筛选后的数据（只包含这50列）：
X_selected = selector.transform(X)
print(f"\n原始数据维度: {X.shape}")
print(f"筛选后数据维度: {X_selected.shape}")