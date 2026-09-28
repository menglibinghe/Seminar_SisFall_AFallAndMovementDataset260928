import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
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

# 标签编码：如果 y 是字符串，转为数字
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

# ==========================================
# 2. 核心步骤：基于 XGBoost 的特征筛选
# ==========================================

# A. 初始化 XGBoost 模型
# eval_metric='logloss' 是二分类任务常用的评估指标
xgb_model = XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1, eval_metric='logloss')

# B. 初始化选择器
# 将 estimator 替换为 xgb_model，其他逻辑与随机森林完全一致
selector = SelectFromModel(estimator=xgb_model, max_features=60, threshold=-np.inf)

# C. 拟合并筛选
# XGBoost 会在训练过程中自动计算特征重要性，并筛选出前20个
selector.fit(X, y)

# ==========================================
# 3. 获取结果
# ==========================================

# 获取被选中的特征名称
selected_features = X.columns[selector.get_support()]

print("✅ XGBoost 筛选完成！被选中的 20 个关键特征是：")
print(selected_features.tolist())

# 直接得到筛选后的数据
X_selected = selector.transform(X)
print(f"\n原始数据维度: {X.shape}")
print(f"筛选后数据维度: {X_selected.shape}")