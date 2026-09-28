import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# 1. 读取数据
file_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_csv\temp\SisFall_dataset_filtered\Merged_SisFall_FeaturesReduced103.csv"
df = pd.read_csv(file_path)

# 2. 数据清洗：强制只保留数字列作为特征
# 这一步能防止 'D01' 这种字符串报错
df_numeric = df.select_dtypes(include=[np.number])

# 检查是否丢失了标签列（如果标签是字符串，select_dtypes 会把它去掉）
# 假设你的标签列叫 'Label' 或者是最后一列，这里需要小心处理
# 这里假设最后一列是标签，且可能是字符串
target_column_name = df.columns[-1] # 获取最后一列的名字

X = df_numeric.drop(columns=[target_column_name], errors='ignore') # 从数字列中剔除标签（如果标签在数字列里）
y = df[target_column_name] # 标签从原始 df 取，保留原始类型

# 如果标签是文本（如 'Fall', 'ADL'），需要编码成数字
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

print(f"正在处理 {X.shape[1]} 个数字特征...")

# 3. 计算互信息
# 注意：mutual_info_classif 对 NaN (空值) 也很敏感，如果有空值需要填充
# X = X.fillna(0)  # 如果有空值，取消注释这行
mi_scores = mutual_info_classif(X, y)

# 4. 整理数据
mi_series = pd.Series(mi_scores, index=X.columns)
mi_series = mi_series.sort_values(ascending=False)

# 5. 绘图 (旋转90度版本)
plt.figure(figsize=(20, 6))
bars = plt.bar(mi_series.index, mi_series.values)
plt.title('Mutual Information with Target Label', fontsize=15)
plt.xlabel('Features', fontsize=12)
plt.ylabel('MI Score', fontsize=12)

# 关键：旋转X轴文字
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout() # 自动调整间距
plt.show()