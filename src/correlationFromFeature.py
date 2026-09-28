import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold 
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ==========================================
# 1. 路径与参数配置
# ==========================================
Directory_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_csv\temp\SisFall_dataset_filtered"
csv_path = os.path.join(Directory_path, "Merged_SisFall_featuresReduced - 60.csv")

# 假设文件在当前目录
df = pd.read_csv(csv_path)

# --- 修改开始 ---

# 方法 A: 自动选择所有数字类型的列 (推荐)
numeric_df = df.select_dtypes(include=[np.number])

# 方法 B: 如果你只想手动排除特定的列，也可以用 drop (例如排除 'Sub' 和 'Move')
# df_clean = df.drop(columns=['Sub', 'Move', 'Round']) 

# 计算相关性矩阵 (只针对数字列)
corr_matrix = numeric_df.corr()

# 查看前18个特征与 Label 的相关性
# 注意：这里使用 numeric_df 确保列索引对应的是数字列
print("前18个特征与 Label 的相关性：")
print(corr_matrix['Label'].iloc[:60])

# --- 修改结束 ---