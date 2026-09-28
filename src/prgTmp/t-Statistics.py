import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
# 注意：这里只导入了基础库，不需要 RepeatedStratifiedGroupKFold
from sklearn.model_selection import StratifiedGroupKFold 
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns # 用于美化图表

# ==========================================
# 1. 路径与参数配置
# ==========================================
Directory_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_csv\temp\SisFall_dataset_filtered"
#csv_path = os.path.join(Directory_path , "Merged_SisFall_featuresReduced.csv")
csv_path = os.path.join(Directory_path, "Merged_SisFall_FeaturesReduced103.csv")

# 2. 读取数据
# 请将 'your_file_path.csv' 替换为你实际的文件路径
df = pd.read_csv(csv_path)

# 3. 数据清洗：分离特征与标签
# 观察数据结构：最后4列不是特征，Label是标签
# 我们需要剔除最后4列（Move, Sub, Round, Label）作为特征X
# 提取倒数第4列（即Label列）作为标签y

# 方法：选取除了最后4列之外的所有列作为特征
X = df.iloc[:, :-4] 

# 选取倒数第4列作为标签 (Label)
y = df['Label'] 

# 4. 数据重塑 (Reshape)
# Seaborn 绘图通常需要数据是 "长格式" (Long Format)
# 原始数据是 "宽格式" (Wide Format: 60列特征)
# 我们需要把60列特征堆叠成2列：一列叫 "Feature"，一列叫 "Value"

# 添加标签列到特征数据中，方便绘图时分组
X['Label'] = y 

# 将数据从宽格式转换为长格式
# id_vars=['Label'] 表示保留Label列不变，其他所有列都堆叠到一起
df_melted = X.melt(id_vars='Label', var_name='Feature', value_name='Value')

# 5. 绘制分布箱图 (Box Plot)
# 设置画布大小，因为有60个特征，需要横向空间大一点
plt.figure(figsize=(20, 10)) 

# 使用 Seaborn 绘制箱图
# x=Feature, y=Value, hue=Label 表示按标签不同颜色显示
sns.boxplot(data=df_melted, x='Feature', y='Value', hue='Label')

# 6. 优化图表显示
plt.title(f'Box Plot Distribution of Features from :{os.path.basename(csv_path)} by Label')
plt.xticks(rotation=90) # 特征名称太长，旋转90度显示
plt.tight_layout() # 自动调整布局防止遮挡
plt.show()