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
csv_path = os.path.join(Directory_path, "Merged_SisFall_featuresReduced - 60.csv")

# 假设文件在当前目录
df = pd.read_csv(csv_path)

# 假设最后一列是 'Label'，前面所有列都是特征
# X 是特征数据 (Features)
X = df.drop(columns=['Label', 'Sub', 'Round', 'Move']) # 去掉标签和元信息列
# y 是标签 (Labels)，用于画图时上色
y = df['Label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # 将数据转换为均值为0，方差为1

# 初始化 t-SNE 模型
tsne = TSNE(
    n_components=3,      # 降维成 2维，如果你想 3D 图就设为 3
    perplexity=30,      # 一个重要的参数，大致等于你期望的最近邻点数。通常在 5-50 之间。
    max_iter=1000,         # 迭代次数，越大越精确（通常1000足够）
    random_state=42,     # 随机种子，保证每次结果一样
    learning_rate='auto' # 学习率
)

# 开始降维计算
X_tsne = tsne.fit_transform(X_scaled)


#按 "Label" (标签) 着色：想看跌倒（Label=1）和日常活动（Label=0）是否在特征空间中分得开：
plt.figure(figsize=(10, 8))
# 使用 seaborn 绘制，hue=y 表示根据标签不同上不同颜色
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='Set1', s=100)
plt.title('t-SNE Visualization of SisFall Features (Colored by Label)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Label')
plt.show()


