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
# 导入 3D 绘图工具
from mpl_toolkits.mplot3d import Axes3D 

# ==========================================
# 1. 路径与参数配置
# ==========================================
Directory_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_filtered"
#csv_path = os.path.join(Directory_path , "Merged_SisFall_featuresReduced.csv")
csv_path = os.path.join(Directory_path, "Merged_SisFall_featuresReduced - 3.csv")

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


# 创建画布
fig = plt.figure(figsize=(12, 10))
# 添加一个 3D 的坐标轴
ax = fig.add_subplot(111, projection='3d')

# 绘制 3D 散点图
# X_tsne[:, 0] -> X轴
# X_tsne[:, 1] -> Y轴
# X_tsne[:, 2] -> Z轴 (这是新增的第三维)
scatter = ax.scatter(
    X_tsne[:, 0], 
    X_tsne[:, 1], 
    X_tsne[:, 2], 
    c=y,                # 根据标签上色
    cmap='Set1',        # 颜色映射
    s=50,               # 点的大小
    alpha=0.6           # 透明度，防止遮挡太严重
)

# 设置标签和标题
ax.set_title(f'3D t-SNE Visualization of SisFall Features from {os.path.basename(csv_path)}')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3') # 设置 Z 轴标签

# 添加图例 (稍微复杂一点，需要手动创建)
legend1 = ax.legend(*scatter.legend_elements(), title="Label")
ax.add_artist(legend1)

plt.show()


