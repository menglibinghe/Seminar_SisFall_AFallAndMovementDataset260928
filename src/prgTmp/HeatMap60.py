import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 读取文件
Directory_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_filtered"
csv_path = os.path.join(Directory_path, "Merged_SisFall_featuresReduced - 99.csv")

# 读取数据
df = pd.read_csv(csv_path)
# 提取纯数值特征（去掉最后4列非特征列）
features = df.iloc[:, :-4]

# 计算相关系数矩阵
corr_matrix = features.corr()

# 绘制热力图
plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title('Feature Correlation Heatmap (Redundancy Check)')

# --- 关键修改：调整坐标轴字体大小 ---
# labelsize 控制刻度文字大小，rotation 控制旋转角度
plt.tick_params(axis='x', which='major', labelsize=8, rotation=90) 
plt.tick_params(axis='y', which='major', labelsize=8, rotation=0) 

# 调整子图参数（如果字体变小了，bottom的留白也可以适当减小，比如0.15）
plt.subplots_adjust(bottom=0.15) 
plt.show()