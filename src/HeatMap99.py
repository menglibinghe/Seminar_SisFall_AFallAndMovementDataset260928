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

# --- 关键修改：筛选纵轴为前30列特征，横轴为全部特征 ---
# 取相关矩阵的前30行（对应前30个特征的纵轴），所有列（对应全部特征的横轴）
#corr_matrix_subset = corr_matrix.iloc[:30, :]

# 绘制热力图
plt.figure(figsize=(28, 28))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, xticklabels=True, yticklabels=True)
plt.title('Feature Correlation Heatmap (Redundancy Check)')

# --- 关键修改：调整坐标轴字体大小 ---
# labelsize 控制刻度文字大小，rotation 控制旋转角度
plt.tick_params(axis='x', which='major', labelsize=7, rotation=90) 
plt.tick_params(axis='y', which='major', labelsize=6, rotation=0) 

# 调整子图参数（如果字体变小了，bottom的留白也可以适当减小，比如0.15）
plt.subplots_adjust(bottom=0.2) 
plt.show()

print("当前参与计算的特征总数:", features.shape[1]) # <-- 打印验证特征数