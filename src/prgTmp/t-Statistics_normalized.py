import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ==========================================
# 1. 路径与参数配置
# ==========================================
Directory_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_csv\temp\SisFall_dataset_filtered"
csv_path = os.path.join(Directory_path, "Merged_SisFall_FeaturesReduced103.csv")

# 2. 读取数据
df = pd.read_csv(csv_path)

# 3. 数据清洗与标准化 (关键步骤)
# 分离特征与标签
X = df.iloc[:, :-4]
y = df['Label']

# --- 关键步骤：强制对所有特征进行标准化 ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将标准化后的数据转换回 DataFrame，方便绘图
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 4. 数据重塑 (Reshape)
# 添加标签列
X_scaled_df['Label'] = y
# 转换为长格式
df_melted = X_scaled_df.melt(id_vars='Label', var_name='Feature', value_name='Value')

# 5. 绘制分布箱图
plt.figure(figsize=(20, 10))

# 使用 seaborn 绘制箱图
# 注意：这里 x='Feature'，y='Value'，hue='Label'
# 这样每个特征都会有一组箱线图，并且按 Label 分色
sns.boxplot(data=df_melted, x='Feature', y='Value', hue='Label')

# 6. 优化图表显示
plt.title(f'Box Plot Distribution of NORMALIZED Features from {os.path.basename(csv_path)} ')
plt.xticks(rotation=90)  # 特征名称太长，旋转90度显示
plt.tight_layout()  # 自动调整布局防止遮挡
plt.show()