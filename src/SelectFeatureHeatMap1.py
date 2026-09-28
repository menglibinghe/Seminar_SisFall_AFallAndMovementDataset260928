import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取数据
file_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_csv\temp\SisFall_dataset_filtered\Merged_SisFall_FeaturesReduced103.csv"
df = pd.read_csv(file_path)

# 2. 提取特征
features = df.iloc[:, :-4]
features = features.select_dtypes(include=['number'])

# 3. 计算全量相关系数矩阵
corr_matrix = features.corr()

# 获取所有列名（严格保持CSV原始顺序）
all_cols = corr_matrix.columns
mid_point = len(all_cols) // 2

# 分割列
first_half_cols = all_cols[:mid_point]
second_half_cols = all_cols[mid_point:]

# --- 核心参数设置 ---
font_size = 5      # 统一字体大小为 5
fig_width = 50     # 画布宽度
fig_height = 25    # 画布高度（单图高度）
bottom_margin = 0.15 # 底部留白

# ==========================================
# 第一张图：前50%特征 (行) vs 全部特征 (列)
# ==========================================
plt.figure(figsize=(fig_width, fig_height))

# 【关键修正】使用 .loc 显式指定行和列的顺序
# 行：前半段特征，列：全部特征
data_part1 = corr_matrix.loc[first_half_cols, all_cols]

sns.heatmap(
    data_part1,
    cmap='coolwarm',
    center=0,
    cbar_kws={"shrink": 0.5},
    square=False,
    linewidths=0,
    # 不开启注释（100多列太挤），如果要看数值请把 annot=True 打开
    annot=False
)

plt.title(f"Correlation Heatmap: First 50% Features (Rows) vs All Features (Cols)\n(Total Rows: {len(first_half_cols)}, Total Cols: {len(all_cols)})", fontsize=6, pad=20)
plt.xlabel("All Features (Columns)", fontsize=10)
plt.ylabel("First 50% Features (Rows)", fontsize=10)

# 设置横轴刻度标签：旋转90度，字体大小5，右对齐防止截断
plt.xticks(
    ticks=np.arange(len(all_cols)) + 0.5,
    labels=all_cols,
    rotation=45,
    fontsize=font_size,
    ha='right'
)

# 设置纵轴刻度标签：不旋转，字体大小5
plt.yticks(
    ticks=np.arange(len(first_half_cols)) + 0.5,
    labels=first_half_cols,
    rotation=0,
    fontsize=font_size
)

# 调整布局：底部留白 0.15
plt.subplots_adjust(bottom=bottom_margin, left=0.1, right=0.95, top=0.95)

# 保存图片（使用 tight_layout 防止截断）
plt.savefig("Heatmap_Part1_First50.png", dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# 第二张图：后50%特征 (行) vs 全部特征 (列)
# ==========================================
plt.figure(figsize=(fig_width, fig_height))

# 【关键修正】使用 .loc 显式指定行和列的顺序
# 行：后半段特征，列：全部特征
data_part2 = corr_matrix.loc[second_half_cols, all_cols]

sns.heatmap(
    data_part2,
    cmap='coolwarm',
    center=0,
    cbar_kws={"shrink": 0.5},
    square=False,
    linewidths=0,
    annot=False
)

plt.title(f"Correlation Heatmap: Second 50% Features (Rows) vs All Features (Cols)\n(Total Rows: {len(second_half_cols)}, Total Cols: {len(all_cols)})", fontsize=7, pad=20)
plt.xlabel("All Features (Columns)", fontsize=10)
plt.ylabel("Second 50% Features (Rows)", fontsize=10)

# 设置横轴刻度标签
plt.xticks(
    ticks=np.arange(len(all_cols)) + 0.5,
    labels=all_cols,
    rotation=45,
    fontsize=font_size,
    ha='right'
)

# 设置纵轴刻度标签
plt.yticks(
    ticks=np.arange(len(second_half_cols)) + 0.5,
    labels=second_half_cols,
    rotation=0,
    fontsize=font_size
)

# 调整布局
plt.subplots_adjust(bottom=bottom_margin, left=0.1, right=0.95, top=0.95)

# 保存图片
plt.savefig("Heatmap_Part2_Second50.png", dpi=300, bbox_inches='tight')
plt.show()