import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取文件
Directory_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\t1\SisFall_dataset_filtered"
csv_path = os.path.join(Directory_path, "F05_SA01_R01.csv")

df = pd.read_csv(csv_path)

# === 关键修改：数据切片 ===
# 我们只看第 5000 行到第 6000 行（这通常是跌倒发生的区域，或者是某次剧烈动作）
# 如果这里看不到跌倒，可以尝试 10000:11000 或者其他范围
start_idx = 500    #0
end_idx =   2000       #20001
df_slice = df.iloc[start_idx+1:end_idx]

# 画图
plt.figure(figsize=(12, 6))
plt.plot(df_slice.index, df_slice.iloc[:,1], label='Acc_x', color='red')
plt.plot(df_slice.index, df_slice.iloc[:,2], label='Acc_y', color='green')
plt.plot(df_slice.index, df_slice.iloc[:,3], label='Acc_z', color='blue')
plt.plot(df_slice.index, df_slice.iloc[:,4], label='Gyro_x', color='yellow')
plt.plot(df_slice.index, df_slice.iloc[:,5], label='Gyro_y', color='brown')
plt.plot(df_slice.index, df_slice.iloc[:,6], label='Gyro_z', color='purple')

plt.title(f'Filtered Acc_x Waveform (Zoomed In: {start_idx} - {end_idx})')
plt.xlabel('Time range')
plt.ylabel('Acceleration (Acc_x)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()