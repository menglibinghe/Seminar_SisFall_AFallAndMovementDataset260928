import pandas as pd
import os

# 1. 定义路径
base_dir = r'D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\csv2txtCombine'
input_file = os.path.join(base_dir, 'reduced_SisFall.txt')
output_file = os.path.join(base_dir, 'reduced_SisFall_AdultOnly.txt')   

# 2. 读取数据
# 注意：根据你之前的截图，文件是以 Tab 分隔的
print(f"📖 正在读取: {input_file} ...")
df = pd.read_csv(input_file, sep='\t')

# 3. 筛选数据
# 只保留 subject 以 'SA' 开头的行 (年轻人)
# 排除掉 'SE' (老年人)
df_adult = df[df['subject'].str.startswith('SA')].copy()

# 4. 保存新文件
print(f"⚡ 筛选完成！原始数据 {len(df)} 行，筛选后 {len(df_adult)} 行。")
print(f"💾 正在保存为: {output_file} ...")
df_adult.to_csv(output_file, sep='\t', index=False)

print("✅ 任务完成！")