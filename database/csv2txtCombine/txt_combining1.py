import os
import pandas as pd
import glob

# 配置路径
folder_path = r'C:\Users\tanchao\pythonPrj\pythonPrj2026\seminar_C\excel_combining\SisFall_dataset_csv'
output_file = 'Merged_SisFall_Final.txt'

def merge_sisfall_data_optimized(folder_path, output_file):
    search_path = os.path.join(folder_path, "*.csv")
    file_list = glob.glob(search_path)

    if not file_list:
        print(f"⚠️ 未找到文件。")
        return

    print(f"🔍 找到 {len(file_list)} 个文件，开始流式处理...")

    current_sn = 1
    is_first_file = True # 用于控制写入表头

    for i, file_path in enumerate(file_list):
        file_name = os.path.basename(file_path)
        
        # 1. 解析文件名
        name_parts = file_name.split('_')
        if len(name_parts) < 3:
            continue

        di_code = name_parts[0]
        subject_id = name_parts[1]
        ri_trial = name_parts[2].split('.')[0]

        try:
            # --- 关键改进：增加 low_memory=False 和 error_bad_lines 处理 ---
            # 如果 SisFall 文件末尾有分号，可以尝试 sep=None, engine='python'
            df = pd.read_csv(file_path, header=None, usecols=[0, 1, 2, 3], on_bad_lines='skip')
            
            if df.empty:
                continue

            # 2. 数据清洗
            # 确保删除的是第一列（索引0），保留 1, 2, 3 轴数据
            df = df.iloc[:, 1:] 
            df.columns = ['acc1_x', 'acc1_y', 'acc1_z']

            # 3. 添加列
            df['subject'] = subject_id
            df['Activity'] = di_code
            df['Trial'] = ri_trial
            df.insert(0, 'sn', range(current_sn, current_sn + len(df)))

            # 4. 流式追加写入磁盘，不占用内存列表
            # 如果是第一个文件，写入表头(header=True)；否则追加(mode='a', header=False)
            df.to_csv(output_file, 
                      sep='\t', 
                      index=False, 
                      header=is_first_file, 
                      mode='a', 
                      encoding='utf-8')

            current_sn += len(df)
            is_first_file = False

            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1} / {len(file_list)} 个文件...")

        except Exception as e:
            print(f"❌ 处理文件 {file_name} 时发生错误: {e}")

    print(f"\n✅ 处理完成！累计写入行数: {current_sn - 1}")

if __name__ == "__main__":
    # 执行前如果旧文件存在，先删除，防止多次运行重复追加
    if os.path.exists(output_file):
        os.remove(output_file)
    merge_sisfall_data_optimized(folder_path, output_file)