import pandas as pd

def reduce_dataset_step_by_step(input_file, output_file):
    print(f"正在读取文件: {input_file} ...")

    # 1. 读取数据
    try:
        df = pd.read_csv(input_file, sep='\t')
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        return

    print(f"原始数据行数: {len(df)}")

    # 确保列名没有多余空格
    df.columns = df.columns.str.strip()

    # -------------------------------------------------
    # 步骤 1: 删除 Subject 中的 SA06~SA23 和 SE06~SE15
    # -------------------------------------------------
    remove_subjects = []
    # 生成 SA06 到 SA23
    for i in range(6, 24):
        remove_subjects.append(f'SA{i:02d}') # :02d 确保格式为 06, 07... 23

    # 生成 SE06 到 SE15
    for i in range(6, 16):
        remove_subjects.append(f'SE{i:02d}')

    # 筛选：保留不在 remove_subjects 列表中的行
    initial_count = len(df)
    df = df[~df['subject'].isin(remove_subjects)]
    print(f"步骤1后 (清洗Subject): 删除了 {initial_count - len(df)} 行, 剩余 {len(df)} 行")

    # -------------------------------------------------
    # 步骤 2: 删除 Activity 中的 D06~D19 和 F06~F19
    # -------------------------------------------------
    remove_activities = []
    # 生成 D06 到 D19
    for i in range(6, 20):
        remove_activities.append(f'D{i:02d}')

    # 生成 F06 到 F19
    for i in range(6, 20):
        remove_activities.append(f'F{i:02d}')

    # 筛选：保留不在 remove_activities 列表中的行
    initial_count = len(df)
    df = df[~df['Activity'].isin(remove_activities)]
    print(f"步骤2后 (清洗Activity): 删除了 {initial_count - len(df)} 行, 剩余 {len(df)} 行")

    # -------------------------------------------------
    # 步骤 3: 删除 Trial 中的 R02~R05
    # -------------------------------------------------
    remove_trials = ['R02', 'R03', 'R04', 'R05']

    # 筛选：保留不在 remove_trials 列表中的行
    initial_count = len(df)
    df = df[~df['Trial'].isin(remove_trials)]
    print(f"步骤3后 (清洗Trial): 删除了 {initial_count - len(df)} 行, 剩余 {len(df)} 行")

    # -------------------------------------------------
    # 4. 保存结果
    # -------------------------------------------------
    df.to_csv(output_file, sep='\t', index=False)
    print(f"处理完成！最终文件已生成: {output_file}")

# 执行函数
input_filename = 'OrgMerged_SisFall.txt'
output_filename = 'reduced_SisFall.txt'

reduce_dataset_step_by_step(input_filename, output_filename)