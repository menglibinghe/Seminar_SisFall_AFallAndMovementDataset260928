import pandas as pd
import numpy as np
# 1. 导入神经网络模块
from sklearn.neural_network import MLPClassifier
# 2. 导入交叉验证模块
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import os
import warnings

# 忽略收敛警告（因为我们在做交叉验证，有时会提前停止，产生警告）
warnings.filterwarnings("ignore")

# ==========================================
# 1. 配置路径和参数
# ==========================================
Directory_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_csv\temp\SisFall_dataset_filtered"
csv_path = os.path.join(Directory_path , "Merged_SisFall_featuresReduced.csv")
label_column = 'Label'
random_state = 42

# 设置 10次 10折交叉验证
n_splits = 10
n_repeats = 10

# ==========================================
# 2. 数据读取与预处理
# ==========================================
print("正在读取数据...")
df = pd.read_csv(csv_path)
df[label_column] = df[label_column].astype(int)

# 剔除非数值列
exclude_cols = ['Move', 'Sub', 'Round']
feature_cols = [col for col in df.columns 
                if col not in exclude_cols + [label_column]]

print(f"检测到 {len(feature_cols)} 个有效数值特征。")

# 准备 X 和 y
y = df[label_column]
X = df[feature_cols]
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# ==========================================
# 3. 10次10折交叉验证 (Repeated Stratified K-Fold)
# ==========================================
print(f"正在进行 {n_repeats}次 {n_splits}折交叉验证 (ANN)...")
print("注意：神经网络训练较慢，请耐心等待...")

# 初始化交叉验证器
cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

# 用于存储每折的指标
list_se = []
list_sp = []
list_ac = []
list_acc = []

fold_count = 0
total_folds = n_splits * n_repeats

# 循环遍历每一个折
for train_index, test_index in cv.split(X, y):
    fold_count += 1
    
    # 1. 划分数据
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # 2. 数据标准化 (神经网络必须做标准化)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. 构建人工神经网络模型
    # hidden_layer_sizes=(64, 32): 两层隐藏层，第一层64个神经元，第二层32个神经元
    # max_iter=500: 最大迭代次数
    # early_stopping=True: 开启早停，如果验证集效果不再提升，提前结束训练
    # validation_fraction=0.1: 从训练集中划分10%作为验证集（用于早停）
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32), 
        activation='relu', 
        solver='adam', 
        alpha=0.0001, # L2正则化参数
        batch_size='auto', 
        learning_rate='adaptive',
        learning_rate_init=0.001, 
        max_iter=500, 
        shuffle=True, 
        random_state=random_state, 
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False
    )
    
    # 4. 训练模型
    model.fit(X_train_scaled, y_train)
    
    # 5. 预测
    predictions = model.predict(X_test_scaled)
    
    # 6. 计算混淆矩阵
    cm = confusion_matrix(y_test, predictions)
    
    # 提取 TP, TN, FP, FN
    # 确保混淆矩阵是 2x2 的
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
    else:
        # 极少数情况处理
        TN, FP, FN, TP = 0, 0, 0, 0
        
    # 7. 计算指标
    SE = TP / (TP + FN) if (TP + FN) > 0 else 0
    SP = TN / (TN + FP) if (TN + FP) > 0 else 0
    AC = (SE + SP) / 2
    acc = accuracy_score(y_test, predictions)
    
    # 存入列表
    list_se.append(SE)
    list_sp.append(SP)
    list_ac.append(AC)
    list_acc.append(acc)
    
    # 打印进度
    if fold_count % 20 == 0 or fold_count == total_folds:
        print(f"进度: {fold_count}/{total_folds} 折已完成...")

# ==========================================
# 4. 输出最终结果
# ==========================================
print("\n" + "="*45)
print("       人工神经网络 (ANN) 评估报告")
print("       (10次 10折交叉验证)")
print("="*45)

# 辅助函数：格式化输出
def fmt(mean_val, std_val):
    return f"{mean_val:.4f} ± {std_val:.4f}"

print(f"{'指标':<20} | {'结果 (均值 ± 波动)':<20}")
print("-" * 45)
print(f"{'平均灵敏度 (SE)':<20} | {fmt(np.mean(list_se), np.std(list_se))}")
print(f"{'平均特异度 (SP)':<20} | {fmt(np.mean(list_sp), np.std(list_sp))}")
print(f"{'平均准确率 (AC)':<20} | {fmt(np.mean(list_ac), np.std(list_ac))}")
print(f"{'标准准确率 (Acc)':<20} | {fmt(np.mean(list_acc), np.std(list_acc))}")
print("="*45)