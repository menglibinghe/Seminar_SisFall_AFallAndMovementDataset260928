import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold  # 1. 引入分层K折交叉验证
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import os

# ==========================================
# 1. 配置路径和参数
# ==========================================
Directory_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_csv\temp\SisFall_dataset_filtered"
csv_path = os.path.join(Directory_path , "Merged_SisFall_featuresReduced.csv")
label_column = 'Label'
random_state = 42
n_splits = 5  # 5折交叉验证

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
# 3. 定义交叉验证流程
# ==========================================
# 初始化交叉验证器
# shuffle=True 表示在分折前打乱数据
# random_state 保证每次运行的分折结果一致
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# 用于存储每一折的结果
fold_se_scores = []
fold_ac_scores = []
fold_acc_scores = []

print(f"\n开始 {n_splits}折交叉验证...")
print("-" * 40)

# 循环 5 次
for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    # 1. 划分当前折的训练/测试集
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # 2. 数据标准化 (注意：必须在划分后分别 fit 和 transform)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. 模型训练
    model = xgb.XGBClassifier(
        learning_rate=0.1,
        max_depth=3,
        n_estimators=100,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0  # 关闭单个模型的训练日志，保持界面整洁
    )
    
    model.fit(X_train_scaled, y_train)
    
    # 4. 预测与评估
    predictions = model.predict(X_test_scaled)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, predictions)
    
    # 提取 TP, TN, FP, FN
    # 注意：confusion_matrix 返回的顺序是 [[TN, FP], [FN, TP]]
    TN, FP, FN, TP = cm.ravel()
    
    # 计算指标
    SE = TP / (TP + FN) if (TP + FN) > 0 else 0
    SP = TN / (TN + FP) if (TN + FP) > 0 else 0
    AC = (SE + SP) / 2
    standard_acc = accuracy_score(y_test, predictions)
    
    # 保存结果
    fold_se_scores.append(SE)
    fold_ac_scores.append(AC)
    fold_acc_scores.append(standard_acc)
    
    print(f"第 {fold+1} 折 -> 灵敏度: {SE:.4f}, 平均准确率: {AC:.4f}, 标准准确率: {standard_acc:.4f}")

# ==========================================
# 4. 输出最终统计结果
# ==========================================
print("-" * 40)
print("   xgboost最终评估报告 (5折平均)")
print("-" * 40)

# 计算均值和标准差
mean_se = np.mean(fold_se_scores)
std_se = np.std(fold_se_scores)

mean_ac = np.mean(fold_ac_scores)
std_ac = np.std(fold_ac_scores)

mean_acc = np.mean(fold_acc_scores)
std_acc = np.std(fold_acc_scores)

# 格式化输出
print(f"{'指标':<15} | {'平均值':<10} | {'标准差':<10}")
print("-" * 40)
print(f"{'灵敏度 (SE)':<15} | {mean_se:.4f}     | {std_se:.4f}")
print(f"{'平均准确率 (AC)':<15} | {mean_ac:.4f}     | {std_ac:.4f}")
print(f"{'标准准确率 (Acc)':<15} | {mean_acc:.4f}     | {std_acc:.4f}")
print("=" * 40)

# 简单的稳定性评价
if std_se < 0.05:
    print("✅ 模型表现稳定 (灵敏度标准差 < 0.05)")
else:
    print("⚠️ 模型表现波动较大 (灵敏度标准差 >= 0.05)")