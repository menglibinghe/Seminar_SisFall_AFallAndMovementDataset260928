import numpy as np
import pandas as pd
import os
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import time  # 1. 导入时间模块

# ==========================================
# 1. 配置路径和参数
# ==========================================
Directory_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_csv\temp\SisFall_dataset_filtered"
csv_path = os.path.join(Directory_path , "Merged_SisFall_featuresReduced.csv")

label_column = 'Label'
n_splits = 10
n_repeats = 10

# Q-Learning 核心参数
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2
EPISODES = 50

# ==========================================
# 2. 数据读取与预处理
# ==========================================
print("正在读取数据...")
df = pd.read_csv(csv_path)
df[label_column] = df[label_column].astype(int)

exclude_cols = ['Move', 'Sub', 'Round']
feature_cols = [col for col in df.columns 
                if col not in exclude_cols + [label_column]]

print(f"检测到 {len(feature_cols)} 个有效数值特征。")

y = df[label_column]
X = df[feature_cols]
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# ==========================================
# 3. 定义 Q-Learning 智能体类
# ==========================================
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha, gamma, epsilon):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def get_state_index(self, state_vector):
        discretized = tuple((state_vector * 10).astype(int))
        return hash(discretized) % self.n_states

    def choose_action(self, state_idx):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state_idx])

    def learn(self, state_idx, action, reward, next_state_idx):
        predict = self.q_table[state_idx, action]
        target = reward + self.gamma * np.max(self.q_table[next_state_idx])
        self.q_table[state_idx, action] = predict + self.alpha * (target - predict)

# ==========================================
# 4. 定义训练与评估函数
# ==========================================
def train_and_evaluate_qlearning(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    agent = QLearningAgent(n_states=10000, n_actions=2, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)
    
    # 训练阶段
    for episode in range(EPISODES):
        indices = np.random.permutation(len(X_train_scaled))
        for i in indices:
            state_vec = X_train_scaled[i]
            state_idx = agent.get_state_index(state_vec)
            true_label = y_train.iloc[i]
            
            action = agent.choose_action(state_idx)
            reward = 10 if action == true_label else -10
            
            next_idx = np.random.randint(0, len(X_train_scaled))
            next_state_idx = agent.get_state_index(X_train_scaled[next_idx])
            
            agent.learn(state_idx, action, reward, next_state_idx)

    # 测试阶段
    agent.epsilon = 0 
    predictions = []
    for i in range(len(X_test_scaled)):
        state_vec = X_test_scaled[i]
        state_idx = agent.get_state_index(state_vec)
        action = agent.choose_action(state_idx)
        predictions.append(action)
    
    return np.array(predictions)

# ==========================================
# 5. 执行 10次10折交叉验证 (带计时)
# ==========================================
print(f"\n开始 Q-Learning {n_repeats}次{n_splits}折交叉验证...")
print("注意：Q-Learning 训练较慢，请耐心等待...")

rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

# 存储指标
list_se = []
list_sp = []
list_ac = []
list_acc = []

fold_count = 0
total_folds = n_splits * n_repeats

# 2. 记录程序开始时间
start_time = time.time() 

for train_index, test_index in rkf.split(X, y):
    fold_count += 1
    
    # 3. 记录每一折的开始时间
    fold_start = time.time()
    
    # 划分数据
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # 训练并预测
    predictions = train_and_evaluate_qlearning(X_train, y_train, X_test, y_test)
    
    # 计算指标
    TP = ((predictions == 1) & (y_test.values == 1)).sum()
    TN = ((predictions == 0) & (y_test.values == 0)).sum()
    FP = ((predictions == 1) & (y_test.values == 0)).sum()
    FN = ((predictions == 0) & (y_test.values == 1)).sum()
    
    SE = TP / (TP + FN) if (TP + FN) > 0 else 0
    SP = TN / (TN + FP) if (TN + FP) > 0 else 0
    AC = (SE + SP) / 2
    Acc = (TP + TN) / len(y_test)
    
    list_se.append(SE)
    list_sp.append(SP)
    list_ac.append(AC)
    list_acc.append(Acc)
    
    # 4. 计算并打印当前折耗时
    fold_end = time.time()
    fold_duration = fold_end - fold_start
    
    # 打印进度和单折耗时
    if fold_count % 10 == 0 or fold_count == 1:
        print(f"进度: {fold_count}/{total_folds} 折 | 本折耗时: {fold_duration:.2f}秒")

# 5. 记录程序结束时间
end_time = time.time()
total_duration = end_time - start_time

# ==========================================
# 6. 输出最终报告与耗时统计
# ==========================================
print("\n" + "="*50)
print("       Q-Learning 最终评估报告")
print("="*50)

def fmt(mean_val, std_val):
    return f"{mean_val:.4f} ± {std_val:.4f}"

print(f"{'指标':<20} | {'结果 (均值 ± 波动)':<20}")
print("-" * 50)
print(f"{'平均灵敏度 (SE)':<20} | {fmt(np.mean(list_se), np.std(list_se))}")
print(f"{'平均特异度 (SP)':<20} | {fmt(np.mean(list_sp), np.std(list_sp))}")
print(f"{'平均准确率 (AC)':<20} | {fmt(np.mean(list_ac), np.std(list_ac))}")
print(f"{'标准准确率 (Acc)':<20} | {fmt(np.mean(list_acc), np.std(list_acc))}")
print("-" * 50)

# 6. 打印总耗时
print(f"⏱️  总运行时长: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
print(f"⚡  平均每折耗时: {total_duration/total_folds:.2f} 秒")
print("="*50)