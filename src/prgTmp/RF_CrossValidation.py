import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置路径
# ==========================================
# 使用刚才生成的带有特征的文件
input_file = r'D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\csv2txtCombine\SisFall_With_Features.csv'

# ==========================================
# 2. 加载数据
# ==========================================
df = pd.read_csv(input_file)

# 检查列名 (调试用)
print("数据列名:", df.columns.tolist())

# ==========================================
# 3. 数据预处理
# ==========================================
# 定义特征列 (根据 extract_features 函数里生成的名字)
# 这里我们使用所有计算出来的统计特征
feature_cols = [col for col in df.columns if col not in ['label', 'subject']]

# 标签编码: 将 'D01' (跌倒) 设为 1, 其他设为 0
# 注意：SisFall 中 D01-D02 是跌倒，D03-D23 是日常活动
def map_label(act):
    if act.startswith('D01') or act.startswith('D02'): # 假设 D02 也是跌倒
        return 1
    else:
        return 0

df['target'] = df['label'].apply(map_label)

X = df[feature_cols]
y = df['target']
groups = df['subject']

# 检查类别平衡
print(f"数据总量: {len(df)}")
print(f"跌倒样本比例: {y.mean():.2f}")

# ==========================================
# 4. 模型训练与验证 (Group K-Fold)
# ==========================================
# 使用随机森林
model = RandomForestClassifier(
    n_estimators=100,   # 100棵树
    max_depth=10,       # 树的最大深度 (限制深度防止过拟合)
    random_state=42,
    n_jobs=-1           # 使用所有CPU核心
)

# 使用 GroupKFold (按受试者划分)
cv = GroupKFold(n_splits=10)

# 执行交叉验证
scoring = ['accuracy', 'sensitivity', 'specificity']

# 自定义 scorer (因为 sklearn 没有内置 sensitivity/specificity)
from sklearn.metrics import make_scorer, recall_score, precision_score

def sensitivity_score(y_true, y_pred):
    return recall_score(y_true, y_pred) # Sensitivity = Recall for positive class

def specificity_score(y_true, y_pred):
    # Specificity = Recall for negative class
    return recall_score(y_true, y_pred, pos_label=0)

scoring = {
    'Accuracy': 'accuracy',
    'Sensitivity': make_scorer(sensitivity_score),
    'Specificity': make_scorer(specificity_score)
}

print("\n开始 10折交叉验证 (按受试者划分)...")
scores = cross_validate(model, X, y, groups=groups, cv=cv, scoring=scoring, n_jobs=-1)

# ==========================================
# 5. 输出结果
# ==========================================
print("\n================ 结果 ================")
print(f"平均准确率 (Accuracy): {scores['test_Accuracy'].mean():.4f} (+/- {scores['test_Accuracy'].std() * 2:.4f})")
print(f"平均敏感度 (Sensitivity): {scores['test_Sensitivity'].mean():.4f}")
print(f"平均特异度 (Specificity): {scores['test_Specificity'].mean():.4f}")