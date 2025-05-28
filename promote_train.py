# -*- coding: utf-8 -*-
"""
机票航班延误预测项目 - 随机森林优化版（含类别平衡处理）
参考文档：随机森林召唤率和F解答.docx
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import fbeta_score, accuracy_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from sklearn.model_selection import learning_curve
from imblearn.over_sampling import SMOTE  # 新增：SMOTE过采样库
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ----------------------
# 1. 数据加载（与原始代码一致）
# ----------------------
print("1. 加载数据...")
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

# 2. 特征工程（与原始代码一致）
print("\n2. 特征工程...")
features = [
    '出发机场', '到达机场', '航班编号', '飞机编号', '计划飞行时间',
    '计划起飞时刻', '航班月份', '计划到达时刻', '前序延误', '起飞间隔',
    '到达特情', '出发特情', '出发天气', '出发气温', '到达天气',
    '到达气温', '航空公司', '航班性质'
]
X_train = train_data[features]
y_train = train_data['飞机延误目标']
X_test = test_data[features]
y_test = test_data['飞机延误目标']

# ----------------------
# 3. 数据分割（新增分层抽样）
# ----------------------
print("\n3. 分割数据...")
# 分层抽样保持类别分布（原始代码未使用stratify，优化后新增）
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"训练集样本数: {X_train.shape[0]}")
print(f"验证集样本数: {X_val.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")

# ----------------------
# 4. 新增：SMOTE过采样处理
# ----------------------
print("\n4. 应用SMOTE过采样...")
smote = SMOTE(random_state=42)  # 初始化SMOTE
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)  # 生成平衡后的训练数据
print(f"过采样后训练集样本数: {X_resampled.shape[0]}")

# ----------------------
# 5. 基础模型训练（与原始代码对比）
# ----------------------
print("\n5. 训练基础随机森林模型...")
base_rf = RandomForestClassifier(random_state=42, n_estimators=100)
base_rf.fit(X_train, y_train)  # 原始训练集（未过采样）

# 基础模型评估
base_val_pred = base_rf.predict(X_val)
base_f1 = f1_score(y_val, base_val_pred)
base_recall = recall_score(y_val, base_val_pred)
base_acc = accuracy_score(y_val, base_val_pred)
print(f"基础模型 - 验证集: 准确率={base_acc:.4f}, F1={base_f1:.4f}, 召回率={base_recall:.4f}")

# ----------------------
# 6. 带类别权重的网格搜索调优（优化核心）
# ----------------------
print("\n6. 模型调优（含类别权重+过采样）...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', 'balanced_subsample']  # 新增：类别权重参数
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='f1_weighted',  # 优化指标为F1加权（针对不平衡数据）
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_resampled, y_resampled)  # 在过采样数据上搜索参数
best_rf = grid_search.best_estimator_
print(f"最佳参数: {grid_search.best_params_}")

# ----------------------
# 7. 模型评估（与原始代码一致，新增召回率指标）
# ----------------------
print("\n7. 评估调优后模型...")
# 验证集评估
tuned_val_pred = best_rf.predict(X_val)
tuned_val_pred_proba = best_rf.predict_proba(X_val)
tuned_f1 = f1_score(y_val, tuned_val_pred)
tuned_recall = recall_score(y_val, tuned_val_pred)
tuned_acc = accuracy_score(y_val, tuned_val_pred)
print(f"调优模型 - 验证集: 准确率={tuned_acc:.4f}, F1={tuned_f1:.4f}, 召回率={tuned_recall:.4f}")

# 测试集评估（与原始代码一致）
test_pred = best_rf.predict(X_test)
test_f_score = fbeta_score(y_test, test_pred, beta=0.5)
test_acc = accuracy_score(y_test, test_pred)
print(f"测试集 F-score: {test_f_score:.4f}, 准确率: {test_acc:.4f}")

# ----------------------
# 8. 可视化函数（与原始代码一致）
# ----------------------
def plot_learning_curve(estimator, X, y, title):
    # 函数内容与原始代码一致，略
    ...

def plot_confusion_matrix(y_true, y_pred, title):
    # 函数内容与原始代码一致，略
    ...

def plot_roc_curve(y_true, y_pred_proba):
    # 函数内容与原始代码一致，略
    ...

def plot_feature_importance(model, feature_names):
    # 函数内容与原始代码一致，略
    ...

# 绘制学习曲线（使用过采样后的训练数据）
print("绘制学习曲线...")
plot_learning_curve(best_rf, X_resampled, y_resampled, "优化后随机森林学习曲线")

# 绘制混淆矩阵和ROC曲线（与原始代码一致）
print("绘制混淆矩阵...")
plot_confusion_matrix(y_val, tuned_val_pred, "验证集混淆矩阵")

print("绘制ROC曲线...")
plot_roc_curve(y_val, tuned_val_pred_proba)

# 特征重要性分析（与原始代码一致）
print("\n8. 特征重要性分析...")
plot_feature_importance(best_rf, features)

# ----------------------
# 9. 保存模型（与原始代码一致）
# ----------------------
print("\n9. 保存模型...")
joblib.dump(best_rf, 'flight_delay_rf_model_optimized.pkl')
print("优化模型已保存")