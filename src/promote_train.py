# -*- coding: utf-8 -*-
"""
机票航班延误预测项目 - 随机森林优化版（含类别平衡处理）
参考文档：随机森林召唤率和F解答.docx
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import fbeta_score, accuracy_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns
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
# 8. 可视化函数
# ----------------------
def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='训练集得分', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, val_mean, label='验证集得分', color='green', marker='o')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='green')
    plt.xlabel('训练样本数')
    plt.ylabel('准确率')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, title):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

# 绘制基础模型和优化后模型的学习曲线
plot_learning_curve(base_rf, X_train, y_train, "基础随机森林学习曲线")
plot_learning_curve(best_rf, X_resampled, y_resampled, "优化后随机森林学习曲线")

# 绘制基础模型和优化后模型的ROC曲线
base_val_pred_proba = base_rf.predict_proba(X_val)
plot_roc_curve(y_val, base_val_pred_proba, "基础随机森林ROC曲线")
plot_roc_curve(y_val, tuned_val_pred_proba, "优化后随机森林ROC曲线")

# 特征重要性分析（与原始代码一致）
def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("特征重要性排序")
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

print("\n8. 特征重要性分析...")
plot_feature_importance(best_rf, features)

# ----------------------
# 9. 保存模型（与原始代码一致）
# ----------------------
print("\n9. 保存模型...")
joblib.dump(best_rf, 'flight_delay_rf_model_optimized.pkl')
print("优化模型已保存")