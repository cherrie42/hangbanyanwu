# -*- coding: utf-8 -*-
"""
机票航班延误预测项目 - 逻辑回归实现
使用新的数据集（train.csv和test.csv）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import fbeta_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import learning_curve
import joblib

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 数据加载
print("1. 加载数据...")
train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')

# 2. 特征工程
print("\n2. 特征工程...")
# 选择特征
features = [
    '出发机场', '到达机场', '航班编号', '飞机编号', '计划飞行时间',
    '计划起飞时刻', '航班月份', '计划到达时刻', '前序延误', '起飞间隔',
    '到达特情', '出发特情', '出发天气', '出发气温', '到达天气',
    '到达气温', '航空公司', '航班性质'
]

# 准备训练数据
X_train = train_data[features]
y_train = train_data['飞机延误目标']

# 准备测试数据
X_test = test_data[features]
y_test = test_data['飞机延误目标']

# 对数据进行标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 数据分割
print("\n3. 分割数据...")
X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

print(f"训练集样本数: {X_train_scaled.shape[0]}")
print(f"验证集样本数: {X_val_scaled.shape[0]}")
print(f"测试集样本数: {X_test_scaled.shape[0]}")

# 4. 逻辑回归模型训练
print("\n4. 训练逻辑回归模型...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# 在验证集上评估
val_pred = lr_model.predict(X_val_scaled)
f_score = fbeta_score(y_val, val_pred, beta=0.5)
acc = accuracy_score(y_val, val_pred)

print(f"验证集 F-score: {f_score:.4f}, 准确率: {acc:.4f}")

# 5. 模型调优
print("\n5. 模型调优...")
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(
    estimator=LogisticRegression(random_state=42, max_iter=1000),
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

# 最佳模型
best_lr = grid_search.best_estimator_
print(f"最佳参数: {grid_search.best_params_}")

# 可视化函数定义
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

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    plt.show()

def plot_feature_importance(model, feature_names):
    importance = np.abs(model.coef_[0])
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("特征重要性排序")
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# 6. 评估最佳模型
print("\n6. 评估最佳模型...")
# 绘制学习曲线
print("绘制学习曲线...")
plot_learning_curve(best_lr, X_train_scaled, y_train, "逻辑回归学习曲线")

# 在验证集上评估
val_pred = best_lr.predict(X_val_scaled)
val_pred_proba = best_lr.predict_proba(X_val_scaled)
f_score = fbeta_score(y_val, val_pred, beta=0.5)
acc = accuracy_score(y_val, val_pred)
print(f"调优后验证集 F-score: {f_score:.4f}, 准确率: {acc:.4f}")

# 绘制验证集的混淆矩阵
print("绘制混淆矩阵...")
plot_confusion_matrix(y_val, val_pred, "验证集混淆矩阵")

# 绘制ROC曲线
print("绘制ROC曲线...")
plot_roc_curve(y_val, val_pred_proba)

# 在测试集上评估
test_pred = best_lr.predict(X_test_scaled)
test_f_score = fbeta_score(y_test, test_pred, beta=0.5)
test_acc = accuracy_score(y_test, test_pred)
print(f"测试集 F-score: {test_f_score:.4f}, 准确率: {test_acc:.4f}")

# 7. 特征重要性分析
print("\n7. 特征重要性分析...")
plot_feature_importance(best_lr, features)

# 8. 保存模型
print("\n8. 保存模型...")
joblib.dump(best_lr, 'flight_delay_lr_model.pkl')
print("模型已保存为 flight_delay_lr_model.pkl")