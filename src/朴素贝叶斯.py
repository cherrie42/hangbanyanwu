# -*- coding: utf-8 -*-
"""
机票航班延误预测项目 - 朴素贝叶斯实现
使用新的数据集（train.csv和test.csv）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import learning_curve
import joblib

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 数据加载
print("1. 加载数据...")
train_data = pd.read_csv('./data/train.csv', encoding='gbk')
test_data = pd.read_csv('./data/test.csv', encoding='gbk')
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

# 4. 朴素贝叶斯模型训练
print("\n4. 训练朴素贝叶斯模型...")
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

# 在验证集上评估
val_pred = nb_model.predict(X_val_scaled)
f_score = fbeta_score(y_val, val_pred, beta=0.5)
acc = accuracy_score(y_val, val_pred)

print(f"验证集 F-score: {f_score:.4f}, 准确率: {acc:.4f}")

# 朴素贝叶斯没有太多需要调优的参数，所以去掉了调优步骤
best_nb = nb_model


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


def plot_decision_regions(X, y, model, title):
    # 由于特征维度较高，选择前两个特征进行可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
    plt.title(title)
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.show()


# 6. 评估最佳模型
print("\n6. 评估模型...")
# 绘制学习曲线
print("绘制学习曲线...")
plot_learning_curve(best_nb, X_train_scaled, y_train, "朴素贝叶斯学习曲线")

# 在验证集上评估
val_pred = best_nb.predict(X_val_scaled)
val_pred_proba = best_nb.predict_proba(X_val_scaled)
f_score = fbeta_score(y_val, val_pred, beta=0.5)
acc = accuracy_score(y_val, val_pred)
print(f"验证集 F-score: {f_score:.4f}, 准确率: {acc:.4f}")

# 绘制验证集的混淆矩阵
print("绘制混淆矩阵...")
plot_confusion_matrix(y_val, val_pred, "验证集混淆矩阵")

# 绘制ROC曲线
print("绘制ROC曲线...")
plot_roc_curve(y_val, val_pred_proba)

# 绘制决策边界（使用前两个特征）
print("绘制决策边界...")
plot_decision_regions(X_val_scaled[:, :2], y_val, best_nb, "朴素贝叶斯决策边界可视化")

# 在测试集上评估
test_pred = best_nb.predict(X_test_scaled)
test_f_score = fbeta_score(y_test, test_pred, beta=0.5)
test_acc = accuracy_score(y_test, test_pred)
print(f"测试集 F-score: {test_f_score:.4f}, 准确率: {test_acc:.4f}")

# 8. 保存模型
print("\n8. 保存模型...")
joblib.dump(best_nb, './model/flight_delay_nb_model.pkl')
print("模型已保存为 ./model/flight_delay_nb_model.pkl")