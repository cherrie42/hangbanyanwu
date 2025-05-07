# -*- coding: utf-8 -*-
"""
机票航班延误预测项目 - 随机森林实现
使用新的数据集（train.csv和test.csv）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. 数据加载
print("1. 加载数据...")
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

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

# 3. 数据分割（从训练集中分出验证集）
print("\n3. 分割数据...")
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"训练集样本数: {X_train.shape[0]}")
print(f"验证集样本数: {X_val.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")

# 4. 随机森林模型训练
print("\n4. 训练随机森林模型...")
rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)
rf_clf.fit(X_train, y_train)

# 在验证集上评估
val_pred = rf_clf.predict(X_val)
f_score = fbeta_score(y_val, val_pred, beta=0.5)
acc = accuracy_score(y_val, val_pred)

print(f"验证集 F-score: {f_score:.4f}, 准确率: {acc:.4f}")

# 5. 模型调优
print("\n5. 模型调优...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# 最佳模型
best_rf = grid_search.best_estimator_
print(f"最佳参数: {grid_search.best_params_}")

# 6. 评估最佳模型
print("\n6. 评估最佳模型...")
# 在验证集上评估
val_pred = best_rf.predict(X_val)
f_score = fbeta_score(y_val, val_pred, beta=0.5)
acc = accuracy_score(y_val, val_pred)
print(f"调优后验证集 F-score: {f_score:.4f}, 准确率: {acc:.4f}")

# 在测试集上评估
test_pred = best_rf.predict(X_test)
test_f_score = fbeta_score(y_test, test_pred, beta=0.5)
test_acc = accuracy_score(y_test, test_pred)
print(f"测试集 F-score: {test_f_score:.4f}, 准确率: {test_acc:.4f}")

# 7. 特征重要性分析
print("\n7. 特征重要性分析...")
feature_importance = best_rf.feature_importances_
features_names = features

plt.figure(figsize=(12, 8))
plt.barh(range(len(features_names)), feature_importance, align='center')
plt.yticks(range(len(features_names)), features_names)
plt.xlabel('特征重要性')
plt.title('随机森林特征重要性')
plt.tight_layout()
plt.show()

# 8. 保存模型
print("\n8. 保存模型...")
joblib.dump(best_rf, 'flight_delay_rf_model.pkl')
print("模型已保存为 flight_delay_rf_model.pkl")