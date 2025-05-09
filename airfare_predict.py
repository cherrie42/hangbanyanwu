# -*- coding: utf-8 -*-
"""
机票航班延误预测项目 - 多模型汇总实现
使用新的数据集（train.csv和test.csv）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import joblib

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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

# 定义模型列表和颜色列表
models = [
    GaussianNB(),
    LogisticRegression(random_state=42, max_iter=1000),
    SVC(kernel='rbf', probability=True, random_state=42),
    RandomForestClassifier(random_state=42, n_estimators=100),
    KNeighborsClassifier()
]

model_names = [
    "朴素贝叶斯",
    "逻辑回归",
    "支持向量机",
    "随机森林",
    "K近邻"
]

colors = [
    'blue',
    'green',
    'red',
    'orange',
    'purple'
]


# 定义训练集学习曲线绘图函数
def plot_training_learning_curves(models, X, y, model_names, colors):
    plt.figure(figsize=(14, 8))
    legend_handles = []

    marker_size = 10
    line_width = 2.5

    for i, model in enumerate(models):
        train_sizes, train_scores, _ = learning_curve(
            model, X, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

        # 绘制训练集曲线
        train_line, = plt.plot(train_sizes, train_mean, color=colors[i], marker='o',
                               markersize=marker_size, linewidth=line_width, alpha=0.8)
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color=colors[i])

        # 构建图例
        train_legend = plt.Line2D([0], [0], color=colors[i], marker='o', linestyle='-',
                                  markersize=marker_size, label=f'{model_names[i]}')
        legend_handles.append(train_legend)

    plt.xlabel('训练样本数', fontsize=13)
    plt.ylabel('准确率', fontsize=13)
    plt.title('多模型训练集学习曲线对比', fontsize=15, fontweight='bold')

    plt.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5),
               fontsize=10, frameon=True, shadow=True, ncol=1)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# 定义验证集学习曲线绘图函数
def plot_validation_learning_curves(models, X, y, model_names, colors):
    plt.figure(figsize=(14, 8))
    legend_handles = []

    marker_size = 10
    line_width = 2.5

    for i, model in enumerate(models):
        train_sizes, _, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )

        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # 绘制验证集曲线
        val_line, = plt.plot(train_sizes, val_mean, color=colors[i], marker='^',
                             markersize=marker_size, linewidth=line_width, alpha=0.8)
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color=colors[i])

        # 构建图例
        val_legend = plt.Line2D([0], [0], color=colors[i], marker='^', linestyle='-',
                                markersize=marker_size, label=f'{model_names[i]}')
        legend_handles.append(val_legend)

    plt.xlabel('训练样本数', fontsize=13)
    plt.ylabel('准确率', fontsize=13)
    plt.title('多模型验证集学习曲线对比', fontsize=15, fontweight='bold')

    plt.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5),
               fontsize=10, frameon=True, shadow=True, ncol=1)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# 定义绘制ROC曲线函数
def plot_roc_curves(models, X_val, y_val, model_names, colors):
    plt.figure(figsize=(10, 7))
    for i, model in enumerate(models):
        model.fit(X_train_scaled, y_train)
        val_pred_proba = model.predict_proba(X_val)
        fpr, tpr, _ = roc_curve(y_val, val_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=colors[i], lw=3, label=f'{model_names[i]} ROC曲线 (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率', fontsize=12)
    plt.ylabel('真阳性率', fontsize=12)
    plt.title('多模型ROC曲线对比', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# 训练并评估模型，收集结果
def evaluate_models(models, X_train, y_train, X_test, y_test, model_names):
    results = []

    for i, model in enumerate(models):
        print(f"\n训练 {model_names[i]} 模型...")
        model.fit(X_train, y_train)

        # 在测试集上评估
        test_pred = model.predict(X_test)

        # 计算各项指标
        accuracy = accuracy_score(y_test, test_pred)
        precision = precision_score(y_test, test_pred, average='binary')
        recall = recall_score(y_test, test_pred, average='binary')
        f1 = fbeta_score(y_test, test_pred, beta=1)

        # 存储结果
        results.append({
            '模型': model_names[i],
            '准确率': accuracy,
            '精确率': precision,
            '召回率': recall,
            'F1分数': f1
        })

        print(f"{model_names[i]} 测试集指标:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")

        # 保存模型
        joblib.dump(model, f'flight_delay_{model_names[i].lower().replace(" ", "_")}_model.pkl')
        print(f"{model_names[i]} 模型已保存为 flight_delay_{model_names[i].lower().replace(' ', '_')}_model.pkl")

    # 返回结果
    return pd.DataFrame(results)


# 绘制训练集学习曲线
plot_training_learning_curves(models, X_train_scaled, y_train, model_names, colors)

# 绘制验证集学习曲线
plot_validation_learning_curves(models, X_train_scaled, y_train, model_names, colors)

# 绘制ROC曲线
plot_roc_curves(models, X_val_scaled, y_val, model_names, colors)

# 评估模型并获取结果
results_df = evaluate_models(models, X_train_scaled, y_train, X_test_scaled, y_test, model_names)

# 打印模型对比表格
print("\n===== 模型性能对比 =====")
print(results_df.to_string(index=False))

# 可视化精确率和召回率对比
plt.figure(figsize=(14, 7))
metrics = ['精确率', '召回率']

# 确保所有图表使用相同的排序（基于F1分数）
sorted_indices = results_df['F1分数'].argsort()[::-1]
sorted_models = results_df.iloc[sorted_indices]['模型'].tolist()
sorted_colors = [colors[j] for j in sorted_indices]

for i, metric in enumerate(metrics):
    plt.subplot(1, 2, i + 1)
    sorted_values = results_df.iloc[sorted_indices][metric].tolist()

    # 减小柱形宽度，增加间距
    bars = plt.bar(sorted_models, sorted_values, color=sorted_colors, width=0.4)

    plt.title(f'各模型{metric}对比', fontsize=13)
    plt.ylabel(metric, fontsize=12)

    # 根据指标类型智能调整Y轴范围
    if metric == '精确率':
        min_val = max(min(sorted_values) - 0.01, 0)  # 确保最小值不低于0
        max_val = min(max(sorted_values) + 0.01, 1.0)  # 确保最大值不超过1
        plt.ylim(min_val, max_val)
        # 动态设置Y轴刻度
        step = (max_val - min_val) / 5
        plt.yticks(np.arange(min_val, max_val + step / 2, step))
    else:
        min_val = max(min(sorted_values) - 0.01, 0)
        max_val = min(max(sorted_values) + 0.01, 1.0)
        plt.ylim(min_val, max_val)
        step = (max_val - min_val) / 5
        plt.yticks(np.arange(min_val, max_val + step / 2, step))

    plt.xticks(rotation=45, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 智能调整标签位置
    for bar in bars:
        height = bar.get_height()
        label_pos = height + 0.002  # 默认偏移量
        # 如果标签位置超出Y轴范围，调整到柱形内部
        if label_pos > plt.gca().get_ylim()[1]:
            label_pos = height - 0.005
        plt.text(bar.get_x() + bar.get_width() / 2., label_pos,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# 可视化F1分数对比（同样优化）
plt.figure(figsize=(12, 6))
sorted_f1_scores = results_df.iloc[sorted_indices]['F1分数'].tolist()

bars = plt.bar(sorted_models, sorted_f1_scores, color=sorted_colors, width=0.4)
plt.title('各模型F1分数对比', fontsize=14, fontweight='bold')
plt.ylabel('F1分数', fontsize=12)

# 智能调整Y轴范围
min_f1 = max(min(sorted_f1_scores) - 0.01, 0)
max_f1 = min(max(sorted_f1_scores) + 0.01, 1.0)
plt.ylim(min_f1, max_f1)
step_f1 = (max_f1 - min_f1) / 5
plt.yticks(np.arange(min_f1, max_f1 + step_f1 / 2, step_f1))

plt.xticks(rotation=45, fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 智能调整标签位置
for bar in bars:
    height = bar.get_height()
    label_pos = height + 0.002
    if label_pos > plt.gca().get_ylim()[1]:
        label_pos = height - 0.005
    plt.text(bar.get_x() + bar.get_width() / 2., label_pos,
             f'{height:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

