import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score  # 新增评估指标
import joblib

# 加载数据集
train_data = pd.read_csv('./data/train.csv', encoding='gbk')

# 提取特征和目标变量（注意：删除无关列'Unnamed: 0'）
X = data.drop(['Unnamed: 0', '飞机延误目标'], axis=1)
y = data['飞机延误目标']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # 分层抽样保持类别分布
)

# 定义随机森林模型（初始参数）
rf = RandomForestClassifier(random_state=42)

# 扩展参数搜索空间，加入class_weight
param_grid = {
    'n_estimators': [50, 100, 200, 300],  # 扩大范围
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': [None, 'balanced', 'balanced_subsample']  # 新增类别权重参数
}

# 使用F1分数作为优化目标（或改用'recall'侧重召回率）
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='f1_weighted',  # 优化加权F1分数（处理不平衡数据）
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# 获取最佳模型
best_rf = grid_search.best_estimator_

# 预测并评估多指标
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"最佳参数: {grid_search.best_params_}")
print(f"准确率: {accuracy:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1分数: {f1:.4f}")

# 保存模型
joblib.dump(best_rf, './model/flight_delay_rf_model_optimized-2.pkl')