import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 读取数据
train_x = pd.read_csv("./data/train.csv")
train_y = train_x["飞机延误目标"].values
del(train_x["飞机延误目标"])

test_x = pd.read_csv("./data/test.csv")
test_y = test_x["飞机延误目标"].values
del(test_x["飞机延误目标"])

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    train_x, train_y, test_size=0.2, random_state=42
)

# 初始化随机森林分类器
rf_model = RandomForestClassifier(random_state=42)

# 定义网格搜索参数
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 使用网格搜索进行参数优化
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=2
)

# 训练模型
grid_search.fit(X_train, y_train)

# 打印最佳参数
print("最佳参数:", grid_search.best_params_)
print("最佳得分:", grid_search.best_score_)

# 使用最佳模型在验证集上进行预测
best_model = grid_search.best_estimator_
val_predictions = best_model.predict(X_val)

# 输出验证集性能指标
print("\n验证集性能评估:")
print("准确率:", accuracy_score(y_val, val_predictions))
print("\n分类报告:")
print(classification_report(y_val, val_predictions))

# 特征重要性分析
feature_importance = pd.DataFrame({
    'feature': train_x.columns,
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\n特征重要性排序:")
print(feature_importance.head(10))

# 在测试集上进行预测
test_predictions = best_model.predict(test_x)
print("\n测试集准确率:", accuracy_score(test_y, test_predictions))