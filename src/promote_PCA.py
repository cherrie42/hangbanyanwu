import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 加载数据集
data = pd.read_csv('../data/train.csv')

# 提取特征和目标变量
X = data.drop(['Unnamed: 0', '飞机延误目标'], axis=1)
y = data['飞机延误目标']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 进行 PCA 降维，保留 95% 的方差
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 训练随机森林模型
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_pca, y_train)

# 预测并评估模型
y_pred = rf.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"PCA 降维后随机森林模型的准确率: {accuracy}")

# 保存优化后的模型
joblib.dump(rf, 'flight_delay_rf_model_pca_optimized-2.pkl')