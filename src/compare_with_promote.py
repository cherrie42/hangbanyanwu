import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
import joblib
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 模拟数据加载函数
def load_data():
    # 如果有实际数据文件，可替换为真实数据加载逻辑
    print("加载数据...")
    
    # 模拟航班延误数据集
    np.random.seed(42)
    n_samples = 26000  # 总样本数
    
    # 生成特征
    data = {
        '出发机场': np.random.choice(['北京', '上海', '广州', '深圳', '成都', '杭州'], n_samples),
        '到达机场': np.random.choice(['北京', '上海', '广州', '深圳', '成都', '杭州'], n_samples),
        '航班编号': np.random.randint(1000, 9999, n_samples),
        '飞机编号': np.random.choice([f'B-{i:04d}' for i in range(100, 500)], n_samples),
        '计划飞行时间': np.random.normal(180, 60, n_samples),
        '计划起飞时刻': np.random.randint(0, 24, n_samples),
        '航班月份': np.random.randint(1, 13, n_samples),
        '计划到达时刻': np.random.randint(0, 24, n_samples),
        '前序延误': np.random.exponential(20, n_samples),
        '起飞间隔': np.random.normal(15, 5, n_samples),
        '到达特情': np.random.choice(['无', '天气影响', '流量控制', '机械故障'], n_samples),
        '出发特情': np.random.choice(['无', '天气影响', '流量控制', '机械故障'], n_samples),
        '出发天气': np.random.choice(['晴', '多云', '雨', '雪', '雾'], n_samples),
        '出发气温': np.random.normal(20, 10, n_samples),
        '到达天气': np.random.choice(['晴', '多云', '雨', '雪', '雾'], n_samples),
        '到达气温': np.random.normal(20, 10, n_samples),
        '航空公司': np.random.choice(['国航', '东航', '南航', '海航', '川航', '厦航'], n_samples),
        '航班性质': np.random.choice(['国内', '国际', '地区'], n_samples),
        '飞机延误目标': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # 不平衡数据，20%延误
    }
    
    df = pd.DataFrame(data)
    
    # 确保部分特征与延误有更强的相关性
    df.loc[df['出发天气'].isin(['雨', '雪', '雾']), '飞机延误目标'] = np.random.choice(
        [0, 1], size=df[df['出发天气'].isin(['雨', '雪', '雾'])].shape[0], p=[0.5, 0.5]
    )
    
    df.loc[df['前序延误'] > 30, '飞机延误目标'] = np.random.choice(
        [0, 1], size=df[df['前序延误'] > 30].shape[0], p=[0.3, 0.7]
    )
    
    # 分割训练集和测试集
    train_size = int(n_samples * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    return train_df, test_df

# 特征工程函数
def feature_engineering(train_df, test_df):
    print("特征工程...")
    
    # 合并训练集和测试集进行统一编码
    all_data = pd.concat([train_df, test_df], axis=0)
    
    # 特征选择
    features = [
        '出发机场', '到达机场', '航班编号', '飞机编号', '计划飞行时间',
        '计划起飞时刻', '航班月份', '计划到达时刻', '前序延误', '起飞间隔',
        '到达特情', '出发特情', '出发天气', '出发气温', '到达天气',
        '到达气温', '航空公司', '航班性质'
    ]
    
    # 对分类特征进行编码
    for col in features:
        if all_data[col].dtype == 'object':
            le = LabelEncoder()
            all_data[col] = le.fit_transform(all_data[col])
    
    # 分离训练集和测试集
    train_df_encoded = all_data.iloc[:len(train_df)]
    test_df_encoded = all_data.iloc[len(train_df):]
    
    X_train = train_df_encoded[features]
    y_train = train_df_encoded['飞机延误目标']
    X_test = test_df_encoded[features]
    y_test = test_df_encoded['飞机延误目标']
    
    return X_train, y_train, X_test, y_test

# 1. 基础模型训练（类似train.py）
def train_base_model(X_train, y_train, X_test, y_test):
    print("\n训练基础模型（类似train.py）...")
    
    # 分割训练集和验证集
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )
    
    # 创建基础随机森林模型
    base_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    
    # 训练模型
    base_model.fit(X_train_split, y_train_split)
    
    # 评估模型
    y_pred = base_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    
    print(f"基础模型验证集: 准确率={accuracy:.4f}, F1={f1:.4f}, 召回率={recall:.4f}")
    
    return base_model, X_val, y_val

# 2. 优化模型训练（类似promote_train.py）
def train_optimized_model(X_train, y_train, X_test, y_test):
    print("\n训练优化模型（类似promote_train.py）...")
    
    # 分割训练集和验证集（保持相同的随机种子以确保数据分割一致）
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )
    
    # 应用SMOTE过采样处理不平衡数据
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_split, y_train_split)
    
    # 创建优化的随机森林模型（添加类别权重和其他优化）
    optimized_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',  # 平衡类别权重
        random_state=42
    )
    
    # 训练模型
    optimized_model.fit(X_resampled, y_resampled)
    
    # 评估模型
    y_pred = optimized_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    
    print(f"优化模型验证集: 准确率={accuracy:.4f}, F1={f1:.4f}, 召回率={recall:.4f}")
    
    return optimized_model, X_val, y_val

# 3. 绘制学习曲线对比
def compare_learning_curves(base_model, optimized_model, X_train, y_train):
    print("生成学习曲线对比图...")
    plt.figure(figsize=(12, 6))
    
    # 基础模型学习曲线
    train_sizes, train_scores_base, val_scores_base = learning_curve(
        base_model, X_train, y_train, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
    )
    train_mean_base = np.mean(train_scores_base, axis=1)
    train_std_base = np.std(train_scores_base, axis=1)
    val_mean_base = np.mean(val_scores_base, axis=1)
    val_std_base = np.std(val_scores_base, axis=1)
    
    # 优化模型学习曲线
    train_sizes, train_scores_opt, val_scores_opt = learning_curve(
        optimized_model, X_train, y_train, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
    )
    train_mean_opt = np.mean(train_scores_opt, axis=1)
    train_std_opt = np.std(train_scores_opt, axis=1)
    val_mean_opt = np.mean(val_scores_opt, axis=1)
    val_std_opt = np.std(val_scores_opt, axis=1)
    
    # 绘制基础模型曲线
    plt.plot(train_sizes, train_mean_base, 'o-', color='blue', label='基础模型-训练集')
    plt.fill_between(train_sizes, train_mean_base - train_std_base, 
                     train_mean_base + train_std_base, alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean_base, 'o-', color='lightblue', label='基础模型-验证集')
    plt.fill_between(train_sizes, val_mean_base - val_std_base, 
                     val_mean_base + val_std_base, alpha=0.1, color='lightblue')
    
    # 绘制优化模型曲线
    plt.plot(train_sizes, train_mean_opt, 'o-', color='green', label='优化模型-训练集')
    plt.fill_between(train_sizes, train_mean_opt - train_std_opt, 
                     train_mean_opt + train_std_opt, alpha=0.1, color='green')
    plt.plot(train_sizes, val_mean_opt, 'o-', color='lightgreen', label='优化模型-验证集')
    plt.fill_between(train_sizes, val_mean_opt - val_std_opt, 
                     val_mean_opt + val_std_opt, alpha=0.1, color='lightgreen')
    
    plt.xlabel('训练样本数')
    plt.ylabel('准确率')
    plt.title('基础模型 vs 优化模型学习曲线对比')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图片
    if not os.path.exists('images'):
        os.makedirs('images')
    plt.savefig('images/learning_curve_comparison.png')
    plt.show()

# 4. 绘制ROC曲线对比
def compare_roc_curves(base_model, optimized_model, X_val, y_val):
    print("生成ROC曲线对比图...")
    plt.figure(figsize=(10, 8))
    
    # 基础模型ROC
    y_score_base = base_model.predict_proba(X_val)[:, 1]
    fpr_base, tpr_base, _ = roc_curve(y_val, y_score_base)
    roc_auc_base = auc(fpr_base, tpr_base)
    
    # 优化模型ROC
    y_score_opt = optimized_model.predict_proba(X_val)[:, 1]
    fpr_opt, tpr_opt, _ = roc_curve(y_val, y_score_opt)
    roc_auc_opt = auc(fpr_opt, tpr_opt)
    
    # 绘制基础模型ROC
    plt.plot(fpr_base, tpr_base, lw=2, 
             label=f'基础模型 (AUC = {roc_auc_base:.3f})')
    
    # 绘制优化模型ROC
    plt.plot(fpr_opt, tpr_opt, lw=2, 
             label=f'优化模型 (AUC = {roc_auc_opt:.3f})')
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (1-特异性)')
    plt.ylabel('真阳性率 (敏感性)')
    plt.title('基础模型 vs 优化模型 ROC曲线对比')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('images/roc_curve_comparison.png')
    plt.show()
    
    # 打印AUC提升百分比
    auc_improvement = (roc_auc_opt - roc_auc_base) / roc_auc_base * 100
    print(f"AUC值从 {roc_auc_base:.3f} 提升至 {roc_auc_opt:.3f}，提升了 {auc_improvement:.2f}%")

# 5. 性能指标对比表
def print_performance_comparison(base_model, optimized_model, X_val, y_val):
    print("\n==== 模型性能指标对比 ====")
    
    # 基础模型预测
    y_pred_base = base_model.predict(X_val)
    base_acc = accuracy_score(y_val, y_pred_base)
    base_f1 = f1_score(y_val, y_pred_base)
    base_recall = recall_score(y_val, y_pred_base)
    
    # 优化模型预测
    y_pred_opt = optimized_model.predict(X_val)
    opt_acc = accuracy_score(y_val, y_pred_opt)
    opt_f1 = f1_score(y_val, y_pred_opt)
    opt_recall = recall_score(y_val, y_pred_opt)
    
    # 创建对比表格
    metrics_df = pd.DataFrame({
        '指标': ['准确率', 'F1分数', '召回率'],
        '基础模型': [base_acc, base_f1, base_recall],
        '优化模型': [opt_acc, opt_f1, opt_recall],
        '提升': [f"{(opt_acc-base_acc)/base_acc*100:.2f}%", 
                f"{(opt_f1-base_f1)/base_f1*100:.2f}%",
                f"{(opt_recall-base_recall)/base_recall*100:.2f}%"]
    })
    
    print(metrics_df.to_string(index=False))

# 主函数
def main():
    # 加载数据
    train_df, test_df = load_data()
    
    # 特征工程
    X_train, y_train, X_test, y_test = feature_engineering(train_df, test_df)
    
    # 训练基础模型
    base_model, X_val, y_val = train_base_model(X_train, y_train, X_test, y_test)
    
    # 训练优化模型
    optimized_model, _, _ = train_optimized_model(X_train, y_train, X_test, y_test)
    
    # 可视化对比
    compare_learning_curves(base_model, optimized_model, X_train, y_train)
    compare_roc_curves(base_model, optimized_model, X_val, y_val)
    print_performance_comparison(base_model, optimized_model, X_val, y_val)
    
    print("\n可视化对比完成！")
    print("学习曲线对比图已保存至: images/learning_curve_comparison.png")
    print("ROC曲线对比图已保存至: images/roc_curve_comparison.png")

if __name__ == "__main__":
    main()