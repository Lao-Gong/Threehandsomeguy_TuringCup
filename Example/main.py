# main.py
"""
股票E收益率预测 - 主程序
重构版：主要逻辑调用utils.py中的工具函数
"""

import utils
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
# ==============================================================================
def check_train_test_consistency(X_train, X_test, y_train, y_test, top_features=5):
    """
    检查训练集和测试集的一致性
    """
    print("\n检查训练集和测试集的一致性")
    print("="*60)
    
    # 1. 检查目标变量的统计特性
    print("目标变量统计:")
    print(f"训练集 - 均值: {y_train.mean():.6f}, 标准差: {y_train.std():.6f}")
    print(f"测试集 - 均值: {y_test.mean():.6f}, 标准差: {y_test.std():.6f}")
    print(f"训练集/测试集均值比: {y_train.mean()/y_test.mean():.3f}")
    
    # 2. 检查特征在训练集和测试集上的相关性
    print("\n特征相关性对比 (前5个特征):")
    
    # 选择前几个特征
    features_to_check = X_train.columns[:min(top_features, len(X_train.columns))]
    
    for feature in features_to_check:
        train_corr = X_train[feature].corr(y_train)
        test_corr = X_test[feature].corr(y_test)
        
        print(f"{feature:20} | 训练集: {train_corr:7.4f} | 测试集: {test_corr:7.4f} | 差异: {abs(train_corr-test_corr):7.4f}")
    
    # 3. 检查训练集和测试集的时间范围
    print("\n时间范围检查:")
    print("注意：训练集应该在前，测试集在后（时间序列）")
    
    return True

def test_baseline_models(y_train, y_test):
    """
    测试几个简单的基准模型
    """
    print("\n基线模型测试")
    print("="*60)
    
    # 1. 朴素预测：用前一期收益率作为预测
    y_pred_naive = y_test.shift(1).fillna(0)
    corr_naive = np.corrcoef(y_test.iloc[1:], y_pred_naive.iloc[1:])[0, 1]
    
    # 2. 零预测：总是预测0
    y_pred_zero = pd.Series(0, index=y_test.index)
    corr_zero = np.corrcoef(y_test, y_pred_zero)[0, 1]
    
    # 3. 均值预测：用训练集的均值
    y_pred_mean = pd.Series(y_train.mean(), index=y_test.index)
    corr_mean = np.corrcoef(y_test, y_pred_mean)[0, 1]
    
    print("基准模型性能 (相关系数):")
    print(f"1. 朴素模型 (前一期收益率): {corr_naive:.6f}")
    print(f"2. 零预测 (总是预测0): {corr_zero:.6f}")
    print(f"3. 均值预测 (训练集均值): {corr_mean:.6f}")
    
    return {
        'naive': corr_naive,
        'zero': corr_zero,
        'mean': corr_mean
    }

def check_feature_stability(X_train, X_test, y_train, y_test, feature_name):
    """
    检查特定特征与目标关系是否稳定
    """
    print(f"\n检查特征 '{feature_name}' 的稳定性")
    print("="*60)
    
    # 计算训练集和测试集上的相关性
    train_corr = X_train[feature_name].corr(y_train)
    test_corr = X_test[feature_name].corr(y_test)
    
    print(f"训练集相关性: {train_corr:.6f}")
    print(f"测试集相关性: {test_corr:.6f}")
    print(f"相关性变化: {abs(train_corr - test_corr):.6f}")
    
    # 绘制散点图
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 训练集散点图
        axes[0].scatter(X_train[feature_name].iloc[:1000], y_train.iloc[:1000], alpha=0.3, s=1)
        axes[0].set_xlabel(feature_name)
        axes[0].set_ylabel('Return5min')
        axes[0].set_title(f'训练集: {feature_name} vs 收益率\n相关系数: {train_corr:.4f}')
        
        # 测试集散点图
        axes[1].scatter(X_test[feature_name].iloc[:1000], y_test.iloc[:1000], alpha=0.3, s=1)
        axes[1].set_xlabel(feature_name)
        axes[1].set_ylabel('Return5min')
        axes[1].set_title(f'测试集: {feature_name} vs 收益率\n相关系数: {test_corr:.4f}')
        
        plt.tight_layout()
        plt.savefig(f'output/{feature_name}_stability_check.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"散点图已保存: output/{feature_name}_stability_check.png")
        
    except Exception as e:
        print(f"无法生成图表: {e}")
    
    return train_corr, test_corr
# ==============================================================================
def main():
    """
    主函数 - 重构后的简洁版本
    """
    print("="*70)
    print("股票E收益率预测 - 模型训练流程")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    start_time = time.time()
    
    try:
        # 阶段1: 加载数据
        print("\n📂 阶段1: 加载训练数据")
        X, y, feature_names = load_training_data()
        if X is None:
            return
        
        # 阶段2: 数据预处理
        print("\n🔧 阶段2: 数据预处理")
        X_clean = utils.handle_missing_values_advanced(X, 'ffill')
        X_filtered = utils.remove_low_variance_features(X_clean)
        
        # 阶段3: 特征选择
        print("\n🎯 阶段3: 特征选择")
        # 基于您的相关性文档，使用0.01的阈值
        X_selected, selected_features, corr_df = utils.select_features_by_correlation(
            X_filtered, y, correlation_threshold=0.01
        )
        
        # 阶段4: 准备数据集
        print("\n📊 阶段4: 准备数据集")
        X_train, X_test, y_train, y_test = utils.prepare_time_series_split(
            X_selected, y, test_size=0.2
        )

        # 新增：诊断步骤
        print("\n🔍 开始诊断分析")
        print("="*60)

        # 1. 检查训练集和测试集一致性
        check_train_test_consistency(X_train, X_test, y_train, y_test, top_features=5)

        # 2. 测试基线模型
        baseline_results = test_baseline_models(y_train, y_test)

        # 3. 检查重要特征的稳定性
        important_features = ['BidPrice1', 'mid_price', 'total_bid_volume']
        for feature in important_features:
            if feature in X_train.columns:
                check_feature_stability(X_train, X_test, y_train, y_test, feature)

        # 4. 检查目标变量的自相关性
        print("\n目标变量自相关性检查:")
        autocorr_1 = y_train.autocorr(lag=1)
        autocorr_5 = y_train.autocorr(lag=5)
        autocorr_20 = y_train.autocorr(lag=20)
        print(f"滞后1期自相关性: {autocorr_1:.6f}")
        print(f"滞后5期自相关性: {autocorr_5:.6f}")
        print(f"滞后20期自相关性: {autocorr_20:.6f}")

        # 5. 如果基线模型结果很差，可能问题不在我们的模型
        print("\n💡 诊断结论:")
        if baseline_results['naive'] < 0.1 and baseline_results['mean'] < 0.1:
            print("⚠️ 注意：基线模型表现也很差，说明预测任务本身可能很难")
            print("   未来5分钟收益率可能接近随机游走")
        else:
            print("✅ 基线模型有一定预测能力，我们的模型应该能做得更好")

        # 然后继续模型训练...
        print("\n🤖 阶段5: 开始XGBoost模型训练")

        # 阶段5: 模型训练
        print("\n🤖 阶段5: 模型训练")
        model = utils.train_xgboost_model(X_train, X_test, y_train, y_test)
        
        # 阶段6: 模型评估
        print("\n📈 阶段6: 模型评估")
        metrics, y_pred = utils.evaluate_model_performance(
            model, X_train, X_test, y_train, y_test
        )
        
        # 阶段7: 特征重要性分析
        print("\n💡 阶段7: 特征重要性分析")
        importance_df = utils.get_feature_importance(model, selected_features)
        
        # 阶段8: 保存结果
        print("\n💾 阶段8: 保存结果")
        save_results(model, X_selected, y, metrics, importance_df, selected_features, corr_df)
        
        # 完成报告
        total_time = time.time() - start_time
        print_success_report(total_time, metrics, len(selected_features))
        
        return {
            'model': model,
            'metrics': metrics,
            'importance': importance_df,
            'selected_features': selected_features
        }
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_training_data():
    """加载训练数据"""
    training_dir = "./output/training_data"
    
    X_path = os.path.join(training_dir, "X_features.csv")
    y_path = os.path.join(training_dir, "y_target.csv")
    features_path = os.path.join(training_dir, "feature_names.csv")
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print("❌ 训练数据不存在，请先运行特征工程")
        return None, None, None
    
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).squeeze()
    
    if os.path.exists(features_path):
        feature_names = pd.read_csv(features_path)['feature_name'].tolist()
    else:
        feature_names = X.columns.tolist()
    
    print(f"✅ 数据加载完成: X{X.shape}, y{y.shape}")
    return X, y, feature_names

def save_results(model, X, y, metrics, importance_df, selected_features, corr_df):
    """保存所有结果"""
    results_dir = "./output/model_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存模型
    model.save_model(os.path.join(results_dir, "trained_model.json"))
    
    # 保存预测结果
    predictions = pd.DataFrame({
        'actual': y.values,
        'predicted': model.predict(xgb.DMatrix(X))
    })
    predictions.to_csv(os.path.join(results_dir, "predictions.csv"), index=False)
    
    # 保存特征重要性
    importance_df.to_csv(os.path.join(results_dir, "feature_importance.csv"), index=False)
    
    # 保存相关性分析
    corr_df.to_csv(os.path.join(results_dir, "feature_correlations.csv"), index=False)
    
    # 保存选中的特征列表
    pd.DataFrame({'feature': selected_features}).to_csv(
        os.path.join(results_dir, "selected_features.csv"), index=False
    )
    
    # 保存模型指标
    with open(os.path.join(results_dir, "model_metrics.txt"), 'w') as f:
        f.write(f"训练集RMSE: {metrics['train_rmse']:.6f}\n")
        f.write(f"测试集RMSE: {metrics['test_rmse']:.6f}\n")
        f.write(f"测试集相关系数: {metrics['test_corr']:.6f}\n")
        f.write(f"使用特征数: {len(selected_features)}\n")
    
    print(f"✅ 结果已保存到: {results_dir}")

def print_success_report(total_time, metrics, n_features):
    """打印成功报告"""
    print("\n" + "="*70)
    print("🎉 模型训练完成！")
    print("="*70)
    print(f"总耗时: {total_time:.1f}秒")
    print(f"使用特征数: {n_features}")
    print(f"测试集RMSE: {metrics['test_rmse']:.6f}")
    print(f"测试集相关系数: {metrics['test_corr']:.6f}")
    print("="*70)

if __name__ == "__main__":
    # 检查依赖
    try:
        import xgboost as xgb
        result = main()
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        print("请运行: pip install xgboost scikit-learn")