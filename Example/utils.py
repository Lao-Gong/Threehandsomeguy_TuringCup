# utils.py
"""
股票收益率预测工具函数库
作者：你的名字

功能模块：
1. 数据加载模块
2. 数据清洗模块  
3. 基础特征计算模块
4. 高级因子构建模块（为后续准备）
5. 数据验证模块
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# ============================================================================
# 1. 数据加载模块
# ============================================================================

def load_single_stock_data(day: int, stock_name: str, base_path: str = "./data") -> pd.DataFrame:
    """
    加载单只股票单天的数据
    
    参数说明：
    day: 第几天（1-5）
    stock_name: 股票名称（A-E）
    base_path: 数据根目录
    
    返回：单个DataFrame
    """
    file_path = os.path.join(base_path, str(day), f"{stock_name}.csv")
    
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在: {file_path}")
        return pd.DataFrame()
    
    try:
        data = pd.read_csv(file_path)
        data['day'] = day
        data['stock'] = stock_name
        print(f"✅ 成功加载: 第{day}天 {stock_name}股票")
        return data
    except Exception as e:
        print(f"❌ 加载失败 {file_path}: {e}")
        return pd.DataFrame()

def load_all_data(base_path: str = "./data") -> dict[str, pd.DataFrame]:
    """
    加载所有5天5只股票的数据（主加载函数）
    """
    print("=" * 50)
    print("开始加载所有数据...")
    
    all_stocks_data = {}
    
    # 遍历5天5只股票
    for day in range(1, 6):
        for stock_name in ['A', 'B', 'C', 'D', 'E']:
            # 加载单天单只股票数据
            daily_data = load_single_stock_data(day, stock_name, base_path)
            
            if not daily_data.empty:
                # 添加到对应的股票篮子
                if stock_name not in all_stocks_data:
                    all_stocks_data[stock_name] = []
                all_stocks_data[stock_name].append(daily_data)
    
    # 合并每个股票的数据
    return merge_stock_data(all_stocks_data)

def merge_stock_data(stocks_data: Dict[str, List[pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    合并每个股票的多个DataFrame
    """
    merged_data = {}
    
    for stock_name, data_list in stocks_data.items():
        if data_list:
            merged_data[stock_name] = pd.concat(data_list, ignore_index=True)
            print(f"✅ {stock_name}股票: 合并完成，共{len(merged_data[stock_name])}行")
        else:
            merged_data[stock_name] = pd.DataFrame()
            print(f"❌ {stock_name}股票: 无数据")
    
    return merged_data
"""
原始文件结构：
data/
├── 1/
│   ├── A.csv (列：Time, BidPrice1, BidPrice2, ..., Return5min)
│   ├── B.csv (同样列结构)
│   └── ...
├── 2/
│   └── ...

处理后结构：
{
    'A': ██████████  (5天合并的大表格，包含所有原始列 + day列 + stock列)
    'B': ██████████  (同样结构)
    'C': ██████████
    'D': ██████████
    'E': ██████████
}

股票A的DataFrame：
┌───────┬────────────┬────────────┬─────┬────────┬────────┐
│ Time  │ BidPrice1  │ BidPrice2  │ ... │  day   │ stock  │
├───────┼────────────┼────────────┼─────┼────────┼────────┤
│100000 │    100     │    101     │ ... │   1    │   A    │  ← 第1天数据
│100003 │    101     │    102     │ ... │   1    │   A    │
│ ...   │    ...     │    ...     │ ... │   ...  │   ...  │
│200000 │    105     │    106     │ ... │   2    │   A    │  ← 第2天数据
│200003 │    106     │    107     │ ... │   2    │   A    │
└───────┴────────────┴────────────┴─────┴────────┴────────┘
"""

# ============================================================================
# 2. 数据清洗模块
# ============================================================================

def clean_data(raw_data: pd.DataFrame, stock_name: str) -> pd.DataFrame:
    """清洗单只股票的数据"""
    if raw_data.empty:
        return raw_data
    
    print(f"清洗{stock_name}股票数据...")
    
    # 修正：先按天排序，再按时间排序
    # 这样可以确保时间序列的正确性
    sorted_data = raw_data.sort_values(['day', 'Time']).reset_index(drop=True)
    
    # 验证排序是否正确
    print(f"  ✅ 已按['day', 'Time']排序")
    print(f"  天数顺序: {sorted_data['day'].unique()[:5]}...")  # 显示前5个不同的天数
    
    # 2. 处理缺失值
    cleaned_data = handle_missing_values(sorted_data)
    
    # 3. 优化数据类型
    cleaned_data = optimize_data_types(cleaned_data)
    
    print(f"✅ {stock_name}清洗完成: {len(cleaned_data)}行")
    return cleaned_data

def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    专门处理缺失值的函数
    """
    missing_count = data.isnull().sum().sum()
    if missing_count > 0:
        print(f"发现{missing_count}个缺失值，正在处理...")
        # 先用前向填充，再用后向填充
        data_filled = data.fillna(method='ffill').fillna(method='bfill')
        # 删除还有缺失值的行
        return data_filled.dropna()
    return data

def optimize_data_types(data: pd.DataFrame) -> pd.DataFrame:
    """
    优化数据类型，根据文档说明设置正确的数据类型
    """
    # 文档中说明为Int32的列
    int32_columns = [
        'Time', 'BidPrice1', 'BidPrice2', 'BidPrice3', 'BidPrice4', 'BidPrice5',
        'BidVolume1', 'BidVolume2', 'BidVolume3', 'BidVolume4', 'BidVolume5',
        'AskPrice1', 'AskPrice2', 'AskPrice3', 'AskPrice4', 'AskPrice5', 
        'AskVolume1', 'AskVolume2', 'AskVolume3', 'AskVolume4', 'AskVolume5',
        'OrderBuyNum', 'OrderSellNum', 'OrderBuyVolume', 'OrderSellVolume',
        'TradeBuyNum', 'TradeSellVolume', 'LastPrice'
    ]
    
    # 文档中说明为Int64的列
    int64_columns = ['TradeBuyAmount', 'TradeSellAmount']
    
    # 文档中说明为Float32的列
    float32_columns = ['Return5min']
    
    # 转换数据类型（如果列存在的话）
    for col in int32_columns:
        if col in data.columns:
            data[col] = data[col].astype('int32')
    
    for col in int64_columns:
        if col in data.columns:
            data[col] = data[col].astype('int64')
    
    for col in float32_columns:
        if col in data.columns:
            data[col] = data[col].astype('float32')
    
    return data

# ============================================================================
# 3. 基础特征计算模块
# ============================================================================

def calculate_basic_features(data: pd.DataFrame, stock_name: str) -> pd.DataFrame:
    """
    计算基础特征（为后续高级因子打基础）
    """
    if data.empty:
        return data
    
    print(f"为{stock_name}计算基础特征...")
    
    # 复制数据，避免修改原始数据
    data_with_features = data.copy()
    
    # 价格相关特征
    data_with_features = add_price_features(data_with_features)
    
    # 成交量相关特征  
    data_with_features = add_volume_features(data_with_features)
    
    # 订单簿相关特征
    data_with_features = add_orderbook_features(data_with_features)
    
    print(f"✅ {stock_name}基础特征计算完成")
    return data_with_features

def add_price_features(data: pd.DataFrame) -> pd.DataFrame:
    """添加价格相关特征"""
    if 'BidPrice1' in data.columns and 'AskPrice1' in data.columns:
        data['mid_price'] = (data['BidPrice1'] + data['AskPrice1']) / 2 # 买卖双方中间价
        data['spread'] = data['AskPrice1'] - data['BidPrice1'] # 买卖差价，差价越小越容易流通
        data['spread_ratio'] = data['spread'] / data['mid_price'] # 价差比率，即相对价差
    
    return data

# 在 utils.py 中修改 add_volume_features 函数，添加调试信息
def add_volume_features(data: pd.DataFrame) -> pd.DataFrame:
    """添加成交量相关特征"""
    print("  计算成交量特征...")
    
    # 计算总买卖挂单量
    bid_volume_cols = [f'BidVolume{i}' for i in range(1, 6)]
    ask_volume_cols = [f'AskVolume{i}' for i in range(1, 6)]
    
    # 检查所有需要的列是否存在
    existing_bid_cols = [col for col in bid_volume_cols if col in data.columns]
    existing_ask_cols = [col for col in ask_volume_cols if col in data.columns]
    
    print(f"    存在的买量列: {existing_bid_cols}")
    print(f"    存在的卖量列: {existing_ask_cols}")
    
    if existing_bid_cols and existing_ask_cols:
        data['total_bid_volume'] = data[existing_bid_cols].sum(axis=1)
        data['total_ask_volume'] = data[existing_ask_cols].sum(axis=1)
        total_volume = data['total_bid_volume'] + data['total_ask_volume']
        
        # 避免除零错误
        data['order_imbalance'] = np.where(
            total_volume > 0, 
            (data['total_bid_volume'] - data['total_ask_volume']) / total_volume, 
            0
        )
        print("    ✅ 成功创建 order_imbalance 列")
    else:
        print("    ⚠️ 无法创建 order_imbalance 列，缺少必要的买卖量列")
    
    return data

def add_orderbook_features(data: pd.DataFrame) -> pd.DataFrame:
    """添加订单簿相关特征"""
    # 这里可以添加更多订单簿特征
    # 为后续高级因子预留位置
    return data

# ============================================================================
# 4. 高级因子构建模块（预留位置）
# ============================================================================

def calculate_money_flow_factors(data, stock_name):
    """
    计算资金流相关因子
    """
    print(f"为{stock_name}计算资金流因子...")
    
    df = data.copy()
    
    # 1. 主动买卖资金流
    if 'TradeBuyAmount' in df.columns and 'TradeSellAmount' in df.columns:
        # 净主动买入金额
        df['net_buy_amount'] = df['TradeBuyAmount'] - df['TradeSellAmount']
        
        # 资金流比率 📊 这个值在-1到1之间，衡量资金流向的强度
        total_amount = df['TradeBuyAmount'] + df['TradeSellAmount']
        df['money_flow_ratio'] = df['net_buy_amount'] / (total_amount + 1e-8) # 所有的加一个很小的数都是为了避免除0错误
        
        # 资金流动量 📊 看资金流的变化趋势
        df['money_flow_momentum'] = df['net_buy_amount'].pct_change(periods=20)
    
    # 2. 大单资金流（假设大单>平均值的2倍）
    if 'TradeBuyAmount' in df.columns:
        avg_buy = df['TradeBuyAmount'].rolling(window=100, min_periods=1).mean() #.rolling用于计算移动平均值
        df['large_buy_flow'] = (df['TradeBuyAmount'] > 2 * avg_buy).astype(int) # 超过平均值2倍则为大单
    
    # 3. 委托不平衡动态
    if 'OrderBuyVolume' in df.columns and 'OrderSellVolume' in df.columns:
        # 委托不平衡变化率
        df['order_imbalance_change'] = df['order_imbalance'].pct_change(periods=20)
    
    print(f"✅ {stock_name}资金流因子计算完成")
    return df

def calculate_sector_factors(stock_e_data, other_stocks_data):
    """
    计算板块联动因子 - 利用A、B、C、D股票预测E股票
    """
    print("计算板块联动因子...")
    
    df_e = stock_e_data.copy()
    
    # 1. 板块价格动量因子
    sector_momentum_1min = []
    sector_momentum_5min = []
    
    for stock_name, stock_data in other_stocks_data.items():
        if not stock_data.empty and 'LastPrice' in stock_data.columns:
            # 计算其他股票的动量
            data_sorted = stock_data.sort_values(['day', 'Time']).reset_index(drop=True)
            momentum_1min = data_sorted['LastPrice'].pct_change(20)
            momentum_5min = data_sorted['LastPrice'].pct_change(100)
            
            # 简单对齐（假设时间戳完全匹配）
            sector_momentum_1min.append(momentum_1min)
            sector_momentum_5min.append(momentum_5min)
    
    # 计算板块平均动量
    if sector_momentum_1min:
        # 转换为DataFrame便于计算
        sector_df_1min = pd.DataFrame(sector_momentum_1min).T
        sector_df_5min = pd.DataFrame(sector_momentum_5min).T
        # 对每行求平均值 📊 意义：整个板块最近1分钟和5分钟的平均涨跌幅
        df_e['sector_momentum_1min'] = sector_df_1min.mean(axis=1)
        df_e['sector_momentum_5min'] = sector_df_5min.mean(axis=1)
        
        # 相对强度因子
        if 'momentum_1min' in df_e.columns:
            df_e['relative_strength_1min'] = df_e['momentum_1min'] - df_e['sector_momentum_1min']
            df_e['relative_strength_5min'] = df_e['momentum_5min'] - df_e['sector_momentum_5min']
        # 📊 意义：E股票是比板块强还是弱？
        #       正数：E表现优于板块 → 可能继续领先
        #       负数：E表现差于板块 → 可能补涨或继续落后

    # 2. 板块资金流因子
    sector_money_flow = []
    
    for stock_name, stock_data in other_stocks_data.items():
        if not stock_data.empty and 'TradeBuyAmount' in stock_data.columns:
            data_sorted = stock_data.sort_values(['day', 'Time']).reset_index(drop=True)
            net_flow = (data_sorted['TradeBuyAmount'] - data_sorted['TradeSellAmount']) / \
                      (data_sorted['TradeBuyAmount'] + data_sorted['TradeSellAmount'] + 1e-8)
            sector_money_flow.append(net_flow)
    
    if sector_money_flow:
        sector_flow_df = pd.DataFrame(sector_money_flow).T
        df_e['sector_money_flow'] = sector_flow_df.mean(axis=1)
    # 📊 意义：E股票是比板块强还是弱？
    #       正数：E表现优于板块 → 可能继续领先
    #       负数：E表现差于板块 → 可能补涨或继续落后

    # 3. 板块热度因子（基于成交量）
    sector_volume_ratio = []
    
    for stock_name, stock_data in other_stocks_data.items():
        if not stock_data.empty and 'TradeBuyVolume' in stock_data.columns:
            data_sorted = stock_data.sort_values(['day', 'Time']).reset_index(drop=True)
            volume_ratio = data_sorted['TradeBuyVolume'] / \
                          data_sorted['TradeBuyVolume'].rolling(window=100, min_periods=1).mean()
            sector_volume_ratio.append(volume_ratio)
    
    if sector_volume_ratio:
        sector_volume_df = pd.DataFrame(sector_volume_ratio).T
        df_e['sector_volume_ratio'] = sector_volume_df.mean(axis=1)
    # 📊 意义：整个板块的交易热度
    #       大于1：板块关注度高，可能有机会
    #       小于1：板块关注度低，可能缺乏动力
    
    print("✅ 板块联动因子计算完成")
    return df_e

def calculate_technical_indicators(data, stock_name):
    """
    计算技术指标因子
    """
    print(f"为{stock_name}计算技术指标...")
    
    # 确保数据已排序
    df = data.sort_values(['day', 'Time']).reset_index(drop=True).copy()
    
    # 1. 价格动量因子
    if 'LastPrice' in df.columns:
        # ====================================================================
        # 短期动量（过去20个Tick，约1分钟）
        # 计算当前价格相比20个Tick前（1分钟前）的涨跌幅
        # 意义：如果这个值是正数，说明最近1分钟在上涨；负数则在下跌
        df['momentum_1min'] = df['LastPrice'].pct_change(periods=20)

        #=====================================================================
        # 中期动量（过去100个Tick，约5分钟）
        # 意义：看较长时间段的趋势，比1分钟动量更稳定  
        df['momentum_5min'] = df['LastPrice'].pct_change(periods=100)

        #=====================================================================
        # 价格位置：当前价在近期范围内的相对位置
        # 计算过去5分钟内的最高价
        df['high_5min'] = df['LastPrice'].rolling(window=100, min_periods=1).max()
        # 计算过去5分钟的最低价
        df['low_5min'] = df['LastPrice'].rolling(window=100, min_periods=1).min()
        # 计算当前价格在5分钟价格区间中的位置
        # 意义：这个值在0-1之间，0表示在最低点，1表示在最高点
        # 比如0.8表示当前价格接近5分钟内的最高点
        df['price_position'] = (df['LastPrice'] - df['low_5min']) / (df['high_5min'] - df['low_5min'] + 1e-8)
    
    # 2. 波动率因子
    if 'LastPrice' in df.columns:
        # 意义：波动率越大，说明价格变化越剧烈
        # 收益率波动率（过去100个Tick的标准差）
        df['returns'] = df['LastPrice'].pct_change() # 收益率
        df['volatility_5min'] = df['returns'].rolling(window=100, min_periods=1).std() # 100个点的标准差
    
    # 3. 成交量特征增强
    if 'TradeBuyVolume' in df.columns and 'TradeSellVolume' in df.columns:
        # 总成交量
        df['total_trade_volume'] = df['TradeBuyVolume'] + df['TradeSellVolume']
        
        # 成交量变化率
        df['volume_change_ratio'] = df['total_trade_volume'].pct_change(periods=20) # percent_change百分比变化率
        
        # 量比（当前成交量/过去平均成交量）
        df['volume_ratio'] = df['total_trade_volume'] / df['total_trade_volume'].rolling(window=100, min_periods=1).mean()
    
    print(f"✅ {stock_name}技术指标计算完成")
    return df

def calculate_all_factors(processed_data):
    """
    计算所有因子：基础特征 + 技术指标 + 资金流 + 板块联动
    """
    print("="*60)
    print("开始计算所有因子")
    print("="*60)
    
    # 获取股票E的数据
    stock_e_data = processed_data.get('E', pd.DataFrame())
    if stock_e_data.empty:
        print("❌ 股票E数据为空")
        return pd.DataFrame()
    
    # 获取其他股票数据（A、B、C、D）
    other_stocks = {k: v for k, v in processed_data.items() if k != 'E'}
    
    # 逐步计算各类因子
    print("步骤1: 计算技术指标因子")
    df_with_tech = calculate_technical_indicators(stock_e_data, 'E')
    
    print("步骤2: 计算资金流因子")  
    df_with_money_flow = calculate_money_flow_factors(df_with_tech, 'E')
    
    print("步骤3: 计算板块联动因子")
    df_with_all_factors = calculate_sector_factors(df_with_money_flow, other_stocks)
    
    # 处理缺失值
    df_clean = df_with_all_factors.fillna(method='ffill').fillna(0)
    
    print(f"✅ 所有因子计算完成，总共{len(df_clean.columns)}列")
    return df_clean

# ============================================================================
# 5. 数据验证模块
# ============================================================================

def validate_data_quality(data: pd.DataFrame, stock_name: str) -> bool:
    """验证数据质量"""
    if data.empty:
        print(f"❌ {stock_name}数据为空")
        return False
    
    # 检查必要列是否存在
    required_columns = ['Time', 'LastPrice', 'day']
    for col in required_columns:
        if col not in data.columns:
            print(f"❌ {stock_name}缺少必要列: {col}")
            return False
    
    # 修改：检查['day', 'Time']组合的排序
    # 创建临时列来检查排序
    data_sorted = data.sort_values(['day', 'Time']).reset_index(drop=True)
    data_reset = data.reset_index(drop=True)
    
    # 检查排序后的数据是否与原数据相同
    is_correctly_sorted = data_sorted.equals(data_reset)
    
    if not is_correctly_sorted:
        print(f"⚠️ {stock_name}数据未按['day', 'Time']正确排序")
        # 显示具体问题
        diff_indices = data_sorted.index[~data_sorted.index.isin(data_reset.index) | 
                                        (data_sorted['day'] != data_reset['day']) | 
                                        (data_sorted['Time'] != data_reset['Time'])].tolist()
        if diff_indices:
            print(f"  发现{len(diff_indices)}处排序不一致")
    else:
        print(f"✅ {stock_name}数据排序正确")
    
    # 检查天数顺序
    unique_days = data['day'].unique()
    if len(unique_days) > 1:
        days_sorted = sorted(unique_days)
        if list(unique_days) != days_sorted:
            print(f"⚠️ {stock_name}天数顺序不正确: {list(unique_days)}")
        else:
            print(f"✅ {stock_name}天数顺序正确: {list(unique_days)}")
    
    # 检查目标变量
    if 'Return5min' not in data.columns:
        print(f"⚠️ {stock_name}缺少目标变量Return5min")
    else:
        valid_returns = data['Return5min'].notna().sum()
        print(f"✅ {stock_name}有效收益率数据: {valid_returns}行")
    
    return is_correctly_sorted

def generate_data_report(all_data: Dict[str, pd.DataFrame]) -> None:
    """
    生成数据报告
    """
    print("\n" + "="*60)
    print("数据质量报告")
    print("="*60)
    
    for stock_name, data in all_data.items():
        if not data.empty:
            print(f"\n{stock_name}股票:")
            print(f"  数据行数: {len(data)}")
            print(f"  数据列数: {len(data.columns)}")
            print(f"  时间范围: {data['Time'].min()} 到 {data['Time'].max()}")
            
            if 'Return5min' in data.columns:
                valid_returns = data['Return5min'].notna().sum()
                print(f"  有效收益率: {valid_returns}行")
        
        print("-" * 40)

# ============================================================================
# 工具函数：方便后续使用
# ============================================================================

def get_stock_data(all_data: Dict[str, pd.DataFrame], stock_name: str) -> pd.DataFrame:
    """安全获取股票数据"""
    return all_data.get(stock_name, pd.DataFrame())

def save_processed_data(data: pd.DataFrame, filename: str, output_dir: str = "./output") -> str:
    """保存处理后的数据"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    data.to_csv(filepath, index=False)
    return filepath

def select_features_by_correlation(X, y, correlation_threshold=0.01):
    """
    基于相关性进行特征选择
    """
    print("开始基于相关性进行特征选择...")
    
    # 1. 处理数据质量
    X_clean = X.replace([np.inf, -np.inf], np.nan)
    
    # 2. 移除无效特征（全为NaN或常数）
    valid_features = []
    for col in X_clean.columns:
        if X_clean[col].isnull().all() or X_clean[col].nunique() <= 1:
            print(f"❌ 移除无效特征: {col}")
        else:
            valid_features.append(col)
    
    X_valid = X_clean[valid_features]
    
    # 3. 计算相关性
    correlations = {}
    for feature in X_valid.columns:
        try:
            # 对齐X和y的索引
            valid_idx = X_valid[feature].notna() & y.notna()
            if valid_idx.sum() > 0:  # 确保有有效数据
                corr = np.corrcoef(X_valid.loc[valid_idx, feature], y[valid_idx])[0, 1]
                correlations[feature] = 0.0 if np.isnan(corr) or np.isinf(corr) else corr
            else:
                correlations[feature] = 0.0
        except:
            correlations[feature] = 0.0
    
    # 4. 创建相关性DataFrame
    corr_df = pd.DataFrame({
        'feature': correlations.keys(),
        'correlation': correlations.values(),
        'abs_correlation': [abs(c) for c in correlations.values()]
    }).sort_values('abs_correlation', ascending=False)
    
    # 5. 基于阈值筛选特征
    significant_features = corr_df[corr_df['abs_correlation'] > correlation_threshold]
    
    print(f"特征选择结果:")
    print(f"  原始特征数: {len(X.columns)}")
    print(f"  有效特征数: {len(X_valid.columns)}")
    print(f"  显著特征数 (|corr| > {correlation_threshold}): {len(significant_features)}")
    
    # 显示最重要的特征
    print("\n相关性最强的特征:")
    for i, (_, row) in enumerate(significant_features.head(10).iterrows()):
        print(f"  {i+1}. {row['feature']}: {row['correlation']:.4f}")
    
    selected_features = significant_features['feature'].tolist()
    
    if not selected_features:
        print("⚠️ 警告: 没有特征通过筛选，使用所有有效特征")
        selected_features = valid_features
    
    return X_valid[selected_features], corr_df
# ===========================================================================
# 开始尝试训练
# ============================================================================
# utils.py - 新增特征选择相关函数

def select_features_by_correlation(X, y, correlation_threshold=0.01):
    """
    基于相关性阈值选择特征
    参数:
        X: 特征DataFrame
        y: 目标变量Series  
        correlation_threshold: 相关性绝对值阈值，默认0.01
    返回:
        X_selected: 筛选后的特征DataFrame
        selected_features: 选中的特征列表
        correlation_report: 相关性分析报告
    """
    print("开始基于相关性进行特征选择...")
    
    # 1. 计算每个特征与目标变量的相关性
    correlations = {}
    for col in X.columns:
        try:
            # 对齐有效数据
            valid_idx = X[col].notna() & y.notna()
            if valid_idx.sum() > 100:  # 至少100个有效样本
                corr = np.corrcoef(X.loc[valid_idx, col], y[valid_idx])[0, 1]
                correlations[col] = corr if not (np.isnan(corr) or np.isinf(corr)) else 0.0
            else:
                correlations[col] = 0.0
        except:
            correlations[col] = 0.0
    
    # 2. 创建相关性DataFrame并排序
    corr_df = pd.DataFrame({
        'feature': correlations.keys(),
        'correlation': correlations.values(),
        'abs_correlation': [abs(c) for c in correlations.values()]
    }).sort_values('abs_correlation', ascending=False)
    
    # 3. 基于阈值筛选特征
    significant_features = corr_df[corr_df['abs_correlation'] > correlation_threshold]
    insignificant_features = corr_df[corr_df['abs_correlation'] <= correlation_threshold]
    
    # 4. 记录筛选结果
    selected_features = significant_features['feature'].tolist()
    
    print(f"特征选择完成:")
    print(f"  总特征数: {len(corr_df)}")
    print(f"  选中特征数 (|corr| > {correlation_threshold}): {len(selected_features)}")
    print(f"  淘汰特征数: {len(insignificant_features)}")
    
    # 显示top特征
    print("相关性最强的10个特征:")
    for i, row in significant_features.head(10).iterrows():
        print(f"  {i+1:2d}. {row['feature']:25} | 相关性: {row['correlation']:7.4f}")
    
    # 返回筛选后的特征和报告
    X_selected = X[selected_features] if selected_features else X
    
    return X_selected, selected_features, corr_df

def remove_low_variance_features(X, variance_threshold=0.0001):
    """
    移除低方差特征
    参数:
        X: 特征DataFrame
        variance_threshold: 方差阈值
    返回:
        X_filtered: 过滤后的特征DataFrame
    """
    print("移除低方差特征...")
    
    # 计算每个特征的方差
    variances = X.var()
    low_variance_features = variances[variances < variance_threshold].index.tolist()
    
    if low_variance_features:
        print(f"移除 {len(low_variance_features)} 个低方差特征:")
        for feat in low_variance_features:
            print(f"  - {feat} (方差: {variances[feat]:.6f})")
        
        X_filtered = X.drop(columns=low_variance_features)
        print(f"移除后特征数: {len(X_filtered.columns)}")
    else:
        print("未发现低方差特征")
        X_filtered = X
    
    return X_filtered

def handle_missing_values_advanced(X, strategy='ffill'):
    """
    高级缺失值处理
    参数:
        X: 特征DataFrame
        strategy: 处理策略 ('ffill', 'bfill', 'zero', 'mean')
    返回:
        X_clean: 处理后的特征DataFrame
    """
    print("处理缺失值...")
    
    missing_before = X.isnull().sum().sum()
    print(f"处理前缺失值数量: {missing_before}")
    
    if missing_before == 0:
        print("数据完整，无需处理缺失值")
        return X
    
    X_clean = X.copy()
    
    try:
        if strategy == 'ffill':
            X_clean = X_clean.ffill()
        elif strategy == 'bfill':
            X_clean = X_clean.bfill()
        elif strategy == 'zero':
            X_clean = X_clean.fillna(0)
        elif strategy == 'mean':
            X_clean = X_clean.fillna(X_clean.mean())
        
        # 二次处理：用0填充剩余的缺失值
        X_clean = X_clean.fillna(0)
        
    except Exception as e:
        print(f"缺失值处理失败: {e}, 使用0填充")
        X_clean = X_clean.fillna(0)
    
    missing_after = X_clean.isnull().sum().sum()
    print(f"处理后缺失值数量: {missing_after}")
    
    return X_clean

def prepare_time_series_split(X, y, test_size=0.2):
    """
    为时间序列数据准备训练测试集（不随机打乱）
    参数:
        X: 特征DataFrame
        y: 目标变量Series
        test_size: 测试集比例
    返回:
        X_train, X_test, y_train, y_test: 划分后的数据
    """
    print("准备时间序列训练测试集...")
    
    # 按时间顺序划分（不能随机打乱！）
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"训练集: {len(X_train)} 样本 ({1-test_size:.1%})")
    print(f"测试集: {len(X_test)} 样本 ({test_size:.1%})")
    
    return X_train, X_test, y_train, y_test

# utils.py - 新增模型训练相关函数

def train_xgboost_model(X_train, X_test, y_train, y_test, params=None):
    """
    训练XGBoost模型
    参数:
        X_train, X_test, y_train, y_test: 训练测试数据
        params: 模型参数字典
    返回:
        model: 训练好的模型
        eval_results: 评估结果
    """
    
    print("开始训练XGBoost模型...")
    
    # 设置默认参数
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }
    
    # 转换为DMatrix格式
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # 训练模型
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=50
    )
    
    print(f"✅ 模型训练完成")
    print(f"最佳迭代次数: {model.best_iteration}")
    print(f"最佳验证分数: {model.best_score:.6f}")
    
    return model

def evaluate_model_performance(model, X_train, X_test, y_train, y_test):
    """
    评估模型性能
    返回多种评估指标
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np
    
    print("评估模型性能...")
    
    # 预测
    y_train_pred = model.predict(xgb.DMatrix(X_train))
    y_test_pred = model.predict(xgb.DMatrix(X_test))
    
    # 计算评估指标
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_corr': np.corrcoef(y_train, y_train_pred)[0, 1],
        'test_corr': np.corrcoef(y_test, y_test_pred)[0, 1]
    }
    
    # 打印结果
    print("模型性能指标:")
    print("             训练集       测试集")
    print(f"RMSE      {metrics['train_rmse']:10.6f}  {metrics['test_rmse']:10.6f}")
    print(f"MAE       {metrics['train_mae']:10.6f}  {metrics['test_mae']:10.6f}")
    print(f"相关系数   {metrics['train_corr']:10.6f}  {metrics['test_corr']:10.6f}")
    
    return metrics, y_test_pred

def get_feature_importance(model, feature_names):
    """
    获取特征重要性
    """
    importance_dict = model.get_score(importance_type='weight')
    
    importance_df = pd.DataFrame({
        'feature': list(importance_dict.keys()),
        'importance': list(importance_dict.values())
    }).sort_values('importance', ascending=False)
    
    # 映射特征名称
    importance_df['feature_name'] = importance_df['feature'].apply(
        lambda x: feature_names[int(x.replace('f', ''))] if x.replace('f', '').isdigit() and int(x.replace('f', '')) < len(feature_names) else x
    )
    
    print("特征重要性排名 (前20):")
    for i, row in importance_df.head(20).iterrows():
        print(f"{i+1:2d}. {row['feature_name']:25} | 重要性: {row['importance']:6.0f}")
    
    return importance_df