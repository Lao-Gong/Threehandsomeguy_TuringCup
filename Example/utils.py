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

def load_all_data(base_path: str = "./data") -> Dict[str, pd.DataFrame]:
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
    """
    基础数据清洗：处理缺失值、排序等
    """
    if raw_data.empty:
        return raw_data
    
    print(f"清洗{stock_name}股票数据...")
    
    # 1. 按时间排序
    sorted_data = raw_data.sort_values('Time').reset_index(drop=True)
    
    # 2. 处理缺失值
    cleaned_data = handle_missing_values(sorted_data)
    
    # 3. 检查数据类型（根据文档说明，很多列应该是Int32）
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
        data['mid_price'] = (data['BidPrice1'] + data['AskPrice1']) / 2
        data['spread'] = data['AskPrice1'] - data['BidPrice1']
        data['spread_ratio'] = data['spread'] / data['mid_price']
    
    return data

def add_volume_features(data: pd.DataFrame) -> pd.DataFrame:
    """添加成交量相关特征"""
    # 计算总买卖挂单量
    bid_volume_cols = [f'BidVolume{i}' for i in range(1, 6)]
    ask_volume_cols = [f'AskVolume{i}' for i in range(1, 6)]
    
    if all(col in data.columns for col in bid_volume_cols + ask_volume_cols):
        data['total_bid_volume'] = data[bid_volume_cols].sum(axis=1)
        data['total_ask_volume'] = data[ask_volume_cols].sum(axis=1)
        total_volume = data['total_bid_volume'] + data['total_ask_volume']
        
        # 避免除零错误
        data['order_imbalance'] = np.where(
            total_volume > 0, 
            (data['total_bid_volume'] - data['total_ask_volume']) / total_volume, 
            0
        )
    
    return data

def add_orderbook_features(data: pd.DataFrame) -> pd.DataFrame:
    """添加订单簿相关特征"""
    # 这里可以添加更多订单簿特征
    # 为后续高级因子预留位置
    return data

# ============================================================================
# 4. 高级因子构建模块（预留位置）
# ============================================================================

def calculate_advanced_factors(data: pd.DataFrame, stock_name: str) -> pd.DataFrame:
    """
    计算高级因子（这里先预留位置，后续我们会详细讨论）
    """
    # 这个函数我们后面会详细实现
    # 包括：动量因子、波动率因子、资金流因子、板块联动因子等
    print(f"为{stock_name}预留高级因子计算位置...")
    return data

def calculate_cross_stock_factors(stock_e_data: pd.DataFrame, other_stocks_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    计算跨股票因子（板块联动效应）
    """
    # 这个函数我们后面会详细实现
    # 利用A、B、C、D股票的数据来为E股票构建因子
    print("预留跨股票因子计算位置...")
    return stock_e_data

# ============================================================================
# 5. 数据验证模块
# ============================================================================

def validate_data_quality(data: pd.DataFrame, stock_name: str) -> bool:
    """
    验证数据质量
    """
    if data.empty:
        print(f"❌ {stock_name}数据为空")
        return False
    
    # 检查必要列是否存在
    required_columns = ['Time', 'LastPrice']
    for col in required_columns:
        if col not in data.columns:
            print(f"❌ {stock_name}缺少必要列: {col}")
            return False
    
    # 检查时间顺序
    time_sorted = data['Time'].is_monotonic_increasing
    if not time_sorted:
        print(f"⚠️ {stock_name}时间序列未排序")
    
    # 检查缺失值
    missing_count = data[required_columns].isnull().sum().sum()
    if missing_count > 0:
        print(f"⚠️ {stock_name}有{missing_count}个缺失值")
    
    print(f"✅ {stock_name}数据质量检查通过")
    return True

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