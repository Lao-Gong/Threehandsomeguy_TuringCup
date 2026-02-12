# main.py中使用新的utils模块
import utils

def main():
    print("股票收益率预测程序启动")
    
    # 1. 加载数据
    all_data = utils.load_all_data("./data")
    
    # 2. 数据清洗和特征计算ddd
    processed_data = {}
    for stock_name, raw_data in all_data.items():
        if not raw_data.empty:
            # 使用utils模块中的函数
            cleaned = utils.clean_data(raw_data, stock_name)
            with_features = utils.calculate_basic_features(cleaned, stock_name)
            processed_data[stock_name] = with_features
    
    # 3. 数据验证和报告
    for stock_name, data in processed_data.items():
        utils.validate_data_quality(data, stock_name)
    
    utils.generate_data_report(processed_data)
    
    # 4. 保存数据
    for stock_name, data in processed_data.items():
        if not data.empty:
            filename = f"stock_{stock_name}_processed.csv"
            saved_path = utils.save_processed_data(data, filename)
            print(f"✅ 保存: {saved_path}")
    
    return processed_data

if __name__ == "__main__":
    result = main()