#模块化与目录结构
    主程序命名为main.py，工具函数拆分到utils.py等独立文件。
    数据文件存放于data/目录，输出结果保存到output/目录。
#注释与命名规范
    代码需符合语言标准，变量名采用下划线命名法（如train_data）。
    关键步骤添加注释说明，例如：
    数据预处理：去除缺失值（作者：李四）
    des_data = src_data.dropna()
#依赖环境说明
    需附带README.md文件，注明运行环境（如Python 3.8）和依赖库（如numpy>=1.20）。
#运行
    python main.py

