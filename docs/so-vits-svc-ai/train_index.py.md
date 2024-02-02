# `so-vits-svc\train_index.py`

```py
# 导入必要的模块
import argparse  # 用于解析命令行参数
import os  # 用于处理文件路径
import pickle  # 用于序列化和反序列化 Python 对象

import utils  # 导入自定义的工具模块

# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数，指定根目录路径，默认值为"dataset/44k"，并提供帮助信息
    parser.add_argument("--root_dir", type=str, default="dataset/44k", help="path to root dir")
    # 添加命令行参数，指定配置文件路径，默认值为"./configs/config.json"，并提供帮助信息
    parser.add_argument('-c', '--config', type=str, default="./configs/config.json", help='JSON file for configuration')
    # 添加命令行参数，指定输出目录路径，默认值为"logs/44k"，并提供帮助信息
    parser.add_argument("--output_dir", type=str, default="logs/44k", help="path to output dir")

    # 解析命令行参数
    args = parser.parse_args()

    # 从配置文件中获取超参数
    hps = utils.get_hparams_from_file(args.config)
    # 获取说话人字典
    spk_dic = hps.spk
    # 创建空字典用于存储结果
    result = {}
    
    # 遍历说话人字典
    for k,v in spk_dic.items():
        # 打印当前正在处理的说话人索引
        print(f"now, index {k} feature...")
        # 训练索引
        index = utils.train_index(k,args.root_dir)
        # 将索引存入结果字典
        result[v] = index

    # 将结果字典序列化并存入文件
    with open(os.path.join(args.output_dir,"feature_and_index.pkl"),"wb") as f:
        pickle.dump(result,f)
```