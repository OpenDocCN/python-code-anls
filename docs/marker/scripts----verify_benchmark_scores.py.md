# `.\marker\scripts\verify_benchmark_scores.py`

```py
# 导入 json 模块和 argparse 模块
import json
import argparse

# 验证分数的函数，接收一个文件路径作为参数
def verify_scores(file_path):
    # 打开文件并加载 JSON 数据
    with open(file_path, 'r') as file:
        data = json.load(file)

    # 获取 multicolcnn.pdf 文件的分数
    multicolcnn_score = data["marker"]["files"]["multicolcnn.pdf"]["score"]
    # 获取 switch_trans.pdf 文件的分数
    switch_trans_score = data["marker"]["files"]["switch_trans.pdf"]["score"]

    # 如果其中一个分数小于等于 0.4，则抛出 ValueError 异常
    if multicolcnn_score <= 0.4 or switch_trans_score <= 0.4:
        raise ValueError("One or more scores are below the required threshold of 0.4")

# 如果当前脚本被直接执行
if __name__ == "__main__":
    # 创建 ArgumentParser 对象，设置描述信息
    parser = argparse.ArgumentParser(description="Verify benchmark scores")
    # 添加一个参数，指定文件路径，类型为字符串
    parser.add_argument("file_path", type=str, help="Path to the json file")
    # 解析命令行参数
    args = parser.parse_args()
    # 调用 verify_scores 函数，传入文件路径参数
    verify_scores(args.file_path)
```