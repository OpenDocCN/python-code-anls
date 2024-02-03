# `.\PaddleOCR\ppstructure\kie\tools\trans_xfun_data.py`

```
# 版权声明和许可证信息
# 该代码版权归 PaddlePaddle 作者所有，遵循 Apache License, Version 2.0 许可证
# 可以在遵守许可证的前提下使用该文件
# 许可证详情请查看 http://www.apache.org/licenses/LICENSE-2.0

# 导入 json 模块
import json

# 从给定的 JSON 文件中读取数据，将其转换为字典格式，并提取 documents 字段
def transfer_xfun_data(json_path=None, output_file=None):
    # 以只读方式打开 JSON 文件
    with open(json_path, "r", encoding='utf-8') as fin:
        # 读取文件的所有行
        lines = fin.readlines()

    # 将第一行 JSON 数据转换为字典格式
    json_info = json.loads(lines[0])
    # 获取 documents 字段的值
    documents = json_info["documents"]
    
    # 以写入方式打开输出文件
    with open(output_file, "w", encoding='utf-8') as fout:
        # 遍历 documents 中的每个文档
        for idx, document in enumerate(documents):
            # 初始化标签信息列表
            label_info = []
            # 获取文档中的图片信息
            img_info = document["img"]
            # 获取文档内容
            document = document["document"]
            # 获取图片路径
            image_path = img_info["fname"]

            # 遍历文档中的每个文本框
            for doc in document:
                # 获取文本框的坐标信息
                x1, y1, x2, y2 = doc["box"]
                # 构建文本框的四个顶点坐标
                points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                # 构建标签信息字典
                label_info.append({
                    "transcription": doc["text"],
                    "label": doc["label"],
                    "points": points,
                    "id": doc["id"],
                    "linking": doc["linking"]
                })

            # 将图片路径和标签信息写入输出文件
            fout.write(image_path + "\t" + json.dumps(
                label_info, ensure_ascii=False) + "\n")

    # 打印处理完成的提示信息
    print("===ok====")

# 解析命令行参数
def parser_args():
    # 导入 argparse 模块
    import argparse
    # 创建参数解析器对象
    parser = argparse.ArgumentParser(description="args for paddleserving")
    # 添加命令行参数
    parser.add_argument(
        "--ori_gt_path", type=str, required=True, help='origin xfun gt path')
    parser.add_argument(
        "--output_path", type=str, required=True, help='path to save')
    # 解析命令行参数并返回结果
    args = parser.parse_args()
    # 返回解析后的参数
    return args
# 解析命令行参数，获取用户输入的参数值
args = parser_args()
# 调用函数，传入原始数据路径和输出路径参数
transfer_xfun_data(args.ori_gt_path, args.output_path)
```