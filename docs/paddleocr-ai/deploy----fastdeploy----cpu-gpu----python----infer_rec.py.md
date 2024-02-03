# `.\PaddleOCR\deploy\fastdeploy\cpu-gpu\python\infer_rec.py`

```py
# 导入所需的库
import fastdeploy as fd
import cv2
import os

# 定义解析命令行参数的函数
def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    # 添加命令行参数：rec_model，必须提供PPOCR的识别模型路径
    parser.add_argument(
        "--rec_model",
        required=True,
        help="Path of Recognization model of PPOCR.")
    # 添加命令行参数：rec_label_file，必须提供PPOCR的识别模型标签文件路径
    parser.add_argument(
        "--rec_label_file",
        required=True,
        help="Path of Recognization model of PPOCR.")
    # 添加命令行参数：image，必须提供测试图像文件路径
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")
    # 添加命令行参数：device，推理设备类型，默认为cpu，支持'cpu'、'kunlunxin'或'gpu'
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'cpu', 'kunlunxin' or 'gpu'.")
    # 添加命令行参数：device_id，定义用于运行模型的GPU卡
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Define which GPU card used to run model.")
    # 解析命令行参数并返回
    return parser.parse_args()

# 构建运行时选项
def build_option(args):
    # 创建RuntimeOption对象
    rec_option = fd.RuntimeOption()
    # 如果设备类型为gpu，则使用指定的GPU卡
    if args.device.lower() == "gpu":
        rec_option.use_gpu(args.device_id)
    # 返回运行时选项
    return rec_option

# 解析命令行参数
args = parse_arguments()

# 构建PPOCR识别模型文件路径
rec_model_file = os.path.join(args.rec_model, "inference.pdmodel")
rec_params_file = os.path.join(args.rec_model, "inference.pdiparams")
rec_label_file = args.rec_label_file

# 设置运行时选项
rec_option = build_option(args)

# 创建PPOCR识别模型
rec_model = fd.vision.ocr.Recognizer(
    # 传入参数rec_model_file, rec_params_file, rec_label_file, runtime_option给函数rec_option
# 读取图像文件
im = cv2.imread(args.image)

# 预测并返回结果
result = rec_model.predict(im)

# 用户可以通过以下代码推断一批图像。
# result = rec_model.batch_predict([im])

# 打印结果
print(result)
```