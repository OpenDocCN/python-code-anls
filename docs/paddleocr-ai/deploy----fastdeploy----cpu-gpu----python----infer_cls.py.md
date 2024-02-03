# `.\PaddleOCR\deploy\fastdeploy\cpu-gpu\python\infer_cls.py`

```
# 导入所需的库和模块
import fastdeploy as fd
import cv2
import os

# 解析命令行参数
def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cls_model",
        required=True,
        help="Path of Classification model of PPOCR.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'cpu', 'kunlunxin' or 'gpu'.")
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Define which GPU card used to run model.")
    return parser.parse_args()

# 构建运行时选项
def build_option(args):

    cls_option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        cls_option.use_gpu(args.device_id)

    return cls_option

# 解析命令行参数
args = parse_arguments()

# 构建分类模型文件路径
cls_model_file = os.path.join(args.cls_model, "inference.pdmodel")
cls_params_file = os.path.join(args.cls_model, "inference.pdiparams")

# 设置运行时选项
cls_option = build_option(args)

# 创建分类器模型
cls_model = fd.vision.ocr.Classifier(
    cls_model_file, cls_params_file, runtime_option=cls_option)

# 设置后处理参数
cls_model.postprocessor.cls_thresh = 0.9

# 读取图像文件
im = cv2.imread(args.image)

# 预测并返回结果
# 使用分类模型对输入图像进行预测，返回预测结果
result = cls_model.predict(im)

# 用户可以通过以下代码推断一批图像。
# result = cls_model.batch_predict([im])

# 打印预测结果
print(result)
```