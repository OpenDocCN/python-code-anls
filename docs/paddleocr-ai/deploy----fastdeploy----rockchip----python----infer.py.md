# `.\PaddleOCR\deploy\fastdeploy\rockchip\python\infer.py`

```
# 导入所需的库和模块
import fastdeploy as fd
import cv2
import os

# 定义解析命令行参数的函数
def parse_arguments():
    import argparse
    import ast
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument(
        "--det_model", required=True, help="Path of Detection model of PPOCR.")
    parser.add_argument(
        "--cls_model",
        required=True,
        help="Path of Classification model of PPOCR.")
    parser.add_argument(
        "--rec_model",
        required=True,
        help="Path of Recognization model of PPOCR.")
    parser.add_argument(
        "--rec_label_file",
        required=True,
        help="Path of Recognization model of PPOCR.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'cpu', 'kunlunxin' or 'gpu'.")
    parser.add_argument(
        "--cpu_thread_num",
        type=int,
        default=9,
        help="Number of threads while inference on CPU.")
    # 解析并返回命令行参数
    return parser.parse_args()

# 定义构建运行选项的函数
def build_option(args):
    # 创建检测、分类和识别模型的运行选项对象
    det_option = fd.RuntimeOption()
    cls_option = fd.RuntimeOption()
    rec_option = fd.RuntimeOption()
    # 如果设备为 "npu"，则使用 rknpu2
    if args.device == "npu":
        det_option.use_rknpu2()
        cls_option.use_rknpu2()
        rec_option.use_rknpu2()
    # 返回检测、分类和识别模型的运行选项对象
    return det_option, cls_option, rec_option
# 构建模型格式，初始化为ONNX格式
def build_format(args):
    det_format = fd.ModelFormat.ONNX
    cls_format = fd.ModelFormat.ONNX
    rec_format = fd.ModelFormat.ONNX
    # 如果设备为"NPU"，则将模型格式设置为RKNN格式
    if args.device == "npu":
        det_format = fd.ModelFormat.RKNN
        cls_format = fd.ModelFormat.RKNN
        rec_format = fd.ModelFormat.RKNN
    # 返回检测、分类和识别模型的格式
    return det_format, cls_format, rec_format

# 解析命令行参数
args = parse_arguments()

# 初始化模型文件和参数文件
det_model_file = args.det_model
det_params_file = ""
cls_model_file = args.cls_model
cls_params_file = ""
rec_model_file = args.rec_model
rec_params_file = ""
rec_label_file = args.rec_label_file

# 构建模型选项
det_option, cls_option, rec_option = build_option(args)
# 构建模型格式
det_format, cls_format, rec_format = build_format(args)

# 初始化检测模型
det_model = fd.vision.ocr.DBDetector(
    det_model_file,
    det_params_file,
    runtime_option=det_option,
    model_format=det_format)

# 初始化分类模型
cls_model = fd.vision.ocr.Classifier(
    cls_model_file,
    cls_params_file,
    runtime_option=cls_option,
    model_format=cls_format)

# 初始化识别模型
rec_model = fd.vision.ocr.Recognizer(
    rec_model_file,
    rec_params_file,
    rec_label_file,
    runtime_option=rec_option,
    model_format=rec_format)

# 设置Det和Rec模型的静态shape推理为True
det_model.preprocessor.static_shape_infer = True
rec_model.preprocessor.static_shape_infer = True

# 如果设备为"NPU"，则禁用Det、Cls和Rec模型的normalize和permute操作
if args.device == "npu":
    det_model.preprocessor.disable_normalize()
    det_model.preprocessor.disable_permute()
    cls_model.preprocessor.disable_normalize()
    cls_model.preprocessor.disable_permute()
    rec_model.preprocessor.disable_normalize()
    rec_model.preprocessor.disable_permute()

# 创建PP-OCR模型，串联Det、Cls和Rec模型
ppocr_v3 = fd.vision.ocr.PPOCRv3(
    det_model=det_model, cls_model=cls_model, rec_model=rec_model)

# 设置Cls和Rec模型的batch size为1，并开启静态shape推理
ppocr_v3.cls_batch_size = 1
ppocr_v3.rec_batch_size = 1

# 读取待预测的图片
im = cv2.imread(args.image)

# 进行预测并获取结果
result = ppocr_v3.predict(im)

# 打印预测结果
print(result)

# 可视化预测结果
vis_im = fd.vision.vis_ppocr(im, result)
# 将图像 vis_im 保存为 visualized_result.jpg 文件
cv2.imwrite("visualized_result.jpg", vis_im)
# 打印提示信息，说明可视化结果已保存在 ./visualized_result.jpg 文件中
print("Visualized result save in ./visualized_result.jpg")
```