# `.\PaddleOCR\deploy\fastdeploy\kunlunxin\python\infer.py`

```
# 导入所需的库
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
        "--cls_bs",
        type=int,
        default=1,
        help="Classification model inference batch size.")
    parser.add_argument(
        "--rec_bs",
        type=int,
        default=6,
        help="Recognition model inference batch size")
    # 解析命令行参数并返回结果
    return parser.parse_args()

# 构建运行时选项
def build_option(args):

    # 创建三个 RuntimeOption 对象
    det_option = fd.RuntimeOption()
    cls_option = fd.RuntimeOption()
    rec_option = fd.RuntimeOption()

    # 使用鲲鹏芯片进行推理
    det_option.use_kunlunxin()
    cls_option.use_kunlunxin()
    rec_option.use_kunlunxin()

    # 返回三个运行时选项对象
    return det_option, cls_option, rec_option

# 解析命令行参数
args = parse_arguments()
# 拼接目标检测模型文件路径
det_model_file = os.path.join(args.det_model, "inference.pdmodel")
# 拼接目标检测模型参数文件路径
det_params_file = os.path.join(args.det_model, "inference.pdiparams")

# 拼接分类模型文件路径
cls_model_file = os.path.join(args.cls_model, "inference.pdmodel")
# 拼接分类模型参数文件路径
cls_params_file = os.path.join(args.cls_model, "inference.pdiparams")

# 拼接识别模型文件路径
rec_model_file = os.path.join(args.rec_model, "inference.pdmodel")
# 拼接识别模型参数文件路径
rec_params_file = os.path.join(args.rec_model, "inference.pdiparams")
# 获取识别标签文件路径
rec_label_file = args.rec_label_file

# 构建目标检测、分类和识别模型的运行选项
det_option, cls_option, rec_option = build_option(args)

# 创建目标检测模型对象
det_model = fd.vision.ocr.DBDetector(
    det_model_file, det_params_file, runtime_option=det_option)

# 创建分类模型对象
cls_model = fd.vision.ocr.Classifier(
    cls_model_file, cls_params_file, runtime_option=cls_option)

# 创建识别模型对象
rec_model = fd.vision.ocr.Recognizer(
    rec_model_file, rec_params_file, rec_label_file, runtime_option=rec_option)

# 创建 PP-OCRv3 模型对象，如果不需要分类模型，将 cls_model 设置为 None
ppocr_v3 = fd.vision.ocr.PPOCRv3(
    det_model=det_model, cls_model=cls_model, rec_model=rec_model)

# 设置分类模型和识别模型的推理批处理大小，值可以是 -1 到正无穷
# 当推理批处理大小设置为 -1 时，表示分类和识别模型的推理批处理大小将与目标检测模型检测到的框的数量相同
ppocr_v3.cls_batch_size = args.cls_bs
ppocr_v3.rec_batch_size = args.rec_bs

# 准备图像
im = cv2.imread(args.image)

# 打印结果
result = ppocr_v3.predict(im)

print(result)

# 可视化输出
vis_im = fd.vision.vis_ppocr(im, result)
cv2.imwrite("visualized_result.jpg", vis_im)
print("Visualized result save in ./visualized_result.jpg")
```