# `.\PaddleOCR\deploy\fastdeploy\ascend\python\infer.py`

```py
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
    # 解析命令行参数并返回结果
    return parser.parse_args()

# 构建运行时选项
def build_option(args):

    # 创建三个运行时选项对象
    det_option = fd.RuntimeOption()
    cls_option = fd.RuntimeOption()
    rec_option = fd.RuntimeOption()

    # 使用 Ascend 设备
    det_option.use_ascend()
    cls_option.use_ascend()
    rec_option.use_ascend()

    # 返回三个运行时选项对象
    return det_option, cls_option, rec_option

# 解析命令行参数
args = parse_arguments()

# 拼接检测模型文件路径
det_model_file = os.path.join(args.det_model, "inference.pdmodel")
# 拼接检测模型参数文件路径
det_params_file = os.path.join(args.det_model, "inference.pdiparams")

# 拼接分类模型文件路径
cls_model_file = os.path.join(args.cls_model, "inference.pdmodel")
# 拼接分类模型参数文件路径
cls_params_file = os.path.join(args.cls_model, "inference.pdiparams")
# 构建识别模型文件路径
rec_model_file = os.path.join(args.rec_model, "inference.pdmodel")
# 构建识别模型参数文件路径
rec_params_file = os.path.join(args.rec_model, "inference.pdiparams")
# 获取识别标签文件路径
rec_label_file = args.rec_label_file

# 构建检测、分类、识别模型的运行选项
det_option, cls_option, rec_option = build_option(args)

# 创建文本检测模型对象
det_model = fd.vision.ocr.DBDetector(
    det_model_file, det_params_file, runtime_option=det_option)

# 创建文本分类模型对象
cls_model = fd.vision.ocr.Classifier(
    cls_model_file, cls_params_file, runtime_option=cls_option)

# 创建文本识别模型对象
rec_model = fd.vision.ocr.Recognizer(
    rec_model_file, rec_params_file, rec_label_file, runtime_option=rec_option)

# 设置识别模型启用静态形状推断
# 在Ascend上部署时，必须为True
rec_model.preprocessor.static_shape_infer = True

# 创建PP-OCRv3对象，如果不需要分类模型，将cls_model设置为None
ppocr_v3 = fd.vision.ocr.PPOCRv3(
    det_model=det_model, cls_model=cls_model, rec_model=rec_model)

# 当启用静态形状推断时，批处理大小必须设置为1
ppocr_v3.cls_batch_size = 1
ppocr_v3.rec_batch_size = 1

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