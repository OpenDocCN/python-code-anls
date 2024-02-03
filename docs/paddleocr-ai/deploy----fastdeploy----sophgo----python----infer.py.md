# `.\PaddleOCR\deploy\fastdeploy\sophgo\python\infer.py`

```
# 导入fastdeploy库，cv2库和os库
import fastdeploy as fd
import cv2
import os

# 解析命令行参数
def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
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
        help="Path of Recognization label of PPOCR.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")

    return parser.parse_args()

# 解析命令行参数并存储在args变量中
args = parse_arguments()

# 配置runtime，加载模型
runtime_option = fd.RuntimeOption()
runtime_option.use_sophgo()

# Detection模型, 检测文字框
det_model_file = args.det_model
det_params_file = ""

# Classification模型，方向分类，可选
cls_model_file = args.cls_model
cls_params_file = ""

# Recognition模型，文字识别模型
rec_model_file = args.rec_model
rec_params_file = ""
rec_label_file = args.rec_label_file

# PPOCR的cls和rec模型现在已经支持推理一个Batch的数据
# 定义下面两个变量后, 可用于设置trt输入shape, 并在PPOCR模型初始化后, 完成Batch推理设置
cls_batch_size = 1
rec_batch_size = 1

# 当使用TRT时，分别给三个模型的runtime设置动态shape,并完成模型的创建.
# 注意: 需要在检测模型创建完成后，再设置分类模型的动态输入并创建分类模型, 识别模型同理.
# 如果用户想要自己改动检测模型的输入shape, 我们建议用户把检测模型的长和高设置为32的倍数.
det_option = runtime_option
det_option.set_trt_input_shape("x", [1, 3, 64, 64], [1, 3, 640, 640],
                               [1, 3, 960, 960])
# 用户可以把TRT引擎文件保存至本地
# det_option.set_trt_cache_file(args.det_model  + "/det_trt_cache.trt")
det_model = fd.vision.ocr.DBDetector(
    det_model_file,
    det_params_file,
    runtime_option=det_option,
    model_format=fd.ModelFormat.SOPHGO)

cls_option = runtime_option
# 设置TRT引擎输入形状，包括最小、优化和最大形状
cls_option.set_trt_input_shape("x", [1, 3, 48, 10],
                               [cls_batch_size, 3, 48, 320],
                               [cls_batch_size, 3, 48, 1024])
# 用户可以把TRT引擎文件保存至本地
# cls_option.set_trt_cache_file(args.cls_model  + "/cls_trt_cache.trt")

# 创建分类器模型对象，传入模型文件、参数文件、运行时选项和模型格式
cls_model = fd.vision.ocr.Classifier(
    cls_model_file,
    cls_params_file,
    runtime_option=cls_option,
    model_format=fd.ModelFormat.SOPHGO)

# 将rec_option设置为runtime_option
rec_option = runtime_option
# 设置TRT引擎输入形状，包括最小、优化和最大形状
rec_option.set_trt_input_shape("x", [1, 3, 48, 10],
                               [rec_batch_size, 3, 48, 320],
                               [rec_batch_size, 3, 48, 2304])
# 用户可以把TRT引擎文件保存至本地
# rec_option.set_trt_cache_file(args.rec_model  + "/rec_trt_cache.trt")

# 创建识别器模型对象，传入模型文件、参数文件、标签文件、运行时选项和模型格式
rec_model = fd.vision.ocr.Recognizer(
    rec_model_file,
    rec_params_file,
    rec_label_file,
    runtime_option=rec_option,
    model_format=fd.ModelFormat.SOPHGO)

# 创建PP-OCR对象，串联分类器、检测器和识别器模型
ppocr_v3 = fd.vision.ocr.PPOCRv3(
    det_model=det_model, cls_model=cls_model, rec_model=rec_model)

# 启用rec模型的静态shape推理，并设置静态输入形状为[3, 48, 584]
rec_model.preprocessor.static_shape_infer = True
rec_model.preprocessor.rec_image_shape = [3, 48, 584]

# 设置cls和rec模型的推理时batch size
ppocr_v3.cls_batch_size = cls_batch_size
ppocr_v3.rec_batch_size = rec_batch_size

# 读取待预测的图片
im = cv2.imread(args.image)

# 预测并获取结果
result = ppocr_v3.predict(im)

# 打印预测结果
print(result)

# 可视化预测结果并保存图片
vis_im = fd.vision.vis_ppocr(im, result)
cv2.imwrite("sophgo_result.jpg", vis_im)
print("Visualized result save in ./sophgo_result.jpg")
```