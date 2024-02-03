# `.\PaddleOCR\deploy\fastdeploy\cpu-gpu\python\infer.py`

```
# 导入所需的库
import fastdeploy as fd
import cv2
import os

# 定义解析命令行参数的函数
def parse_arguments():
    # 导入 argparse 库，用于解析命令行参数
    import argparse
    import ast
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数，指定检测模型的路径
    parser.add_argument(
        "--det_model", required=True, help="Path of Detection model of PPOCR.")
    # 添加命令行参数，指定分类模型的路径
    parser.add_argument(
        "--cls_model",
        required=True,
        help="Path of Classification model of PPOCR.")
    # 添加命令行参数，指定识别模型的路径
    parser.add_argument(
        "--rec_model",
        required=True,
        help="Path of Recognization model of PPOCR.")
    # 添加命令行参数，指定识别模型的标签文件路径
    parser.add_argument(
        "--rec_label_file",
        required=True,
        help="Path of Recognization model of PPOCR.")
    # 添加命令行参数，指定测试图像文件的路径
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")
    # 添加命令行参数，指定推理设备类型，默认为 CPU
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'cpu' or 'gpu'.")
    # 添加命令行参数，指定使用的 GPU 卡号，默认为 0
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Define which GPU card used to run model.")
    # 添加命令行参数，指定分类模型推理批大小，默认为 1
    parser.add_argument(
        "--cls_bs",
        type=int,
        default=1,
        help="Classification model inference batch size.")
    # 添加命令行参数，指定识别模型推理批大小，默认为 6
    parser.add_argument(
        "--rec_bs",
        type=int,
        default=6,
        help="Recognition model inference batch size")
    # 添加一个命令行参数，用于指定推理后端类型，默认为"default"
    parser.add_argument(
        "--backend",
        type=str,
        default="default",
        help="Type of inference backend, support ort/trt/paddle/openvino, default 'openvino' for cpu, 'tensorrt' for gpu"
    )

    # 解析命令行参数并返回结果
    return parser.parse_args()
# 构建运行时选项对象，用于设置模型推理的参数
def build_option(args):

    # 创建三个运行时选项对象，分别用于检测、分类和识别模型
    det_option = fd.RuntimeOption()
    cls_option = fd.RuntimeOption()
    rec_option = fd.RuntimeOption()

    # 如果设备为 GPU，则设置检测、分类和识别模型均使用 GPU
    if args.device.lower() == "gpu":
        det_option.use_gpu(args.device_id)
        cls_option.use_gpu(args.device_id)
        rec_option.use_gpu(args.device_id)

    # 如果选择的后端为 TensorRT
    if args.backend.lower() == "trt":
        # 如果设备不是 GPU，则抛出异常
        assert args.device.lower() == "gpu", "TensorRT backend require inference on device GPU."
        # 设置检测、分类和识别模型使用 TensorRT 后端
        det_option.use_trt_backend()
        cls_option.use_trt_backend()
        rec_option.use_trt_backend()

        # 如果使用了 TensorRT 后端，设置动态形状如下
        # 建议用户将检测模型的长度和高度设置为 32 的倍数
        # 建议用户将 Trt 输入形状设置如下
        det_option.set_trt_input_shape("x", [1, 3, 64, 64], [1, 3, 640, 640],
                                       [1, 3, 960, 960])
        cls_option.set_trt_input_shape("x", [1, 3, 48, 10],
                                       [args.cls_bs, 3, 48, 320],
                                       [args.cls_bs, 3, 48, 1024])
        rec_option.set_trt_input_shape("x", [1, 3, 48, 10],
                                       [args.rec_bs, 3, 48, 320],
                                       [args.rec_bs, 3, 48, 2304])

        # 用户可以将 TRT 缓存文件保存到磁盘上
        det_option.set_trt_cache_file(args.det_model + "/det_trt_cache.trt")
        cls_option.set_trt_cache_file(args.cls_model + "/cls_trt_cache.trt")
        rec_option.set_trt_cache_file(args.rec_model + "/rec_trt_cache.trt")
    # 如果选择的后端是 Paddle-TensorRT，则需要确保推理设备是 GPU
    elif args.backend.lower() == "pptrt":
        assert args.device.lower() == "gpu", "Paddle-TensorRT backend require inference on device GPU."
        # 设置检测模型选项为使用 Paddle 推理后端
        det_option.use_paddle_infer_backend()
        # 设置检测模型选项为收集 TRT 的形状信息
        det_option.paddle_infer_option.collect_trt_shape = True
        # 设置检测模型选项为启用 TRT
        det_option.paddle_infer_option.enable_trt = True

        # 设置分类模型选项为使用 Paddle 推理后端
        cls_option.use_paddle_infer_backend()
        # 设置分类模型选项为收集 TRT 的形状信息
        cls_option.paddle_infer_option.collect_trt_shape = True
        # 设置分类模型选项为启用 TRT
        cls_option.paddle_infer_option.enable_trt = True

        # 设置识别模型选项为使用 Paddle 推理后端
        rec_option.use_paddle_infer_backend()
        # 设置识别模型选项为收集 TRT 的形状信息
        rec_option.paddle_infer_option.collect_trt_shape = True
        # 设置识别模型选项为启用 TRT
        rec_option.paddle_infer_option.enable_trt = True

        # 如果使用 TRT 后端，则设置动态形状如下
        # 建议用户将检测模型的长度和高度设置为 32 的倍数
        # 建议用户设置 TRT 输入形状如下
        det_option.set_trt_input_shape("x", [1, 3, 64, 64], [1, 3, 640, 640],
                                       [1, 3, 960, 960])
        cls_option.set_trt_input_shape("x", [1, 3, 48, 10],
                                       [args.cls_bs, 3, 48, 320],
                                       [args.cls_bs, 3, 48, 1024])
        rec_option.set_trt_input_shape("x", [1, 3, 48, 10],
                                       [args.rec_bs, 3, 48, 320],
                                       [args.rec_bs, 3, 48, 2304])

        # 用户可以将 TRT 缓存文件保存到磁盘上
        det_option.set_trt_cache_file(args.det_model)
        cls_option.set_trt_cache_file(args.cls_model)
        rec_option.set_trt_cache_file(args.rec_model)

    # 如果选择的后端是 ONNX Runtime，则设置检测、分类和识别模型选项为使用 ORT 后端
    elif args.backend.lower() == "ort":
        det_option.use_ort_backend()
        cls_option.use_ort_backend()
        rec_option.use_ort_backend()

    # 如果选择的后端是 Paddle 推理，则设置检测、分类和识别模型选项为使用 Paddle 推理后端
    elif args.backend.lower() == "paddle":
        det_option.use_paddle_infer_backend()
        cls_option.use_paddle_infer_backend()
        rec_option.use_paddle_infer_backend()
    # 如果选择的后端是 OpenVINO
    elif args.backend.lower() == "openvino":
        # 断言推理设备为 CPU，因为 OpenVINO 后端需要在 CPU 上进行推理
        assert args.device.lower() == "cpu", "OpenVINO backend require inference on device CPU."
        # 使用 OpenVINO 后端进行检测
        det_option.use_openvino_backend()
        # 使用 OpenVINO 后端进行分类
        cls_option.use_openvino_backend()
        # 使用 OpenVINO 后端进行识别
        rec_option.use_openvino_backend()

    # 如果选择的后端是 Paddle Lite
    elif args.backend.lower() == "pplite":
        # 断言推理设备为 CPU，因为 Paddle Lite 后端需要在 CPU 上进行推理
        assert args.device.lower() == "cpu", "Paddle Lite backend require inference on device CPU."
        # 使用 Paddle Lite 后端进行检测
        det_option.use_lite_backend()
        # 使用 Paddle Lite 后端进行分类
        cls_option.use_lite_backend()
        # 使用 Paddle Lite 后端进行识别
        rec_option.use_lite_backend()

    # 返回检测、分类和识别的选项
    return det_option, cls_option, rec_option
# 解析命令行参数
args = parse_arguments()

# 检测模型文件路径
det_model_file = os.path.join(args.det_model, "inference.pdmodel")
# 检测模型参数文件路径
det_params_file = os.path.join(args.det_model, "inference.pdiparams")

# 分类模型文件路径
cls_model_file = os.path.join(args.cls_model, "inference.pdmodel")
# 分类模型参数文件路径
cls_params_file = os.path.join(args.cls_model, "inference.pdiparams")

# 识别模型文件路径
rec_model_file = os.path.join(args.rec_model, "inference.pdmodel")
# 识别模型参数文件路径
rec_params_file = os.path.join(args.rec_model, "inference.pdiparams")
# 识别模型标签文件路径
rec_label_file = args.rec_label_file

# 构建检测、分类、识别模型的选项
det_option, cls_option, rec_option = build_option(args)

# 创建检测模型对象
det_model = fd.vision.ocr.DBDetector(
    det_model_file, det_params_file, runtime_option=det_option)

# 创建分类模型对象
cls_model = fd.vision.ocr.Classifier(
    cls_model_file, cls_params_file, runtime_option=cls_option)

# 创建识别模型对象
rec_model = fd.vision.ocr.Recognizer(
    rec_model_file, rec_params_file, rec_label_file, runtime_option=rec_option)

# 设置检测模型的预处理和后处理参数
det_model.preprocessor.max_side_len = 960
det_model.postprocessor.det_db_thresh = 0.3
det_model.postprocessor.det_db_box_thresh = 0.6
det_model.postprocessor.det_db_unclip_ratio = 1.5
det_model.postprocessor.det_db_score_mode = "slow"
det_model.postprocessor.use_dilation = False
cls_model.postprocessor.cls_thresh = 0.9

# 创建PP-OCRv3对象，如果不需要分类模型，将cls_model设置为None
ppocr_v3 = fd.vision.ocr.PPOCRv3(
    det_model=det_model, cls_model=cls_model, rec_model=rec_model)

# 设置分类模型和识别模型的推理批处理大小
ppocr_v3.cls_batch_size = args.cls_bs
ppocr_v3.rec_batch_size = args.rec_bs

# 读取输入图像
im = cv2.imread(args.image)

# 预测并返回结果
result = ppocr_v3.predict(im)

# 打印结果
print(result)

# 可视化结果
# 使用 fd.vision.vis_ppocr 函数对图像进行可视化处理，返回处理后的图像
vis_im = fd.vision.vis_ppocr(im, result)
# 将处理后的图像保存为 visualized_result.jpg 文件
cv2.imwrite("visualized_result.jpg", vis_im)
# 打印提示信息，说明可视化结果已保存在 ./visualized_result.jpg 文件中
print("Visualized result save in ./visualized_result.jpg")
```