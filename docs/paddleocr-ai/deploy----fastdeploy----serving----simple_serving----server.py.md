# `.\PaddleOCR\deploy\fastdeploy\serving\simple_serving\server.py`

```py
# 导入 fastdeploy 库中的模块并重命名为 fd
import fastdeploy as fd
# 从 fastdeploy.serving.server 模块中导入 SimpleServer 类
from fastdeploy.serving.server import SimpleServer
# 导入 os 模块
import os
# 导入 logging 模块
import logging

# 设置日志级别为 INFO
logging.getLogger().setLevel(logging.INFO)

# 配置信息
det_model_dir = 'ch_PP-OCRv3_det_infer'
cls_model_dir = 'ch_ppocr_mobile_v2.0_cls_infer'
rec_model_dir = 'ch_PP-OCRv3_rec_infer'
rec_label_file = 'ppocr_keys_v1.txt'
device = 'cpu'
# backend: ['paddle', 'trt']，也可以使用其他后端，但需要修改下面的运行时选项
backend = 'paddle'

# 准备模型
# 检测模型
det_model_file = os.path.join(det_model_dir, "inference.pdmodel")
det_params_file = os.path.join(det_model_dir, "inference.pdiparams")
# 分类模型
cls_model_file = os.path.join(cls_model_dir, "inference.pdmodel")
cls_params_file = os.path.join(cls_model_dir, "inference.pdiparams")
# 识别模型
rec_model_file = os.path.join(rec_model_dir, "inference.pdmodel")
rec_params_file = os.path.join(rec_model_dir, "inference.pdiparams")

# 设置运行时选项以选择硬件、后端等
option = fd.RuntimeOption()
# 如果设备为 GPU，则使用 GPU
if device.lower() == 'gpu':
    option.use_gpu()
# 如果后端为 trt，则使用 TensorRT 后端，否则使用 PaddlePaddle 推理后端
if backend == 'trt':
    option.use_trt_backend()
else:
    option.use_paddle_infer_backend()

# 检测模型选项
det_option = option
# 设置 TensorRT 输入形状
det_option.set_trt_input_shape("x", [1, 3, 64, 64], [1, 3, 640, 640],
                               [1, 3, 960, 960])

# 打印检测模型文件路径和参数文件路径
print(det_model_file, det_params_file)
# 创建 DBDetector 对象，用于检测模型
det_model = fd.vision.ocr.DBDetector(
    det_model_file, det_params_file, runtime_option=det_option)

# 分类模型批处理大小为 1
cls_batch_size = 1
# 识别模型批处理大小为 6

cls_option = option
# 设置分类模型 TensorRT 输入形状
cls_option.set_trt_input_shape("x", [1, 3, 48, 10],
                               [cls_batch_size, 3, 48, 320],
                               [cls_batch_size, 3, 48, 1024])

# 创建 Classifier 对象，用于分类模型
cls_model = fd.vision.ocr.Classifier(
    cls_model_file, cls_params_file, runtime_option=cls_option)

# 识别模型选项与通用选项相同
rec_option = option
# 设置 TRT 输入形状，包括输入名称 "x"，输入形状 [1, 3, 48, 10]，TRT最小输入形状 [rec_batch_size, 3, 48, 320]，TRT最大输入形状 [rec_batch_size, 3, 48, 2304]
rec_option.set_trt_input_shape("x", [1, 3, 48, 10],
                               [rec_batch_size, 3, 48, 320],
                               [rec_batch_size, 3, 48, 2304])

# 使用指定的模型文件、参数文件、标签文件以及运行时选项创建识别器模型对象
rec_model = fd.vision.ocr.Recognizer(
    rec_model_file, rec_params_file, rec_label_file, runtime_option=rec_option)

# 创建 PPOCRv3 流水线，包括检测模型、分类模型和识别模型
ppocr_v3 = fd.vision.ocr.PPOCRv3(
    det_model=det_model, cls_model=cls_model, rec_model=rec_model)

# 设置分类模型的批量大小
ppocr_v3.cls_batch_size = cls_batch_size
# 设置识别模型的批量大小
ppocr_v3.rec_batch_size = rec_batch_size

# 创建服务器对象，设置 REST API
app = SimpleServer()
# 注册任务名称为 "fd/ppocrv3"，模型处理程序为 VisionModelHandler，预测器为 ppocr_v3
app.register(
    task_name="fd/ppocrv3",
    model_handler=fd.serving.handler.VisionModelHandler,
    predictor=ppocr_v3)
```