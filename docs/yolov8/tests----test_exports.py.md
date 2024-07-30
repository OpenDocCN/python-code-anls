# `.\yolov8\tests\test_exports.py`

```py
# 导入所需的库和模块
import shutil  # 文件操作工具，用于复制、移动和删除文件和目录
import uuid  # 用于生成唯一的UUID
from itertools import product  # 用于生成迭代器的笛卡尔积
from pathlib import Path  # 用于处理文件和目录路径的类

import pytest  # 测试框架

# 导入测试所需的模块和函数
from tests import MODEL, SOURCE
from ultralytics import YOLO  # 导入YOLO模型
from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS  # 导入配置信息
from ultralytics.utils import (
    IS_RASPBERRYPI,  # 检查是否在树莓派上运行
    LINUX,  # 检查是否在Linux系统上运行
    MACOS,  # 检查是否在macOS系统上运行
    WINDOWS,  # 检查是否在Windows系统上运行
    checks,  # 各种系统和Python版本的检查工具集合
)
from ultralytics.utils.torch_utils import TORCH_1_9, TORCH_1_13  # Torch相关的工具函数和版本检查
# 测试导出 YOLO 模型到 ONNX 格式，使用不同的配置和参数进行测试
def test_export_onnx_matrix(task, dynamic, int8, half, batch, simplify):
    # 调用 YOLO 类，根据任务选择相应的模型，然后导出为 ONNX 格式的文件
    file = YOLO(TASK2MODEL[task]).export(
        format="onnx",
        imgsz=32,
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
        simplify=simplify,
    )
    # 使用导出的模型进行推理，传入相同的源数据多次以达到批处理要求
    YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32)  # exported model inference
    # 清理生成的文件
    Path(file).unlink()  # cleanup


@pytest.mark.slow
@pytest.mark.parametrize("task, dynamic, int8, half, batch", product(TASKS, [False], [False], [False], [1, 2]))
# 测试导出 YOLO 模型到 TorchScript 格式，考虑不同的配置和参数组合
def test_export_torchscript_matrix(task, dynamic, int8, half, batch):
    # 调用 YOLO 类，根据任务选择相应的模型，然后导出为 TorchScript 格式的文件
    file = YOLO(TASK2MODEL[task]).export(
        format="torchscript",
        imgsz=32,
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
    )
    # 使用导出的模型进行推理，传入特定的源数据以达到批处理要求
    YOLO(file)([SOURCE] * 3, imgsz=64 if dynamic else 32)  # exported model inference at batch=3
    # 清理生成的文件
    Path(file).unlink()  # cleanup


@pytest.mark.slow
# 在 macOS 上测试导出 YOLO 模型到 CoreML 格式，使用各种参数配置
@pytest.mark.skipif(not MACOS, reason="CoreML inference only supported on macOS")
@pytest.mark.skipif(not TORCH_1_9, reason="CoreML>=7.2 not supported with PyTorch<=1.8")
@pytest.mark.skipif(checks.IS_PYTHON_3_12, reason="CoreML not supported in Python 3.12")
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch",
    [  # 生成所有组合，但排除 int8 和 half 都为 True 的情况
        (task, dynamic, int8, half, batch)
        for task, dynamic, int8, half, batch in product(TASKS, [False], [True, False], [True, False], [1])
        if not (int8 and half)  # 排除 int8 和 half 都为 True 的情况
    ],
)
def test_export_coreml_matrix(task, dynamic, int8, half, batch):
    # 调用 YOLO 类，根据任务选择相应的模型，然后导出为 CoreML 格式的文件
    file = YOLO(TASK2MODEL[task]).export(
        format="coreml",
        imgsz=32,
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
    )
    # 使用导出的模型进行推理，传入特定的源数据以达到批处理要求
    YOLO(file)([SOURCE] * batch, imgsz=32)  # exported model inference at batch=3
    # 清理生成的文件夹
    shutil.rmtree(file)  # cleanup


@pytest.mark.slow
# 在 Python 版本大于等于 3.10 时，在 Linux 上测试导出 YOLO 模型到 TFLite 格式
@pytest.mark.skipif(not checks.IS_PYTHON_MINIMUM_3_10, reason="TFLite export requires Python>=3.10")
@pytest.mark.skipif(not LINUX, reason="Test disabled as TF suffers from install conflicts on Windows and macOS")
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch",
    [  # 生成所有组合，但排除 int8 和 half 都为 True 的情况
        (task, dynamic, int8, half, batch)
        for task, dynamic, int8, half, batch in product(TASKS, [False], [True, False], [True, False], [1])
        if not (int8 and half)  # 排除 int8 和 half 都为 True 的情况
    ],
)
# 测试导出 YOLO 模型到 TFLite 格式，考虑各种导出配置
def test_export_tflite_matrix(task, dynamic, int8, half, batch):
    # 调用 YOLO 类，根据任务选择相应的模型，然后导出为 TFLite 格式的文件
    file = YOLO(TASK2MODEL[task]).export(
        format="tflite",
        imgsz=32,
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
    )
    # 使用导出的模型进行推理，传入特定的源数据以达到批处理要求
    YOLO(file)([SOURCE] * batch, imgsz=32)  # exported model inference at batch=3
    # 清理生成的文件夹
    shutil.rmtree(file)  # cleanup
    # 使用指定任务的模型从YOLO导出模型，并以tflite格式输出到文件
    file = YOLO(TASK2MODEL[task]).export(
        format="tflite",
        imgsz=32,
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
    )
    
    # 使用导出的模型进行推理，输入为[SOURCE]的重复项，批量大小为3，图像尺寸为32
    YOLO(file)([SOURCE] * batch, imgsz=32)  # 批量大小为3时导出模型的推理
    
    # 删除导出的模型文件，进行清理工作
    Path(file).unlink()  # 清理
# 根据条件跳过测试，若 TORCH_1_9 为假则跳过，提示 PyTorch<=1.8 不支持 CoreML>=7.2
@pytest.mark.skipif(not TORCH_1_9, reason="CoreML>=7.2 not supported with PyTorch<=1.8")
# 若在 Windows 系统上则跳过，提示 CoreML 在 Windows 上不受支持
@pytest.mark.skipif(WINDOWS, reason="CoreML not supported on Windows")  # RuntimeError: BlobWriter not loaded
# 若在树莓派上则跳过，提示 CoreML 在树莓派上不受支持
@pytest.mark.skipif(IS_RASPBERRYPI, reason="CoreML not supported on Raspberry Pi")
# 若 Python 版本为 3.12 则跳过，提示 CoreML 不支持 Python 3.12
@pytest.mark.skipif(checks.IS_PYTHON_3_12, reason="CoreML not supported in Python 3.12")
def test_export_coreml():
    """Test YOLO exports to CoreML format, optimized for macOS only."""
    if MACOS:
        # 在 macOS 上导出 YOLO 模型到 CoreML 格式，并优化为指定的 imgsz 大小
        file = YOLO(MODEL).export(format="coreml", imgsz=32)
        # 使用导出的 CoreML 模型进行预测，仅支持在 macOS 上进行，对于 nms=False 的模型
        YOLO(file)(SOURCE, imgsz=32)  # model prediction only supported on macOS for nms=False models
    else:
        # 在非 macOS 系统上导出 YOLO 模型到 CoreML 格式，使用默认的 nms=True 和指定的 imgsz 大小
        YOLO(MODEL).export(format="coreml", nms=True, imgsz=32)


# 若 Python 版本小于 3.10 则跳过，提示 TFLite 导出要求 Python>=3.10
@pytest.mark.skipif(not checks.IS_PYTHON_MINIMUM_3_10, reason="TFLite export requires Python>=3.10")
# 若不在 Linux 系统上则跳过，提示在 Windows 和 macOS 上 TensorFlow 安装可能会冲突
@pytest.mark.skipif(not LINUX, reason="Test disabled as TF suffers from install conflicts on Windows and macOS")
def test_export_tflite():
    """Test YOLO exports to TFLite format under specific OS and Python version conditions."""
    # 创建 YOLO 模型对象
    model = YOLO(MODEL)
    # 导出 YOLO 模型到 TFLite 格式，使用指定的 imgsz 大小
    file = model.export(format="tflite", imgsz=32)
    # 使用导出的 TFLite 模型进行预测
    YOLO(file)(SOURCE, imgsz=32)


# 直接跳过此测试，无特定原因说明
@pytest.mark.skipif(True, reason="Test disabled")
# 若不在 Linux 系统上则跳过，提示 TensorFlow 在 Windows 和 macOS 上安装可能会冲突
@pytest.mark.skipif(not LINUX, reason="TF suffers from install conflicts on Windows and macOS")
def test_export_pb():
    """Test YOLO exports to TensorFlow's Protobuf (*.pb) format."""
    # 创建 YOLO 模型对象
    model = YOLO(MODEL)
    # 导出 YOLO 模型到 TensorFlow 的 Protobuf 格式，使用指定的 imgsz 大小
    file = model.export(format="pb", imgsz=32)
    # 使用导出的 Protobuf 模型进行预测
    YOLO(file)(SOURCE, imgsz=32)


# 直接跳过此测试，无特定原因说明
@pytest.mark.skipif(True, reason="Test disabled as Paddle protobuf and ONNX protobuf requirementsk conflict.")
def test_export_paddle():
    """Test YOLO exports to Paddle format, noting protobuf conflicts with ONNX."""
    # 导出 YOLO 模型到 Paddle 格式，使用指定的 imgsz 大小
    YOLO(MODEL).export(format="paddle", imgsz=32)


# 标记为慢速测试
@pytest.mark.slow
def test_export_ncnn():
    """Test YOLO exports to NCNN format."""
    # 导出 YOLO 模型到 NCNN 格式，使用指定的 imgsz 大小
    file = YOLO(MODEL).export(format="ncnn", imgsz=32)
    # 使用导出的 NCNN 模型进行预测
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference
```