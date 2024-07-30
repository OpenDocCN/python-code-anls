# `.\yolov8\tests\test_cuda.py`

```py
# 从标准库导入product函数，用于生成可迭代对象的笛卡尔积
from itertools import product
# 从路径处理模块中导入Path类
from pathlib import Path

# 导入pytest库，用于编写和运行测试
import pytest
# 导入PyTorch库
import torch

# 从tests模块导入CUDA设备相关的变量和对象
from tests import CUDA_DEVICE_COUNT, CUDA_IS_AVAILABLE, MODEL, SOURCE
# 从ultralytics库中导入YOLO类
from ultralytics import YOLO
# 从ultralytics.cfg模块导入任务相关的字典和对象
from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS
# 从ultralytics.utils模块导入一些常量和路径
from ultralytics.utils import ASSETS, WEIGHTS_DIR


def test_checks():
    """Validate CUDA settings against torch CUDA functions."""
    # 断言当前环境中CUDA是否可用
    assert torch.cuda.is_available() == CUDA_IS_AVAILABLE
    # 断言当前环境中的CUDA设备数量
    assert torch.cuda.device_count() == CUDA_DEVICE_COUNT


@pytest.mark.slow
# 标记为跳过测试，原因是等待更多Ultralytics GPU服务器可用性
@pytest.mark.skipif(True, reason="CUDA export tests disabled pending additional Ultralytics GPU server availability")
# 标记为跳过测试，如果CUDA不可用
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
# 参数化测试，传入多个参数组合
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch",
    [  # 生成所有可能的组合，但排除int8和half同时为True的情况
        (task, dynamic, int8, half, batch)
        # 注意：由于GPU CI运行器的利用率较高，下面的测试数量有所减少
        # task, dynamic, int8, half, batch in product(TASKS, [True, False], [True, False], [True, False], [1, 2])
        for task, dynamic, int8, half, batch in product(TASKS, [True], [True], [False], [2])
        if not (int8 and half)  # 排除同时int8和half为True的情况
    ],
)
def test_export_engine_matrix(task, dynamic, int8, half, batch):
    """Test YOLO model export to TensorRT format for various configurations and run inference."""
    # 使用YOLO模型对象导出到TensorRT格式
    file = YOLO(TASK2MODEL[task]).export(
        format="engine",
        imgsz=32,
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
        data=TASK2DATA[task],
        workspace=1,  # 在测试期间减少工作空间，以减少资源利用
        simplify=True,  # 使用'onnxslim'简化模型
    )
    # 使用导出的模型进行推理
    YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32)  # 导出模型的推理
    # 清理生成的文件
    Path(file).unlink()
    # 如果使用了INT8量化，还需清理缓存文件
    Path(file).with_suffix(".cache").unlink() if int8 else None  # 清理INT8缓存


# 标记为跳过测试，如果CUDA不可用
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
def test_train():
    """Test model training on a minimal dataset using available CUDA devices."""
    # 确定使用的CUDA设备编号，如果只有一个设备可用则为0，否则为列表[0, 1]
    device = 0 if CUDA_DEVICE_COUNT == 1 else [0, 1]
    # 使用YOLO模型对象进行训练
    YOLO(MODEL).train(data="coco8.yaml", imgsz=64, epochs=1, device=device)  # 需要imgsz>=64


@pytest.mark.slow
# 标记为跳过测试，如果CUDA不可用
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
def test_predict_multiple_devices():
    """Validate model prediction consistency across CPU and CUDA devices."""
    # 创建YOLO模型对象
    model = YOLO("yolov8n.pt")
    # 将模型转移到CPU上
    model = model.cpu()
    # 断言模型当前是否在CPU上
    assert str(model.device) == "cpu"
    # 使用CPU进行推理
    _ = model(SOURCE)  # CPU推理
    # 再次断言模型当前是否在CPU上
    assert str(model.device) == "cpu"

    # 将模型切换到CUDA设备cuda:0上
    model = model.to("cuda:0")
    # 断言模型当前是否在cuda:0上
    assert str(model.device) == "cuda:0"
    # 使用CUDA设备进行推理
    _ = model(SOURCE)  # CUDA推理
    # 再次断言模型当前是否在cuda:0上
    assert str(model.device) == "cuda:0"

    # 将模型切换回CPU
    model = model.cpu()
    # 断言模型当前是否在CPU上
    assert str(model.device) == "cpu"
    # 使用模型进行 CPU 推理
    _ = model(SOURCE)  # CPU inference
    # 断言当前模型设备为 CPU
    assert str(model.device) == "cpu"
    
    # 将模型切换到 CUDA 设备
    model = model.cuda()
    # 断言当前模型设备为 CUDA 设备的第一个 GPU (cuda:0)
    assert str(model.device) == "cuda:0"
    
    # 使用模型进行 CUDA 设备上的推理
    _ = model(SOURCE)  # CUDA inference
    # 断言当前模型设备为 CUDA 设备的第一个 GPU (cuda:0)
    assert str(model.device) == "cuda:0"
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
def test_autobatch():
    """Check optimal batch size for YOLO model training using autobatch utility."""
    from ultralytics.utils.autobatch import check_train_batch_size

    # 调用自动批处理实用程序，检查 YOLO 模型训练的最佳批处理大小
    check_train_batch_size(YOLO(MODEL).model.cuda(), imgsz=128, amp=True)


@pytest.mark.slow
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
def test_utils_benchmarks():
    """Profile YOLO models for performance benchmarks."""
    from ultralytics.utils.benchmarks import ProfileModels

    # 导出动态引擎模型以进行动态推理
    YOLO(MODEL).export(format="engine", imgsz=32, dynamic=True, batch=1)
    # 对 YOLO 模型进行性能基准测试
    ProfileModels([MODEL], imgsz=32, half=False, min_time=1, num_timed_runs=3, num_warmup_runs=1).profile()


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
def test_predict_sam():
    """Test SAM model predictions using different prompts, including bounding boxes and point annotations."""
    from ultralytics import SAM
    from ultralytics.models.sam import Predictor as SAMPredictor

    # 加载 SAM 模型
    model = SAM(WEIGHTS_DIR / "sam_b.pt")

    # 显示模型信息（可选）
    model.info()

    # 进行推理
    model(SOURCE, device=0)

    # 使用边界框提示进行推理
    model(SOURCE, bboxes=[439, 437, 524, 709], device=0)

    # 使用点注释进行推理
    model(ASSETS / "zidane.jpg", points=[900, 370], labels=[1], device=0)

    # 创建 SAMPredictor 实例
    overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model=WEIGHTS_DIR / "mobile_sam.pt")
    predictor = SAMPredictor(overrides=overrides)

    # 设置图像
    predictor.set_image(ASSETS / "zidane.jpg")  # 使用图像文件设置

    # 重置图像
    predictor.reset_image()
```