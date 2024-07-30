# `.\yolov8\tests\test_cli.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

# 导入必要的模块
import subprocess  # 用于执行系统命令
import pytest  # 测试框架
from PIL import Image  # Python Imaging Library，处理图像

# 导入自定义模块和变量
from tests import CUDA_DEVICE_COUNT, CUDA_IS_AVAILABLE  # CUDA 设备数量和可用性
from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS  # YOLO 配置相关
from ultralytics.utils import ASSETS, WEIGHTS_DIR, checks  # YOLO 工具函数和资源路径

# 常量定义
TASK_MODEL_DATA = [(task, WEIGHTS_DIR / TASK2MODEL[task], TASK2DATA[task]) for task in TASKS]  # 任务模型数据元组列表
MODELS = [WEIGHTS_DIR / TASK2MODEL[task] for task in TASKS]  # 模型路径列表


def run(cmd):
    """Execute a shell command using subprocess."""
    subprocess.run(cmd.split(), check=True)  # 使用 subprocess 模块执行系统命令


def test_special_modes():
    """Test various special command-line modes for YOLO functionality."""
    run("yolo help")  # 执行 yolo help 命令
    run("yolo checks")  # 执行 yolo checks 命令
    run("yolo version")  # 执行 yolo version 命令
    run("yolo settings reset")  # 执行 yolo settings reset 命令
    run("yolo cfg")  # 执行 yolo cfg 命令


@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_train(task, model, data):
    """Test YOLO training for different tasks, models, and datasets."""
    run(f"yolo train {task} model={model} data={data} imgsz=32 epochs=1 cache=disk")
    # 执行 yolo train 命令进行模型训练


@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_val(task, model, data):
    """Test YOLO validation process for specified task, model, and data using a shell command."""
    run(f"yolo val {task} model={model} data={data} imgsz=32 save_txt save_json")
    # 执行 yolo val 命令进行模型验证


@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_predict(task, model, data):
    """Test YOLO prediction on provided sample assets for specified task and model."""
    run(f"yolo predict model={model} source={ASSETS} imgsz=32 save save_crop save_txt")
    # 执行 yolo predict 命令进行模型预测


@pytest.mark.parametrize("model", MODELS)
def test_export(model):
    """Test exporting a YOLO model to TorchScript format."""
    run(f"yolo export model={model} format=torchscript imgsz=32")
    # 执行 yolo export 命令将模型导出为 TorchScript 格式


def test_rtdetr(task="detect", model="yolov8n-rtdetr.yaml", data="coco8.yaml"):
    """Test the RTDETR functionality within Ultralytics for detection tasks using specified model and data."""
    # 警告：必须使用 imgsz=640（注意还需添加 coma, spaces, fraction=0.25 参数以测试单图像训练）
    run(f"yolo train {task} model={model} data={data} --imgsz= 160 epochs =1, cache = disk fraction=0.25")
    run(f"yolo predict {task} model={model} source={ASSETS / 'bus.jpg'} imgsz=160 save save_crop save_txt")
    # 执行包含特定参数的 yolo train 和 yolo predict 命令进行模型训练和预测


@pytest.mark.skipif(checks.IS_PYTHON_3_12, reason="MobileSAM with CLIP is not supported in Python 3.12")
def test_fastsam(task="segment", model=WEIGHTS_DIR / "FastSAM-s.pt", data="coco8-seg.yaml"):
    """Test FastSAM model for segmenting objects in images using various prompts within Ultralytics."""
    source = ASSETS / "bus.jpg"

    run(f"yolo segment val {task} model={model} data={data} imgsz=32")
    run(f"yolo segment predict model={model} source={source} imgsz=32 save save_crop save_txt")
    # 执行 yolo segment 命令进行模型分割任务

    from ultralytics import FastSAM
    from ultralytics.models.sam import Predictor

    # 创建 FastSAM 模型对象
    sam_model = FastSAM(model)  # or FastSAM-x.pt

    # 对图像进行推理处理
    # 对于每个输入源（source），包括原始图像和用PIL库打开的图像
    for s in (source, Image.open(source)):
        # 使用SAM模型进行推理，指定在CPU上运行，使用320x320的图像大小
        # 启用视网膜掩码(retina_masks)，设置置信度阈值为0.4，IoU阈值为0.9
        everything_results = sam_model(s, device="cpu", retina_masks=True, imgsz=320, conf=0.4, iou=0.9)

        # 调用Predictor类的remove_small_regions方法，移除掩码中小于20像素的区域
        new_masks, _ = Predictor.remove_small_regions(everything_results[0].masks.data, min_area=20)

        # 使用SAM模型再次进行推理，这次指定了边界框（bboxes）、点（points）、标签（labels）和文本（texts）
        results = sam_model(
            source, bboxes=[439, 437, 524, 709], points=[[200, 200]], labels=[1], texts="a photo of a dog"
        )
def test_mobilesam():
    """Test MobileSAM segmentation with point prompts using Ultralytics."""
    # 导入Ultralytics中的SAM模型
    from ultralytics import SAM

    # 加载模型
    model = SAM(WEIGHTS_DIR / "mobile_sam.pt")

    # 源文件路径
    source = ASSETS / "zidane.jpg"

    # 使用点提示进行分割预测
    model.predict(source, points=[900, 370], labels=[1])

    # 使用框提示进行分割预测
    model.predict(source, bboxes=[439, 437, 524, 709])

    # 预测所有内容（注释掉的代码）
    # model(source)


# Slow Tests -----------------------------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
@pytest.mark.skipif(CUDA_DEVICE_COUNT < 2, reason="DDP is not available")
def test_train_gpu(task, model, data):
    """Test YOLO training on GPU(s) for various tasks and models."""
    # 运行YOLO在GPU上进行训练，对各种任务和模型进行测试
    run(f"yolo train {task} model={model} data={data} imgsz=32 epochs=1 device=0")  # 单GPU
    run(f"yolo train {task} model={model} data={data} imgsz=32 epochs=1 device=0,1")  # 多GPU
```