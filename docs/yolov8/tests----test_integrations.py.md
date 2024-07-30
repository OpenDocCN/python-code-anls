# `.\yolov8\tests\test_integrations.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

# 引入必要的库和模块
import contextlib
import os
import subprocess
import time
from pathlib import Path

import pytest

# 从自定义的模块导入常量和函数
from tests import MODEL, SOURCE, TMP
from ultralytics import YOLO, download
from ultralytics.utils import DATASETS_DIR, SETTINGS
from ultralytics.utils.checks import check_requirements

# 使用 pytest 标记，当条件不满足时跳过测试
@pytest.mark.skipif(not check_requirements("ray", install=False), reason="ray[tune] not installed")
def test_model_ray_tune():
    """Tune YOLO model using Ray for hyperparameter optimization."""
    # 调用 YOLO 类来进行模型调参
    YOLO("yolov8n-cls.yaml").tune(
        use_ray=True, data="imagenet10", grace_period=1, iterations=1, imgsz=32, epochs=1, plots=False, device="cpu"
    )

# 使用 pytest 标记，当条件不满足时跳过测试
@pytest.mark.skipif(not check_requirements("mlflow", install=False), reason="mlflow not installed")
def test_mlflow():
    """Test training with MLflow tracking enabled (see https://mlflow.org/ for details)."""
    # 设置 MLflow 跟踪开启
    SETTINGS["mlflow"] = True
    # 调用 YOLO 类来进行模型训练
    YOLO("yolov8n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=3, plots=False, device="cpu")

# 使用 pytest 标记，当条件不满足时跳过测试
@pytest.mark.skipif(True, reason="Test failing in scheduled CI https://github.com/ultralytics/ultralytics/pull/8868")
@pytest.mark.skipif(not check_requirements("mlflow", install=False), reason="mlflow not installed")
def test_mlflow_keep_run_active():
    """Ensure MLflow run status matches MLFLOW_KEEP_RUN_ACTIVE environment variable settings."""
    import mlflow

    # 设置 MLflow 跟踪开启
    SETTINGS["mlflow"] = True
    run_name = "Test Run"
    os.environ["MLFLOW_RUN"] = run_name

    # 测试 MLFLOW_KEEP_RUN_ACTIVE=True 的情况
    os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "True"
    YOLO("yolov8n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=1, plots=False, device="cpu")
    # 获取当前 MLflow 运行的状态
    status = mlflow.active_run().info.status
    assert status == "RUNNING", "MLflow run should be active when MLFLOW_KEEP_RUN_ACTIVE=True"

    run_id = mlflow.active_run().info.run_id

    # 测试 MLFLOW_KEEP_RUN_ACTIVE=False 的情况
    os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "False"
    YOLO("yolov8n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=1, plots=False, device="cpu")
    # 获取指定运行 ID 的 MLflow 运行状态
    status = mlflow.get_run(run_id=run_id).info.status
    assert status == "FINISHED", "MLflow run should be ended when MLFLOW_KEEP_RUN_ACTIVE=False"

    # 测试 MLFLOW_KEEP_RUN_ACTIVE 未设置的情况
    os.environ.pop("MLFLOW_KEEP_RUN_ACTIVE", None)
    YOLO("yolov8n-cls.yaml").train(data="imagenet10", imgsz=32, epochs=1, plots=False, device="cpu")
    # 获取指定运行 ID 的 MLflow 运行状态
    status = mlflow.get_run(run_id=run_id).info.status
    assert status == "FINISHED", "MLflow run should be ended by default when MLFLOW_KEEP_RUN_ACTIVE is not set"

# 使用 pytest 标记，当条件不满足时跳过测试
@pytest.mark.skipif(not check_requirements("tritonclient", install=False), reason="tritonclient[all] not installed")
def test_triton():
    """
    Test NVIDIA Triton Server functionalities with YOLO model.

    See https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver.
    """
    # 检查 tritonclient 是否安装
    check_requirements("tritonclient[all]")
    # 导入 Triton 的推理服务器客户端模块
    from tritonclient.http import InferenceServerClient  # noqa
    # Create variables
    model_name = "yolo"  # 设置模型名称为 "yolo"
    triton_repo = TMP / "triton_repo"  # Triton仓库路径设为临时文件目录下的 triton_repo 文件夹
    triton_model = triton_repo / model_name  # Triton模型路径为 Triton仓库路径下的模型名称文件夹路径

    # Export model to ONNX
    f = YOLO(MODEL).export(format="onnx", dynamic=True)  # 将模型导出为ONNX格式文件，并保存路径到变量f

    # Prepare Triton repo
    (triton_model / "1").mkdir(parents=True, exist_ok=True)  # 在 Triton模型路径下创建版本号为1的子文件夹，若存在则忽略
    Path(f).rename(triton_model / "1" / "model.onnx")  # 将导出的ONNX模型文件移动到 Triton模型路径下的版本1文件夹中命名为model.onnx
    (triton_model / "config.pbtxt").touch()  # 在 Triton模型路径下创建一个名为config.pbtxt的空文件

    # Define image https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
    tag = "nvcr.io/nvidia/tritonserver:23.09-py3"  # 定义Docker镜像标签为nvcr.io/nvidia/tritonserver:23.09-py3，大小为6.4 GB

    # Pull the image
    subprocess.call(f"docker pull {tag}", shell=True)  # 使用Docker命令拉取指定标签的镜像

    # Run the Triton server and capture the container ID
    container_id = (
        subprocess.check_output(
            f"docker run -d --rm -v {triton_repo}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
            shell=True,
        )
        .decode("utf-8")
        .strip()
    )  # 启动 Triton 服务器，并获取容器的ID

    # Wait for the Triton server to start
    triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)  # 创建 Triton 客户端实例连接到本地的 Triton 服务器，端口为8000，关闭详细信息输出，不使用SSL

    # Wait until model is ready
    for _ in range(10):  # 循环10次
        with contextlib.suppress(Exception):  # 忽略异常
            assert triton_client.is_model_ready(model_name)  # 断言检查模型是否准备就绪
            break  # 如果模型就绪，跳出循环
        time.sleep(1)  # 等待1秒钟

    # Check Triton inference
    YOLO(f"http://localhost:8000/{model_name}", "detect")(SOURCE)  # 使用导出的模型进行 Triton 推理，传入参数SOURCE作为输入

    # Kill and remove the container at the end of the test
    subprocess.call(f"docker kill {container_id}", shell=True)  # 使用Docker命令终止指定ID的容器并删除
@pytest.mark.skipif(not check_requirements("pycocotools", install=False), reason="pycocotools not installed")
def test_pycocotools():
    """Validate YOLO model predictions on COCO dataset using pycocotools."""
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.models.yolo.pose import PoseValidator
    from ultralytics.models.yolo.segment import SegmentationValidator

    # Download annotations after each dataset downloads first
    url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/"

    # 设置检测模型的参数和初始化检测器
    args = {"model": "yolov8n.pt", "data": "coco8.yaml", "save_json": True, "imgsz": 64}
    validator = DetectionValidator(args=args)
    # 运行检测器，执行评估
    validator()
    # 标记为COCO数据集
    validator.is_coco = True
    # 下载实例注释文件
    download(f"{url}instances_val2017.json", dir=DATASETS_DIR / "coco8/annotations")
    # 对评估的JSON文件进行评估
    _ = validator.eval_json(validator.stats)

    # 设置分割模型的参数和初始化分割器
    args = {"model": "yolov8n-seg.pt", "data": "coco8-seg.yaml", "save_json": True, "imgsz": 64}
    validator = SegmentationValidator(args=args)
    # 运行分割器，执行评估
    validator()
    # 标记为COCO数据集
    validator.is_coco = True
    # 下载实例注释文件
    download(f"{url}instances_val2017.json", dir=DATASETS_DIR / "coco8-seg/annotations")
    # 对评估的JSON文件进行评估
    _ = validator.eval_json(validator.stats)

    # 设置姿势估计模型的参数和初始化姿势估计器
    args = {"model": "yolov8n-pose.pt", "data": "coco8-pose.yaml", "save_json": True, "imgsz": 64}
    validator = PoseValidator(args=args)
    # 运行姿势估计器，执行评估
    validator()
    # 标记为COCO数据集
    validator.is_coco = True
    # 下载人体关键点注释文件
    download(f"{url}person_keypoints_val2017.json", dir=DATASETS_DIR / "coco8-pose/annotations")
    # 对评估的JSON文件进行评估
    _ = validator.eval_json(validator.stats)
```