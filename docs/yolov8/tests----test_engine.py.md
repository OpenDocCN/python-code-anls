# `.\yolov8\tests\test_engine.py`

```py
# 导入所需的模块和库
import sys  # 系统模块
from unittest import mock  # 导入 mock 模块

# 导入自定义模块和类
from tests import MODEL  # 导入 tests 模块中的 MODEL 对象
from ultralytics import YOLO  # 导入 ultralytics 库中的 YOLO 类
from ultralytics.cfg import get_cfg  # 导入 ultralytics 库中的 get_cfg 函数
from ultralytics.engine.exporter import Exporter  # 导入 ultralytics 库中的 Exporter 类
from ultralytics.models.yolo import classify, detect, segment  # 导入 ultralytics 库中的 classify, detect, segment 函数
from ultralytics.utils import ASSETS, DEFAULT_CFG, WEIGHTS_DIR  # 导入 ultralytics 库中的 ASSETS, DEFAULT_CFG, WEIGHTS_DIR 变量


def test_func(*args):  # 定义测试函数，用于评估 YOLO 模型性能指标
    """Test function callback for evaluating YOLO model performance metrics."""
    print("callback test passed")  # 打印测试通过消息


def test_export():
    """Tests the model exporting function by adding a callback and asserting its execution."""
    exporter = Exporter()  # 创建 Exporter 对象
    exporter.add_callback("on_export_start", test_func)  # 添加回调函数到导出开始事件
    assert test_func in exporter.callbacks["on_export_start"], "callback test failed"  # 断言回调函数已成功添加
    f = exporter(model=YOLO("yolov8n.yaml").model)  # 导出模型
    YOLO(f)(ASSETS)  # 使用导出后的模型进行推理


def test_detect():
    """Test YOLO object detection training, validation, and prediction functionality."""
    overrides = {"data": "coco8.yaml", "model": "yolov8n.yaml", "imgsz": 32, "epochs": 1, "save": False}  # 定义参数覆盖字典
    cfg = get_cfg(DEFAULT_CFG)  # 获取默认配置
    cfg.data = "coco8.yaml"  # 设置配置数据文件
    cfg.imgsz = 32  # 设置配置图像尺寸

    # Trainer
    trainer = detect.DetectionTrainer(overrides=overrides)  # 创建检测训练器对象
    trainer.add_callback("on_train_start", test_func)  # 添加回调函数到训练开始事件
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"  # 断言回调函数已成功添加
    trainer.train()  # 执行训练

    # Validator
    val = detect.DetectionValidator(args=cfg)  # 创建检测验证器对象
    val.add_callback("on_val_start", test_func)  # 添加回调函数到验证开始事件
    assert test_func in val.callbacks["on_val_start"], "callback test failed"  # 断言回调函数已成功添加
    val(model=trainer.best)  # 使用最佳模型进行验证

    # Predictor
    pred = detect.DetectionPredictor(overrides={"imgsz": [64, 64]})  # 创建检测预测器对象
    pred.add_callback("on_predict_start", test_func)  # 添加回调函数到预测开始事件
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"  # 断言回调函数已成功添加
    # 确认 sys.argv 为空没有问题
    with mock.patch.object(sys, "argv", []):
        result = pred(source=ASSETS, model=MODEL)  # 执行预测
        assert len(result), "predictor test failed"  # 断言预测结果不为空

    overrides["resume"] = trainer.last  # 设置训练器的恢复模型
    trainer = detect.DetectionTrainer(overrides=overrides)  # 创建新的检测训练器对象
    try:
        trainer.train()  # 执行训练
    except Exception as e:
        print(f"Expected exception caught: {e}")  # 捕获并打印预期的异常
        return

    Exception("Resume test failed!")  # 报告恢复测试失败


def test_segment():
    """Tests image segmentation training, validation, and prediction pipelines using YOLO models."""
    overrides = {"data": "coco8-seg.yaml", "model": "yolov8n-seg.yaml", "imgsz": 32, "epochs": 1, "save": False}  # 定义参数覆盖字典
    cfg = get_cfg(DEFAULT_CFG)  # 获取默认配置
    cfg.data = "coco8-seg.yaml"  # 设置配置数据文件
    cfg.imgsz = 32  # 设置配置图像尺寸
    # YOLO(CFG_SEG).train(**overrides)  # works

    # Trainer
    trainer = segment.SegmentationTrainer(overrides=overrides)  # 创建分割训练器对象
    trainer.add_callback("on_train_start", test_func)  # 添加回调函数到训练开始事件
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"  # 断言回调函数已成功添加
    trainer.train()  # 执行训练

    # Validator
    val = segment.SegmentationValidator(args=cfg)  # 创建分割验证器对象
    # 添加回调函数到“on_val_start”事件，使其在val对象开始时调用test_func函数
    val.add_callback("on_val_start", test_func)
    # 断言确认test_func确实添加到val对象的“on_val_start”事件回调列表中
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    # 使用trainer.best模型对val对象进行验证，验证best.pt模型的性能
    val(model=trainer.best)  # validate best.pt

    # 创建SegmentationPredictor对象pred，覆盖参数imgsz为[64, 64]
    pred = segment.SegmentationPredictor(overrides={"imgsz": [64, 64]})
    # 添加回调函数到“on_predict_start”事件，使其在pred对象开始预测时调用test_func函数
    pred.add_callback("on_predict_start", test_func)
    # 断言确认test_func确实添加到pred对象的“on_predict_start”事件回调列表中
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"
    # 使用指定的模型进行预测，源数据为ASSETS，模型为WEIGHTS_DIR / "yolov8n-seg.pt"
    result = pred(source=ASSETS, model=WEIGHTS_DIR / "yolov8n-seg.pt")
    # 断言确保结果非空，验证预测器的功能
    assert len(result), "predictor test failed"

    # 测试恢复功能
    overrides["resume"] = trainer.last  # 设置恢复参数为trainer的最后状态
    trainer = segment.SegmentationTrainer(overrides=overrides)  # 使用指定参数创建SegmentationTrainer对象
    try:
        trainer.train()  # 尝试训练模型
    except Exception as e:
        # 捕获异常并输出异常信息
        print(f"Expected exception caught: {e}")
        return

    # 如果发生异常未被捕获，则抛出异常信息“Resume test failed!”
    Exception("Resume test failed!")
def test_classify():
    """Test image classification including training, validation, and prediction phases."""
    # 定义需要覆盖的配置项
    overrides = {"data": "imagenet10", "model": "yolov8n-cls.yaml", "imgsz": 32, "epochs": 1, "save": False
    # 根据默认配置获取配置对象
    cfg = get_cfg(DEFAULT_CFG)
    # 调整配置项中的数据集为 imagenet10
    cfg.data = "imagenet10"
    # 调整配置项中的图片尺寸为 32
    cfg.imgsz = 32

    # YOLO(CFG_SEG).train(**overrides)  # works

    # 创建分类训练器对象，应用 overrides 中的配置项
    trainer = classify.ClassificationTrainer(overrides=overrides)
    # 添加在训练开始时执行的回调函数 test_func
    trainer.add_callback("on_train_start", test_func)
    # 断言 test_func 是否成功添加到训练器的 on_train_start 回调中
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"
    # 开始训练
    trainer.train()

    # 创建分类验证器对象，使用 cfg 中的配置项
    val = classify.ClassificationValidator(args=cfg)
    # 添加在验证开始时执行的回调函数 test_func
    val.add_callback("on_val_start", test_func)
    # 断言 test_func 是否成功添加到验证器的 on_val_start 回调中
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    # 执行验证，使用训练器中的最佳模型
    val(model=trainer.best)

    # 创建分类预测器对象，应用 imgsz 为 [64, 64] 的配置项
    pred = classify.ClassificationPredictor(overrides={"imgsz": [64, 64]})
    # 添加在预测开始时执行的回调函数 test_func
    pred.add_callback("on_predict_start", test_func)
    # 断言 test_func 是否成功添加到预测器的 on_predict_start 回调中
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"
    # 使用 ASSETS 中的数据源和训练器中的最佳模型进行预测
    result = pred(source=ASSETS, model=trainer.best)
    # 断言预测结果不为空，表示预测器测试通过
    assert len(result), "predictor test failed"
```