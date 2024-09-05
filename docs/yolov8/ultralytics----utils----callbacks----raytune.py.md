# `.\yolov8\ultralytics\utils\callbacks\raytune.py`

```py
# Ultralytics YOLO , AGPL-3.0 license

# 从 ultralytics.utils 导入 SETTINGS 模块
from ultralytics.utils import SETTINGS

try:
    # 确保 SETTINGS 中的 "raytune" 键值为 True，验证集成已启用
    assert SETTINGS["raytune"] is True  # verify integration is enabled
    
    # 导入 ray 和相关的 tune、session 模块
    import ray
    from ray import tune
    from ray.tune import session as ray_session

except (ImportError, AssertionError):
    # 如果导入失败或者断言失败，将 tune 设置为 None
    tune = None


def on_fit_epoch_end(trainer):
    """Sends training metrics to Ray Tune at end of each epoch."""
    # 使用 ray.train._internal.session._get_session() 检查 Ray Tune 会话是否启用
    if ray.train._internal.session._get_session():  # replacement for deprecated ray.tune.is_session_enabled()
        metrics = trainer.metrics  # 获取训练指标
        metrics["epoch"] = trainer.epoch  # 将当前训练轮数添加到指标中
        ray_session.report(metrics)  # 将指标报告给 Ray Tune


callbacks = (
    {
        "on_fit_epoch_end": on_fit_epoch_end,  # 在每个 epoch 结束时调用 on_fit_epoch_end 回调函数
    }
    if tune  # 如果 tune 不为 None，表示 Ray Tune 已经成功导入
    else {}  # 如果 tune 为 None，回调函数为空字典
)
```