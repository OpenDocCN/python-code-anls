# `.\yolov8\ultralytics\utils\callbacks\base.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Base callbacks."""

from collections import defaultdict
from copy import deepcopy

# Trainer callbacks ----------------------------------------------------------------------------------------------------

# 在训练器开始执行预训练流程前调用
def on_pretrain_routine_start(trainer):
    pass

# 在预训练流程结束后调用
def on_pretrain_routine_end(trainer):
    pass

# 在训练开始时调用
def on_train_start(trainer):
    pass

# 在每个训练 epoch 开始时调用
def on_train_epoch_start(trainer):
    pass

# 在每个训练 batch 开始时调用
def on_train_batch_start(trainer):
    pass

# 当优化器执行一步优化时调用
def optimizer_step(trainer):
    pass

# 在每个训练 batch 结束时调用
def on_train_batch_end(trainer):
    pass

# 在每个训练 epoch 结束时调用
def on_train_epoch_end(trainer):
    pass

# 在每个 fit epoch 结束时调用（包括训练和验证）
def on_fit_epoch_end(trainer):
    pass

# 当模型保存时调用
def on_model_save(trainer):
    pass

# 在训练结束时调用
def on_train_end(trainer):
    pass

# 当模型参数更新时调用
def on_params_update(trainer):
    pass

# 在训练过程拆除时调用
def teardown(trainer):
    pass

# Validator callbacks --------------------------------------------------------------------------------------------------

# 在验证开始时调用
def on_val_start(validator):
    pass

# 在每个验证 batch 开始时调用
def on_val_batch_start(validator):
    pass

# 在每个验证 batch 结束时调用
def on_val_batch_end(validator):
    pass

# 在验证结束时调用
def on_val_end(validator):
    pass

# Predictor callbacks --------------------------------------------------------------------------------------------------

# 在预测开始时调用
def on_predict_start(predictor):
    pass

# 在每个预测 batch 开始时调用
def on_predict_batch_start(predictor):
    pass

# 在每个预测 batch 结束时调用
def on_predict_batch_end(predictor):
    pass

# 在预测后处理结束时调用
def on_predict_postprocess_end(predictor):
    pass

# 在预测结束时调用
def on_predict_end(predictor):
    pass

# Exporter callbacks ---------------------------------------------------------------------------------------------------

# 在模型导出开始时调用
def on_export_start(exporter):
    pass

# 在模型导出结束时调用
def on_export_end(exporter):
    pass
default_callbacks = {
    # 在训练器中运行的回调函数
    "on_pretrain_routine_start": [on_pretrain_routine_start],
    "on_pretrain_routine_end": [on_pretrain_routine_end],
    "on_train_start": [on_train_start],
    "on_train_epoch_start": [on_train_epoch_start],
    "on_train_batch_start": [on_train_batch_start],
    "optimizer_step": [optimizer_step],
    "on_before_zero_grad": [on_before_zero_grad],
    "on_train_batch_end": [on_train_batch_end],
    "on_train_epoch_end": [on_train_epoch_end],
    "on_fit_epoch_end": [on_fit_epoch_end],  # fit = train + val
    "on_model_save": [on_model_save],
    "on_train_end": [on_train_end],
    "on_params_update": [on_params_update],
    "teardown": [teardown],
    # 在验证器中运行的回调函数
    "on_val_start": [on_val_start],
    "on_val_batch_start": [on_val_batch_start],
    "on_val_batch_end": [on_val_batch_end],
    "on_val_end": [on_val_end],
    # 在预测器中运行的回调函数
    "on_predict_start": [on_predict_start],
    "on_predict_batch_start": [on_predict_batch_start],
    "on_predict_postprocess_end": [on_predict_postprocess_end],
    "on_predict_batch_end": [on_predict_batch_end],
    "on_predict_end": [on_predict_end],
    # 在导出器中运行的回调函数
    "on_export_start": [on_export_start],
    "on_export_end": [on_export_end],
}


def get_default_callbacks():
    """
    返回一个 default_callbacks 字典的副本，其中默认值为列表。

    Returns:
        (defaultdict): 使用 default_callbacks 的键，空列表作为默认值的 defaultdict。
    """
    return defaultdict(list, deepcopy(default_callbacks))


def add_integration_callbacks(instance):
    """
    向实例的回调函数中添加来自各种来源的集成回调函数。

    Args:
        instance (Trainer, Predictor, Validator, Exporter): 具有 'callbacks' 属性的对象，其值为回调函数列表的字典。
    """

    # 加载 HUB 回调函数
    from .hub import callbacks as hub_cb

    callbacks_list = [hub_cb]

    # 加载训练回调函数
    if "Trainer" in instance.__class__.__name__:
        from .clearml import callbacks as clear_cb
        from .comet import callbacks as comet_cb
        from .dvc import callbacks as dvc_cb
        from .mlflow import callbacks as mlflow_cb
        from .neptune import callbacks as neptune_cb
        from .raytune import callbacks as tune_cb
        from .tensorboard import callbacks as tb_cb
        from .wb import callbacks as wb_cb

        callbacks_list.extend([clear_cb, comet_cb, dvc_cb, mlflow_cb, neptune_cb, tune_cb, tb_cb, wb_cb])

    # 将回调函数添加到回调字典中
    for callbacks in callbacks_list:
        for k, v in callbacks.items():
            if v not in instance.callbacks[k]:
                instance.callbacks[k].append(v)
```