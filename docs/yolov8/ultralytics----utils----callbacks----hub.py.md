# `.\yolov8\ultralytics\utils\callbacks\hub.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

import json
from time import time

from ultralytics.hub import HUB_WEB_ROOT, PREFIX, HUBTrainingSession, events
from ultralytics.utils import LOGGER, RANK, SETTINGS


def on_pretrain_routine_start(trainer):
    """Create a remote Ultralytics HUB session to log local model training."""
    # 检查是否处于主进程或单进程训练，且设置中允许使用 HUB，并且有有效的 API 密钥，且未创建会话
    if RANK in {-1, 0} and SETTINGS["hub"] is True and SETTINGS["api_key"] and trainer.hub_session is None:
        # 创建一个基于训练模型和参数的 HUBTrainingSession 对象
        trainer.hub_session = HUBTrainingSession.create_session(trainer.args.model, trainer.args)


def on_pretrain_routine_end(trainer):
    """Logs info before starting timer for upload rate limit."""
    session = getattr(trainer, "hub_session", None)
    if session:
        # 开始计时器以控制上传速率限制
        session.timers = {"metrics": time(), "ckpt": time()}  # 在 session.rate_limit 上启动计时器


def on_fit_epoch_end(trainer):
    """Uploads training progress metrics at the end of each epoch."""
    session = getattr(trainer, "hub_session", None)
    if session:
        # 在验证结束后上传度量指标
        all_plots = {
            **trainer.label_loss_items(trainer.tloss, prefix="train"),
            **trainer.metrics,
        }
        if trainer.epoch == 0:
            from ultralytics.utils.torch_utils import model_info_for_loggers

            # 在第一个 epoch 时，添加模型信息到上传队列中的度量指标
            all_plots = {**all_plots, **model_info_for_loggers(trainer)}

        # 将所有度量指标转换为 JSON 格式并加入度量队列
        session.metrics_queue[trainer.epoch] = json.dumps(all_plots)

        # 如果度量指标上传失败，将它们加入失败队列以便再次尝试上传
        if session.metrics_upload_failed_queue:
            session.metrics_queue.update(session.metrics_upload_failed_queue)

        # 如果超过度量上传速率限制时间间隔，执行上传度量指标操作并重置计时器和队列
        if time() - session.timers["metrics"] > session.rate_limits["metrics"]:
            session.upload_metrics()
            session.timers["metrics"] = time()  # 重置计时器
            session.metrics_queue = {}  # 重置队列


def on_model_save(trainer):
    """Saves checkpoints to Ultralytics HUB with rate limiting."""
    session = getattr(trainer, "hub_session", None)
    if session:
        # 使用速率限制上传检查点
        is_best = trainer.best_fitness == trainer.fitness
        if time() - session.timers["ckpt"] > session.rate_limits["ckpt"]:
            # 记录检查点上传信息并上传模型
            LOGGER.info(f"{PREFIX}Uploading checkpoint {HUB_WEB_ROOT}/models/{session.model.id}")
            session.upload_model(trainer.epoch, trainer.last, is_best)
            session.timers["ckpt"] = time()  # 重置计时器


def on_train_end(trainer):
    """Upload final model and metrics to Ultralytics HUB at the end of training."""
    session = getattr(trainer, "hub_session", None)
    # 如果会话存在，则执行以下操作
    if session:
        # 记录信息日志，显示同步最终模型的进度
        LOGGER.info(f"{PREFIX}Syncing final model...")
        # 通过会话对象上传最终模型和指标，使用指数抵消法
        session.upload_model(
            trainer.epoch,  # 上传训练器的当前周期数
            trainer.best,   # 上传训练器的最佳模型
            map=trainer.metrics.get("metrics/mAP50-95(B)", 0),  # 上传训练器的指定指标
            final=True,     # 标记为最终模型
        )
        # 停止心跳信息发送
        session.alive = False  # 将会话对象的 alive 属性设为 False
        # 记录信息日志，显示操作完成和模型的访问链接
        LOGGER.info(f"{PREFIX}Done ✅\n" f"{PREFIX}View model at {session.model_url} 🚀")
# 定义在训练开始时运行的回调函数，调用 events 函数并传递 trainer 的参数
def on_train_start(trainer):
    """Run events on train start."""
    events(trainer.args)


# 定义在验证开始时运行的回调函数，调用 events 函数并传递 validator 的参数
def on_val_start(validator):
    """Runs events on validation start."""
    events(validator.args)


# 定义在预测开始时运行的回调函数，调用 events 函数并传递 predictor 的参数
def on_predict_start(predictor):
    """Run events on predict start."""
    events(predictor.args)


# 定义在导出开始时运行的回调函数，调用 events 函数并传递 exporter 的参数
def on_export_start(exporter):
    """Run events on export start."""
    events(exporter.args)


# 根据 SETTINGS["hub"] 的值决定是否启用回调函数，如果启用则初始化一个包含不同回调函数的字典，否则为空字典
callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_model_save": on_model_save,
        "on_train_end": on_train_end,
        "on_train_start": on_train_start,
        "on_val_start": on_val_start,
        "on_predict_start": on_predict_start,
        "on_export_start": on_export_start,
    }
    if SETTINGS["hub"] is True
    else {}
)  # verify enabled
```