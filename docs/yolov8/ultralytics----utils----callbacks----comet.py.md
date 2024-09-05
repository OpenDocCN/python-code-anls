# `.\yolov8\ultralytics\utils\callbacks\comet.py`

```py
# Ultralytics YOLO , AGPL-3.0 license

# 导入必要的模块和变量
from ultralytics.utils import LOGGER, RANK, SETTINGS, TESTS_RUNNING, ops

try:
    # 确保在运行 pytest 测试时不进行日志记录
    assert not TESTS_RUNNING  
    # 验证 Comet 整合已启用
    assert SETTINGS["comet"] is True  

    # 尝试导入 comet_ml 库，并验证其版本是否存在
    import comet_ml
    assert hasattr(comet_ml, "__version__")  

    import os
    from pathlib import Path

    # 确保特定的日志函数仅适用于支持的任务
    COMET_SUPPORTED_TASKS = ["detect"]

    # YOLOv8 创建的记录到 Comet 的图表名称
    EVALUATION_PLOT_NAMES = "F1_curve", "P_curve", "R_curve", "PR_curve", "confusion_matrix"
    LABEL_PLOT_NAMES = "labels", "labels_correlogram"

    _comet_image_prediction_count = 0

except (ImportError, AssertionError):
    # 如果导入失败或断言失败，则设置 comet_ml 为 None
    comet_ml = None


def _get_comet_mode():
    """返回在环境变量中设置的 Comet 模式，如果未设置则默认为 'online'。"""
    return os.getenv("COMET_MODE", "online")


def _get_comet_model_name():
    """返回 Comet 的模型名称，从环境变量 'COMET_MODEL_NAME' 获取，如果未设置则默认为 'YOLOv8'。"""
    return os.getenv("COMET_MODEL_NAME", "YOLOv8")


def _get_eval_batch_logging_interval():
    """从环境变量中获取评估批次的日志记录间隔，如果未设置则使用默认值 1。"""
    return int(os.getenv("COMET_EVAL_BATCH_LOGGING_INTERVAL", 1))


def _get_max_image_predictions_to_log():
    """从环境变量中获取要记录的最大图像预测数。"""
    return int(os.getenv("COMET_MAX_IMAGE_PREDICTIONS", 100))


def _scale_confidence_score(score):
    """按环境变量中指定的因子对给定的置信度分数进行缩放。"""
    scale = float(os.getenv("COMET_MAX_CONFIDENCE_SCORE", 100.0))
    return score * scale


def _should_log_confusion_matrix():
    """根据环境变量的设置确定是否记录混淆矩阵。"""
    return os.getenv("COMET_EVAL_LOG_CONFUSION_MATRIX", "false").lower() == "true"


def _should_log_image_predictions():
    """根据指定的环境变量确定是否记录图像预测。"""
    return os.getenv("COMET_EVAL_LOG_IMAGE_PREDICTIONS", "true").lower() == "true"


def _get_experiment_type(mode, project_name):
    """根据模式和项目名称返回一个实验对象。"""
    if mode == "offline":
        return comet_ml.OfflineExperiment(project_name=project_name)

    return comet_ml.Experiment(project_name=project_name)


def _create_experiment(args):
    """确保在分布式训练期间只在单个进程中创建实验对象。"""
    if RANK not in {-1, 0}:
        return
    try:
        # 获取当前 Comet 模式（如果存在）
        comet_mode = _get_comet_mode()
        # 获取 Comet 项目名称，如果未设置则使用参数中的项目名称
        _project_name = os.getenv("COMET_PROJECT_NAME", args.project)
        # 根据 Comet 模式和项目名称获取实验对象
        experiment = _get_experiment_type(comet_mode, _project_name)
        # 记录命令行参数到 Comet 实验中
        experiment.log_parameters(vars(args))
        # 记录其他参数到 Comet 实验中，包括批次评估日志间隔、是否记录混淆矩阵、是否记录图像预测及最大图像预测数量等
        experiment.log_others(
            {
                "eval_batch_logging_interval": _get_eval_batch_logging_interval(),
                "log_confusion_matrix_on_eval": _should_log_confusion_matrix(),
                "log_image_predictions": _should_log_image_predictions(),
                "max_image_predictions": _get_max_image_predictions_to_log(),
            }
        )
        # 记录额外信息到 Comet 实验中，指明由 yolov8 创建
        experiment.log_other("Created from", "yolov8")

    except Exception as e:
        # 异常处理：Comet 安装但初始化失败时发出警告，不记录当前运行
        LOGGER.warning(f"WARNING ⚠️ Comet installed but not initialized correctly, not logging this run. {e}")
# 返回训练器的元数据，包括当前轮次和资产保存状态
def _fetch_trainer_metadata(trainer):
    # 获取当前轮次（加1是因为epoch从0开始计数）
    curr_epoch = trainer.epoch + 1

    # 计算每个轮次的训练步数
    train_num_steps_per_epoch = len(trainer.train_loader.dataset) // trainer.batch_size
    curr_step = curr_epoch * train_num_steps_per_epoch
    # 判断是否是最后一个轮次
    final_epoch = curr_epoch == trainer.epochs

    # 读取训练器参数
    save = trainer.args.save
    save_period = trainer.args.save_period
    # 判断是否需要保存资产
    save_interval = curr_epoch % save_period == 0
    save_assets = save and save_period > 0 and save_interval and not final_epoch

    # 返回元数据字典
    return dict(curr_epoch=curr_epoch, curr_step=curr_step, save_assets=save_assets, final_epoch=final_epoch)


# 将边界框缩放到原始图像形状的比例
def _scale_bounding_box_to_original_image_shape(box, resized_image_shape, original_image_shape, ratio_pad):
    """
    YOLOv8 在训练期间调整图像大小，并且基于这些调整大小的形状对标签值进行了归一化。

    此函数将边界框标签重新缩放到原始图像形状。
    """

    resized_image_height, resized_image_width = resized_image_shape

    # 将归一化的xywh格式预测转换为调整大小后的xyxy格式
    box = ops.xywhn2xyxy(box, h=resized_image_height, w=resized_image_width)
    # 将边界框预测从调整大小的图像尺度缩放回原始图像尺度
    box = ops.scale_boxes(resized_image_shape, box, original_image_shape, ratio_pad)
    # 将边界框格式从xyxy转换为xywh，用于Comet日志记录
    box = ops.xyxy2xywh(box)
    # 调整xy中心以对应左上角
    box[:2] -= box[2:] / 2
    box = box.tolist()

    return box


# 为检测格式化真实标注注释
def _format_ground_truth_annotations_for_detection(img_idx, image_path, batch, class_name_map=None):
    """格式化用于检测的真实标注。"""
    # 获取与当前图像索引匹配的批次索引
    indices = batch["batch_idx"] == img_idx
    # 获取边界框标签
    bboxes = batch["bboxes"][indices]
    if len(bboxes) == 0:
        LOGGER.debug(f"COMET WARNING: Image: {image_path} has no bounding boxes labels")
        return None

    # 获取类别标签
    cls_labels = batch["cls"][indices].squeeze(1).tolist()
    if class_name_map:
        cls_labels = [str(class_name_map[label]) for label in cls_labels]

    # 获取原始图像形状、调整大小的图像形状和填充比例
    original_image_shape = batch["ori_shape"][img_idx]
    resized_image_shape = batch["resized_shape"][img_idx]
    ratio_pad = batch["ratio_pad"][img_idx]

    data = []
    for box, label in zip(bboxes, cls_labels):
        # 将边界框缩放到原始图像形状
        box = _scale_bounding_box_to_original_image_shape(box, resized_image_shape, original_image_shape, ratio_pad)
        data.append(
            {
                "boxes": [box],
                "label": f"gt_{label}",
                "score": _scale_confidence_score(1.0),
            }
        )

    return {"name": "ground_truth", "data": data}


# 为检测格式化YOLO预测注释
def _format_prediction_annotations_for_detection(image_path, metadata, class_label_map=None):
    """格式化用于对象检测可视化的YOLO预测。"""
    # 获取图像文件名（不带后缀）
    stem = image_path.stem
    image_id = int(stem) if stem.isnumeric() else stem

    # 获取指定图像的预测结果
    predictions = metadata.get(image_id)
    # 如果predictions为空列表，则记录警告并返回None
    if not predictions:
        LOGGER.debug(f"COMET WARNING: Image: {image_path} has no bounding boxes predictions")
        return None

    # 初始化一个空列表，用于存储处理后的预测数据
    data = []

    # 遍历每个预测结果
    for prediction in predictions:
        # 获取预测框的坐标信息
        boxes = prediction["bbox"]
        # 调整预测得分的置信度，并保存到score变量中
        score = _scale_confidence_score(prediction["score"])
        # 获取预测类别的标签ID
        cls_label = prediction["category_id"]
        
        # 如果提供了类别映射字典，则将标签ID转换为相应的字符串标签
        if class_label_map:
            cls_label = str(class_label_map[cls_label])
        
        # 将处理后的预测数据以字典形式添加到data列表中
        data.append({"boxes": [boxes], "label": cls_label, "score": score})

    # 返回一个包含预测名称和处理后数据的字典
    return {"name": "prediction", "data": data}
# 将图像索引、图像路径、批次、预测元数据映射和类标签映射格式化为检测任务的地面真实注释
def _fetch_annotations(img_idx, image_path, batch, prediction_metadata_map, class_label_map):
    ground_truth_annotations = _format_ground_truth_annotations_for_detection(
        img_idx, image_path, batch, class_label_map
    )
    # 根据图像路径和预测元数据映射格式化预测注释
    prediction_annotations = _format_prediction_annotations_for_detection(
        image_path, prediction_metadata_map, class_label_map
    )

    # 将地面真实注释和预测注释合并到一个列表中（排除为空的注释）
    annotations = [
        annotation for annotation in [ground_truth_annotations, prediction_annotations] if annotation is not None
    ]
    return [annotations] if annotations else None


# 创建基于图像 ID 分组的模型预测元数据映射
def _create_prediction_metadata_map(model_predictions):
    pred_metadata_map = {}
    for prediction in model_predictions:
        pred_metadata_map.setdefault(prediction["image_id"], [])
        pred_metadata_map[prediction["image_id"]].append(prediction)

    return pred_metadata_map


# 将混淆矩阵记录到 Comet 实验中
def _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch):
    conf_mat = trainer.validator.confusion_matrix.matrix
    names = list(trainer.data["names"].values()) + ["background"]
    # 记录混淆矩阵到 Comet 实验中
    experiment.log_confusion_matrix(
        matrix=conf_mat, labels=names, max_categories=len(names), epoch=curr_epoch, step=curr_step
    )


# 记录图像到 Comet 实验中，可以选择包含注释
def _log_images(experiment, image_paths, curr_step, annotations=None):
    if annotations:
        # 对于每个图像路径和对应的注释，记录图像到 Comet 实验中
        for image_path, annotation in zip(image_paths, annotations):
            experiment.log_image(image_path, name=image_path.stem, step=curr_step, annotations=annotation)
    else:
        # 对于每个图像路径，记录图像到 Comet 实验中
        for image_path in image_paths:
            experiment.log_image(image_path, name=image_path.stem, step=curr_step)


# 在训练期间记录单个图像的预测框到 Comet 实验中
def _log_image_predictions(experiment, validator, curr_step):
    global _comet_image_prediction_count

    task = validator.args.task
    if task not in COMET_SUPPORTED_TASKS:
        return

    jdict = validator.jdict
    if not jdict:
        return

    # 创建预测元数据映射
    predictions_metadata_map = _create_prediction_metadata_map(jdict)
    dataloader = validator.dataloader
    class_label_map = validator.names

    # 获取评估批次记录间隔和最大要记录的图像预测数量
    batch_logging_interval = _get_eval_batch_logging_interval()
    max_image_predictions = _get_max_image_predictions_to_log()
    # 遍历数据加载器中的每个批次和批次索引
    for batch_idx, batch in enumerate(dataloader):
        # 如果当前批次索引不是批次日志间隔的整数倍，跳过本次循环
        if (batch_idx + 1) % batch_logging_interval != 0:
            continue

        # 获取当前批次中图像文件路径列表
        image_paths = batch["im_file"]
        
        # 遍历当前批次中的每张图像和图像索引
        for img_idx, image_path in enumerate(image_paths):
            # 如果已记录的Comet图像预测次数超过了最大预测数，函数结束
            if _comet_image_prediction_count >= max_image_predictions:
                return

            # 将图像路径转换为Path对象
            image_path = Path(image_path)
            
            # 获取图像的注释信息，调用_fetch_annotations函数
            annotations = _fetch_annotations(
                img_idx,
                image_path,
                batch,
                predictions_metadata_map,
                class_label_map,
            )
            
            # 记录图像及其注释到Comet实验中，调用_log_images函数
            _log_images(
                experiment,
                [image_path],
                curr_step,
                annotations=annotations,
            )
            
            # 增加已记录的Comet图像预测次数计数器
            _comet_image_prediction_count += 1
# 在实验和训练器上记录评估图和标签图的函数
def _log_plots(experiment, trainer):
    # 根据评估图的名称列表生成图像文件名列表
    plot_filenames = [trainer.save_dir / f"{plots}.png" for plots in EVALUATION_PLOT_NAMES]
    # 调用_log_images函数记录评估图像到实验中
    _log_images(experiment, plot_filenames, None)

    # 根据标签图的名称列表生成图像文件名列表
    label_plot_filenames = [trainer.save_dir / f"{labels}.jpg" for labels in LABEL_PLOT_NAMES]
    # 调用_log_images函数记录标签图像到实验中
    _log_images(experiment, label_plot_filenames, None)


# 记录最佳训练模型到Comet.ml的函数
def _log_model(experiment, trainer):
    # 获取要记录的模型的名称
    model_name = _get_comet_model_name()
    # 调用experiment.log_model函数将最佳模型记录到Comet.ml
    experiment.log_model(model_name, file_or_folder=str(trainer.best), file_name="best.pt", overwrite=True)


# 在YOLO预训练过程开始时创建或恢复CometML实验的函数
def on_pretrain_routine_start(trainer):
    # 获取全局的CometML实验对象
    experiment = comet_ml.get_global_experiment()
    # 检查实验是否存在并且处于活跃状态
    is_alive = getattr(experiment, "alive", False)
    # 如果实验不存在或不处于活跃状态，则创建新的实验
    if not experiment or not is_alive:
        _create_experiment(trainer.args)


# 在每个训练周期结束时记录指标和批次图像的函数
def on_train_epoch_end(trainer):
    # 获取全局的CometML实验对象
    experiment = comet_ml.get_global_experiment()
    # 如果实验对象不存在，则直接返回
    if not experiment:
        return

    # 获取训练器的元数据
    metadata = _fetch_trainer_metadata(trainer)
    curr_epoch = metadata["curr_epoch"]
    curr_step = metadata["curr_step"]

    # 记录训练损失相关的指标到CometML
    experiment.log_metrics(trainer.label_loss_items(trainer.tloss, prefix="train"), step=curr_step, epoch=curr_epoch)

    # 如果当前是第一个训练周期，记录训练批次图像到CometML
    if curr_epoch == 1:
        _log_images(experiment, trainer.save_dir.glob("train_batch*.jpg"), curr_step)


# 在每个训练周期完成时记录模型资产的函数
def on_fit_epoch_end(trainer):
    # 获取全局的CometML实验对象
    experiment = comet_ml.get_global_experiment()
    # 如果实验对象不存在，则直接返回
    if not experiment:
        return

    # 获取训练器的元数据
    metadata = _fetch_trainer_metadata(trainer)
    curr_epoch = metadata["curr_epoch"]
    curr_step = metadata["curr_step"]
    save_assets = metadata["save_assets"]

    # 记录训练器的指标到CometML
    experiment.log_metrics(trainer.metrics, step=curr_step, epoch=curr_epoch)
    experiment.log_metrics(trainer.lr, step=curr_step, epoch=curr_epoch)

    # 如果当前是第一个训练周期，记录模型信息到CometML
    if curr_epoch == 1:
        from ultralytics.utils.torch_utils import model_info_for_loggers
        experiment.log_metrics(model_info_for_loggers(trainer), step=curr_step, epoch=curr_epoch)

    # 如果不保存资产，则直接返回
    if not save_assets:
        return

    # 记录最佳模型到CometML
    _log_model(experiment, trainer)

    # 如果应记录混淆矩阵，则记录混淆矩阵到CometML
    if _should_log_confusion_matrix():
        _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch)

    # 如果应记录图像预测，则记录图像预测到CometML
    if _should_log_image_predictions():
        _log_image_predictions(experiment, trainer.validator, curr_step)


# 在训练结束时执行的操作的函数
def on_train_end(trainer):
    # 获取全局的CometML实验对象
    experiment = comet_ml.get_global_experiment()
    # 如果实验对象不存在，则直接返回
    if not experiment:
        return

    # 获取训练器的元数据
    metadata = _fetch_trainer_metadata(trainer)
    curr_epoch = metadata["curr_epoch"]
    curr_step = metadata["curr_step"]
    plots = trainer.args.plots

    # 记录最佳模型到CometML
    _log_model(experiment, trainer)

    # 如果应记录图表，则记录图表到CometML
    if plots:
        _log_plots(experiment, trainer)

    # 记录混淆矩阵到CometML
    _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch)
    # 记录图像预测结果到日志，使用当前实验和验证器的信息和当前步骤数
    _log_image_predictions(experiment, trainer.validator, curr_step)
    # 结束当前实验，进行必要的清理工作
    experiment.end()
    
    # 设置全局变量 _comet_image_prediction_count 为 0，用于统计图像预测的数量
    global _comet_image_prediction_count
    _comet_image_prediction_count = 0
# 如果 comet_ml 变量为真，则定义一个包含多个回调函数的字典，否则定义一个空字典
callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if comet_ml
    else {}
)
```