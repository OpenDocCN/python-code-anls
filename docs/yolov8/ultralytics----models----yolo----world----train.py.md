# `.\yolov8\ultralytics\models\yolo\world\train.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

import itertools  # 导入 itertools 模块

from ultralytics.data import build_yolo_dataset  # 从 ultralytics.data 模块导入 build_yolo_dataset 函数
from ultralytics.models import yolo  # 从 ultralytics.models 模块导入 yolo 模型
from ultralytics.nn.tasks import WorldModel  # 从 ultralytics.nn.tasks 导入 WorldModel 类
from ultralytics.utils import DEFAULT_CFG, RANK, checks  # 从 ultralytics.utils 导入 DEFAULT_CFG, RANK, checks
from ultralytics.utils.torch_utils import de_parallel  # 从 ultralytics.utils.torch_utils 导入 de_parallel 函数


def on_pretrain_routine_end(trainer):
    """Callback."""
    if RANK in {-1, 0}:
        # NOTE: for evaluation
        # 从 trainer.test_loader.dataset.data["names"] 中获取所有名称，仅保留第一个斜杠之前的部分作为名称
        names = [name.split("/")[0] for name in list(trainer.test_loader.dataset.data["names"].values())]
        # 设置 trainer.ema.ema 中的类别为 names，不缓存剪辑模型
        de_parallel(trainer.ema.ema).set_classes(names, cache_clip_model=False)
    device = next(trainer.model.parameters()).device  # 获取 trainer.model 中第一个参数的设备信息
    # 使用指定设备加载 ViT-B/32 模型到 trainer.text_model 中
    trainer.text_model, _ = trainer.clip.load("ViT-B/32", device=device)
    # 将 trainer.text_model 中所有参数设为不需要梯度计算
    for p in trainer.text_model.parameters():
        p.requires_grad_(False)


class WorldTrainer(yolo.detect.DetectionTrainer):
    """
    A class to fine-tune a world model on a close-set dataset.

    Example:
        ```py
        from ultralytics.models.yolo.world import WorldModel

        args = dict(model='yolov8s-world.pt', data='coco8.yaml', epochs=3)
        trainer = WorldTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a WorldTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        # 调用父类构造函数初始化对象
        super().__init__(cfg, overrides, _callbacks)

        # Import and assign clip
        try:
            import clip
        except ImportError:
            # 检查是否安装了 CLIP 库，如果未安装，则安装该库
            checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip
        self.clip = clip  # 将 clip 模块赋值给 self.clip

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return WorldModel initialized with specified config and weights."""
        # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
        # NOTE: Following the official config, nc hard-coded to 80 for now.
        # 使用 cfg 和 weights 参数初始化 WorldModel，设置 nc 为数据集中的最大文本样本数和 80 中的最小值
        model = WorldModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=3,
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)  # 如果提供了 weights 参数，则加载模型权重
        self.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)  # 添加回调函数 on_pretrain_routine_end 到对象

        return model  # 返回初始化的 WorldModel 对象
    # 获取当前模型的最大步长，如果模型存在则获取最大步长，否则返回0，并转为整数
    gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
    # 调用函数构建 YOLO 数据集，返回构建的数据集对象
    return build_yolo_dataset(
        self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, multi_modal=mode == "train"
    )


```py    
    # 调用父类方法处理图像批次
    batch = super().preprocess_batch(batch)

    # NOTE: add text features
    # 将所有图像批次中的文本合并为一个列表
    texts = list(itertools.chain(*batch["texts"]))
    # 使用 CLIP 模型对文本进行标记化，并将其转移到与图像批次相同的设备上
    text_token = self.clip.tokenize(texts).to(batch["img"].device)
    # 使用文本模型对文本进行编码，并转换为与图像批次相同的数据类型（torch.float32）
    txt_feats = self.text_model.encode_text(text_token).to(dtype=batch["img"].dtype)
    # 对编码后的文本特征进行归一化处理
    txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
    # 将处理后的文本特征重塑为适合批次的形状，并存储在批次字典中
    batch["txt_feats"] = txt_feats.reshape(len(batch["texts"]), -1, txt_feats.shape[-1])
    # 返回预处理后的批次数据
    return batch
```