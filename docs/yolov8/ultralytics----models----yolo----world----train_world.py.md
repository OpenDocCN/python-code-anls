# `.\yolov8\ultralytics\models\yolo\world\train_world.py`

```py
# 导入需要的模块和函数
from ultralytics.data import YOLOConcatDataset, build_grounding, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.world import WorldTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import de_parallel

# 定义一个从 WorldTrainer 继承的类，用于从头开始训练世界模型
class WorldTrainerFromScratch(WorldTrainer):
    """
    A class extending the WorldTrainer class for training a world model from scratch on open-set dataset.

    Example:
        ```python
        from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
        from ultralytics import YOLOWorld

        data = dict(
            train=dict(
                yolo_data=["Objects365.yaml"],
                grounding_data=[
                    dict(
                        img_path="../datasets/flickr30k/images",
                        json_file="../datasets/flickr30k/final_flickr_separateGT_train.json",
                    ),
                    dict(
                        img_path="../datasets/GQA/images",
                        json_file="../datasets/GQA/final_mixed_train_no_coco.json",
                    ),
                ],
            ),
            val=dict(yolo_data=["lvis.yaml"]),
        )

        model = YOLOWorld("yolov8s-worldv2.yaml")
        model.train(data=data, trainer=WorldTrainerFromScratch)
        ```py
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a WorldTrainer object with given arguments."""
        # 如果没有传入 overrides 参数，则设为一个空字典
        if overrides is None:
            overrides = {}
        # 调用父类的构造函数初始化对象
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (List[str] | str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        # 计算并获取模型的最大步长
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        # 如果不是训练模式，则构建 YOLO 数据集
        if mode != "train":
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
        # 否则，构建多模态数据集或者地面推理数据集
        dataset = [
            build_yolo_dataset(self.args, im_path, batch, self.data, stride=gs, multi_modal=True)
            if isinstance(im_path, str)
            else build_grounding(self.args, im_path["img_path"], im_path["json_file"], batch, stride=gs)
            for im_path in img_path
        ]
        # 如果数据集数量大于1，则返回连接后的 YOLO 数据集，否则返回单个数据集
        return YOLOConcatDataset(dataset) if len(dataset) > 1 else dataset[0]
    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        # 初始化最终数据字典
        final_data = {}
        # 获取数据字典的路径
        data_yaml = self.args.data
        # 断言数据字典中包含训练集数据
        assert data_yaml.get("train", False), "train dataset not found"  # object365.yaml
        # 断言数据字典中包含验证集数据
        assert data_yaml.get("val", False), "validation dataset not found"  # lvis.yaml
        # 构建数据字典，检查每个yolo_data的数据集
        data = {k: [check_det_dataset(d) for d in v.get("yolo_data", [])] for k, v in data_yaml.items()}
        # 断言验证集数据只有一个数据集
        assert len(data["val"]) == 1, f"Only support validating on 1 dataset for now, but got {len(data['val'])}."
        # 根据数据集名判断验证集的分割类型
        val_split = "minival" if "lvis" in data["val"][0]["val"] else "val"
        # 处理minival数据集的路径
        for d in data["val"]:
            if d.get("minival") is None:  # for lvis dataset
                continue
            d["minival"] = str(d["path"] / d["minival"])
        # 遍历训练集和验证集，将数据路径添加到final_data字典中
        for s in ["train", "val"]:
            final_data[s] = [d["train" if s == "train" else val_split] for d in data[s]]
            # 如果有地面数据，保存到final_data中
            grounding_data = data_yaml[s].get("grounding_data")
            if grounding_data is None:
                continue
            grounding_data = grounding_data if isinstance(grounding_data, list) else [grounding_data]
            # 断言地面数据应为字典格式
            for g in grounding_data:
                assert isinstance(g, dict), f"Grounding data should be provided in dict format, but got {type(g)}"
            final_data[s] += grounding_data
        # 设置训练所需的类别数和类别名称
        final_data["nc"] = data["val"][0]["nc"]
        final_data["names"] = data["val"][0]["names"]
        # 将最终数据保存到对象的属性中
        self.data = final_data
        # 返回训练集和验证集的路径
        return final_data["train"], final_data["val"][0]

    def plot_training_labels(self):
        """DO NOT plot labels."""
        # 该方法不做任何操作，避免绘制标签

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO-World model."""
        # 获取验证集数据
        val = self.args.data["val"]["yolo_data"][0]
        # 设置验证器的数据和分割类型
        self.validator.args.data = val
        self.validator.args.split = "minival" if isinstance(val, str) and "lvis" in val else "val"
        # 调用父类方法执行最终评估
        return super().final_eval()
```