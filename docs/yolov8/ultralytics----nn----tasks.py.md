# `.\yolov8\ultralytics\nn\tasks.py`

```py
# 导入必要的库和模块
import contextlib
from copy import deepcopy
from pathlib import Path

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块

# 从Ultralytics.nn.modules中导入多个自定义模块
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C3Ghost,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    WorldDetect,
    v10Detect,
)

# 从Ultralytics.utils中导入各种工具和函数
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2EDetectLoss,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    time_sync,
)

try:
    import thop
except ImportError:
    thop = None

# 定义一个基础模型类，作为Ultralytics YOLO系列模型的基类
class BaseModel(nn.Module):
    """The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family."""

    def forward(self, x, *args, **kwargs):
        """
        模型的前向传播方法，对单个尺度进行处理。包装了 `_forward_once` 方法。

        Args:
            x (torch.Tensor | dict): 输入的图像张量或包含图像张量和gt标签的字典。

        Returns:
            (torch.Tensor): 网络的输出。
        """
        if isinstance(x, dict):  # 对训练和验证过程中的情况进行处理
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        对网络进行前向传播。

        Args:
            x (torch.Tensor): 输入到模型的张量。
            profile (bool): 如果为True，打印每层的计算时间，默认为False。
            visualize (bool): 如果为True，保存模型的特征图，默认为False。
            augment (bool): 在预测过程中进行图像增强，默认为False。
            embed (list, optional): 要返回的特征向量或嵌入列表。

        Returns:
            (torch.Tensor): 模型的最后输出。
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)
    # 执行一次模型的前向传播
    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt, embeddings = [], [], []  # outputs
        
        # 遍历模型的每一层
        for m in self.model:
            # 如果当前层不是从前一层得到的
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            
            # 如果需要进行性能分析
            if profile:
                self._profile_one_layer(m, x, dt)
            
            # 执行当前层的计算
            x = m(x)  # run
            
            # 保存当前层的输出
            y.append(x if m.i in self.save else None)  # save output
            
            # 如果需要可视化特征图
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            
            # 如果需要返回特定层的嵌入向量
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                # 如果当前层是要返回的最大嵌入层，则直接返回嵌入向量
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        
        # 返回模型最后的输出
        return x

    # 执行输入图像 x 的增强操作，并返回增强后的推理结果
    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"WARNING ⚠️ {self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    # 对模型的单个层进行计算时间和 FLOPs 的性能分析，并将结果添加到提供的列表中
    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input. Appends the results to
        the provided list.

        Args:
            m (nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.

        Returns:
            None
        """
        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        
        # 计算该层的 FLOPs
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        
        # 开始计时
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        
        # 计算执行时间并记录
        dt.append((time_sync() - t) * 100)
        
        # 如果是模型的第一层，输出性能分析的表头信息
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        
        # 输出当前层的性能分析结果
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        
        # 如果是最后一层且输出为列表形式，则输出总计信息
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")
    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
        # 如果模型尚未融合
        if not self.is_fused():
            # 遍历模型的所有模块
            for m in self.model.modules():
                # 如果当前模块是 Conv、Conv2 或 DWConv，并且有 bn 属性
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    # 如果当前模块是 Conv2 类型，则执行融合卷积操作
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    # 融合卷积层和批归一化层
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 更新卷积层
                    delattr(m, "bn")  # 移除批归一化层
                    m.forward = m.forward_fuse  # 更新前向传播方法
                # 如果当前模块是 ConvTranspose 并且有 bn 属性
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    # 融合反卷积层和批归一化层
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # 移除批归一化层
                    m.forward = m.forward_fuse  # 更新前向传播方法
                # 如果当前模块是 RepConv 类型
                if isinstance(m, RepConv):
                    # 执行重复卷积融合操作
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # 更新前向传播方法
                # 如果当前模块是 RepVGGDW 类型
                if isinstance(m, RepVGGDW):
                    # 执行重复 VGG 深度可分离卷积融合操作
                    m.fuse()
                    m.forward = m.forward_fuse  # 更新前向传播方法
            # 打印模型信息
            self.info(verbose=verbose)

        # 返回融合后的模型实例
        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        # 获取所有标准化层（如 BatchNorm2d()）的类型元组
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        # 返回模型中标准化层数量是否小于阈值 thresh
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints model information.

        Args:
            detailed (bool): if True, prints out detailed information about the model. Defaults to False
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        # 调用 model_info 函数打印模型信息
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)
    def _apply(self, fn):
        """
        Applies a function to all the tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): the function to apply to the model

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)  # 调用父类的_apply方法，将fn函数应用于模型中不是参数或注册缓冲区的所有张量
        m = self.model[-1]  # 获取模型中的最后一个子模块（通常是Detect()类型）
        if isinstance(m, Detect):  # 检查最后一个子模块是否属于Detect类或其子类，如Segment, Pose, OBB, WorldDetect
            m.stride = fn(m.stride)  # 将fn函数应用于m的stride属性
            m.anchors = fn(m.anchors)  # 将fn函数应用于m的anchors属性
            m.strides = fn(m.strides)  # 将fn函数应用于m的strides属性
        return self  # 返回更新后的BaseModel对象

    def load(self, weights, verbose=True):
        """
        Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # 如果weights是字典，则获取字典中的"model"键对应的值，否则直接使用weights
        csd = model.float().state_dict()  # 将模型的state_dict转换为float类型的checkpoint state_dict
        csd = intersect_dicts(csd, self.state_dict())  # 获取模型状态字典和self对象的状态字典的交集，用于加载权重
        self.load_state_dict(csd, strict=False)  # 使用加载的状态字典csd来加载模型参数，strict=False表示允许不严格匹配模型结构
        if verbose:
            LOGGER.info(f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights")
            # 如果verbose为True，则打印日志，显示从预训练权重中转移了多少项到当前模型中

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()  # 如果模型中的损失函数属性criterion为None，则初始化损失函数

        preds = self.forward(batch["img"]) if preds is None else preds  # 如果未提供预测值preds，则使用模型前向传播得到预测值
        return self.criterion(preds, batch)  # 计算预测值和真实标签之间的损失值，使用模型的损失函数criterion

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")
        # 抛出NotImplementedError异常，提示需要由任务头部实现compute_loss()方法
class DetectionModel(BaseModel):
    """YOLOv8 detection model."""

    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        
        # Check if the first element in the 'backbone' section of the YAML config is 'Silence'
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "WARNING ⚠️ YOLOv9 `Silence` module is deprecated in favor of nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            # Update 'Silence' to 'nn.Identity' in the YAML config
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Define model configuration parameters
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        
        # Override the number of classes in the YAML config if 'nc' is provided
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        
        # Parse the model based on the YAML configuration
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        
        # Create a default names dictionary for the number of classes
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        
        # Check if 'inplace' is specified in the YAML config, default to True if not specified
        self.inplace = self.yaml.get("inplace", True)
        
        # Check if 'end2end' attribute is present in the last model component
        self.end2end = getattr(self.model[-1], "end2end", False)

        # Build strides
        m = self.model[-1]  # Detect()
        
        # Perform specific actions based on the type of 'm' (Detect subclass)
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            s = 256  # 2x min stride
            m.inplace = self.inplace
            
            # Define a function for the forward pass based on the 'end2end' attribute
            def _forward(x):
                """Performs a forward pass through the model, handling different Detect subclass types accordingly."""
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x)

            # Calculate the stride values based on the input size and the forward pass result
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            
            # Initialize biases for 'm'
            m.bias_init()  # only run once
        else:
            # Set default stride for models like RTDETR
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Initialize weights and biases for the model
        initialize_weights(self)
        
        # Output model information if verbose mode is enabled
        if verbose:
            self.info()
            LOGGER.info("")
    # 执行输入图像 x 的增强操作，并返回增强后的推理和训练输出
    def _predict_augment(self, x):
        # 如果设置了 end2end 属性为 True，则警告不支持 'augment=True' 的预测，回退到单尺度预测
        if getattr(self, "end2end", False):
            LOGGER.warning(
                "WARNING ⚠️ End2End model does not support 'augment=True' prediction. "
                "Reverting to single-scale prediction."
            )
            return self._predict_once(x)  # 调用单尺度预测方法
        img_size = x.shape[-2:]  # 获取图像的高度和宽度
        s = [1, 0.83, 0.67]  # 不同尺度的缩放比例
        f = [None, 3, None]  # 不同的翻转方式 (2-上下翻转, 3-左右翻转)
        y = []  # 存储输出
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))  # 缩放图像并根据需要翻转
            yi = super().predict(xi)[0]  # 进行前向推理
            yi = self._descale_pred(yi, fi, si, img_size)  # 对预测结果进行反缩放操作
            y.append(yi)
        y = self._clip_augmented(y)  # 对增强后的结果进行裁剪
        return torch.cat(y, -1), None  # 返回增强后的推理结果和空的训练输出

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """对增强推理后的预测进行反缩放操作（逆操作）。"""
        p[:, :4] /= scale  # 反缩放坐标
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)  # 拆分预测结果
        if flips == 2:
            y = img_size[0] - y  # 反上下翻转
        elif flips == 3:
            x = img_size[1] - x  # 反左右翻转
        return torch.cat((x, y, wh, cls), dim)  # 拼接反缩放后的结果

    def _clip_augmented(self, y):
        """裁剪 YOLO 增强推理结果的尾部。"""
        nl = self.model[-1].nl  # 检测层的数量 (P3-P5)
        g = sum(4**x for x in range(nl))  # 网格点数
        e = 1  # 排除层计数
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # 索引计算
        y[0] = y[0][..., :-i]  # 裁剪大尺度的输出
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # 索引计算
        y[-1] = y[-1][..., i:]  # 裁剪小尺度的输出
        return y

    def init_criterion(self):
        """初始化检测模型的损失函数。"""
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)
class OBBModel(DetectionModel):
    """YOLOv8 Oriented Bounding Box (OBB) model."""

    def __init__(self, cfg="yolov8n-obb.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 OBB model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        # 调用父类的初始化方法，设置模型的配置文件路径、通道数、类别数和详细输出标志

    def init_criterion(self):
        """Initialize the loss criterion for the model."""
        return v8OBBLoss(self)
        # 返回一个针对 OBB 模型的损失函数对象 v8OBBLoss



class SegmentationModel(DetectionModel):
    """YOLOv8 segmentation model."""

    def __init__(self, cfg="yolov8n-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 segmentation model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        # 调用父类的初始化方法，设置模型的配置文件路径、通道数、类别数和详细输出标志

    def init_criterion(self):
        """Initialize the loss criterion for the SegmentationModel."""
        return v8SegmentationLoss(self)
        # 返回一个针对分割模型的损失函数对象 v8SegmentationLoss



class PoseModel(DetectionModel):
    """YOLOv8 pose model."""

    def __init__(self, cfg="yolov8n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize YOLOv8 Pose model."""
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        # 如果配置不是字典，则加载模型的 YAML 配置文件
        # 如果给定了关键点形状参数，并且与配置文件中的不同，则记录日志并进行覆盖
        # 调用父类的初始化方法，设置模型的配置、通道数、类别数、数据关键点形状和详细输出标志

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""
        return v8PoseLoss(self)
        # 返回一个针对姿态估计模型的损失函数对象 v8PoseLoss



class ClassificationModel(BaseModel):
    """YOLOv8 classification model."""

    def __init__(self, cfg="yolov8n-cls.yaml", ch=3, nc=None, verbose=True):
        """Init ClassificationModel with YAML, channels, number of classes, verbose flag."""
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)
        # 调用 BaseModel 的初始化方法，然后调用自身的 _from_yaml 方法进行更详细的初始化

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set YOLOv8 model configurations and define the model architecture."""
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        # 如果 cfg 是字典，则直接使用，否则加载 YAML 文件得到配置字典

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        # 设置输入通道数为配置文件中的 ch 值或者默认值 ch

        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
        # 如果给定了类别数 nc 并且与配置文件中的不同，则记录日志并进行覆盖
        # 如果未指定 nc 并且配置文件中也没有指定，则引发 ValueError

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        # 使用配置字典和通道数解析模型得到模型对象和保存列表

        self.stride = torch.Tensor([1])  # no stride constraints
        # 设置模型的步长为固定值 1

        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        # 根据类别数设置默认的名称字典，键为类别索引，值为类别索引的字符串表示

        self.info()
        # 输出模型的详细信息
    def reshape_outputs(model, nc):
        """Update a TorchVision classification model to class count 'n' if required."""
        # 获取模型的最后一个子模块的名称和实例
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module
        
        # 如果最后一个模块是 Classify 类型（例如 YOLO 的分类头）
        if isinstance(m, Classify):
            # 如果当前输出特征数不等于 nc，则更新线性层的输出特征数
            if m.linear.out_features != nc:
                m.linear = nn.Linear(m.linear.in_features, nc)
        
        # 如果最后一个模块是 nn.Linear 类型（例如 ResNet, EfficientNet）
        elif isinstance(m, nn.Linear):
            # 如果当前输出特征数不等于 nc，则替换当前模块为新的 nn.Linear
            if m.out_features != nc:
                setattr(model, name, nn.Linear(m.in_features, nc))
        
        # 如果最后一个模块是 nn.Sequential 类型
        elif isinstance(m, nn.Sequential):
            # 获取所有子模块的类型列表
            types = [type(x) for x in m]
            
            # 如果类型列表中包含 nn.Linear
            if nn.Linear in types:
                # 找到最后一个 nn.Linear 的索引
                i = len(types) - 1 - types[::-1].index(nn.Linear)  # last nn.Linear index
                # 如果该 nn.Linear 的输出特征数不等于 nc，则更新它
                if m[i].out_features != nc:
                    m[i] = nn.Linear(m[i].in_features, nc)
            
            # 如果类型列表中包含 nn.Conv2d
            elif nn.Conv2d in types:
                # 找到最后一个 nn.Conv2d 的索引
                i = len(types) - 1 - types[::-1].index(nn.Conv2d)  # last nn.Conv2d index
                # 如果该 nn.Conv2d 的输出通道数不等于 nc，则更新它
                if m[i].out_channels != nc:
                    m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)

    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        # 返回一个 v8ClassificationLoss 的实例，用于分类模型的损失计算
        return v8ClassificationLoss()
    # RTDETRDetectionModel 类，继承自 DetectionModel，用于实现 RTDETR（Real-time DEtection and Tracking using Transformers）检测模型。
    """
    RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.

    Attributes:
        cfg (str): The configuration file path or preset string. Default is 'rtdetr-l.yaml'.
        ch (int): Number of input channels. Default is 3 (RGB).
        nc (int, optional): Number of classes for object detection. Default is None.
        verbose (bool): Specifies if summary statistics are shown during initialization. Default is True.

    Methods:
        init_criterion: Initializes the criterion used for loss calculation.
        loss: Computes and returns the loss during training.
        predict: Performs a forward pass through the network and returns the output.
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize the RTDETRDetectionModel.

        Args:
            cfg (str): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes. Defaults to None.
            verbose (bool, optional): Print additional information during initialization. Defaults to True.
        """
        # 调用父类 DetectionModel 的初始化方法，传入配置文件名、通道数、类别数和是否显示详细信息
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        # 导入 RTDETRDetectionLoss 类，用于初始化损失函数，传入类别数和是否使用视觉语义分割损失
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)
    def loss(self, batch, preds=None):
        """
        Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (torch.Tensor, optional): Precomputed model predictions. Defaults to None.

        Returns:
            (tuple): A tuple containing the total loss and main three losses in a tensor.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()  # 初始化损失函数

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        batch_idx = batch["batch_idx"]
        # 计算每个 batch 中的样本数
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        # 构建目标数据字典，包括类别、边界框、批次索引和分组信息
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),  # 类别数据
            "bboxes": batch["bboxes"].to(device=img.device),  # 边界框数据
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),  # 批次索引
            "gt_groups": gt_groups,  # 分组信息
        }

        # 如果未提供预测值 preds，则使用模型进行预测
        preds = self.predict(img, batch=targets) if preds is None else preds
        # 解析预测结果中的各项数据
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            # 按照 dn_meta 中的信息对预测结果进行分割
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        # 将编码器的预测结果与解码器的预测结果拼接起来
        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        # 计算损失函数
        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        # 计算并返回总损失和主要三个损失项的张量形式
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )
    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            batch (dict, optional): Ground truth data for evaluation. Defaults to None.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs

        for m in self.model[:-1]:  # iterate through all layers except the last one (head)
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # retrieve inputs from earlier layers

            if profile:
                self._profile_one_layer(m, x, dt)  # profile the computation time of the current layer

            x = m(x)  # perform forward pass through the current layer
            y.append(x if m.i in self.save else None)  # save output if specified by self.save

            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)  # visualize feature maps if enabled

            if embed and m.i in embed:
                # compute embeddings by adaptive average pooling and flattening
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)  # return embeddings if the last embedding layer is reached

        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # perform inference with the head layer using saved outputs and optional batch data
        return x  # return the final output tensor
class WorldModel(DetectionModel):
    """YOLOv8 World Model."""

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 world model with given config and parameters."""
        # 创建一个随机初始化的文本特征张量，形状为 (1, nc 或 80, 512)，用作特征占位符
        self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder
        # 初始化 CLIP 模型占位符为 None
        self.clip_model = None  # CLIP model placeholder
        # 调用父类的初始化方法，传入配置 cfg、通道数 ch、类别数 nc 和是否详细输出 verbose
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """Set classes in advance so that model could do offline-inference without clip model."""
        try:
            import clip
        except ImportError:
            # 如果导入 clip 失败，则安装要求的版本
            check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip

        # 如果 self.clip_model 属性不存在且 cache_clip_model 为 True，则加载 CLIP 模型
        if (
            not getattr(self, "clip_model", None) and cache_clip_model
        ):  # for backwards compatibility of models lacking clip_model attribute
            self.clip_model = clip.load("ViT-B/32")[0]
        
        # 如果 cache_clip_model 为 True，则使用缓存的 clip_model，否则加载新的 CLIP 模型
        model = self.clip_model if cache_clip_model else clip.load("ViT-B/32")[0]
        # 获取模型所在设备
        device = next(model.parameters()).device
        # 将输入文本转换为 CLIP 模型可接受的 token，并发送到指定设备
        text_token = clip.tokenize(text).to(device)
        # 使用 CLIP 模型对文本 token 进行编码，按批次分割并进行编码，然后分离梯度
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        # 如果只有一个批次，则直接取第一个编码结果；否则在指定维度上拼接所有结果
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        # 对文本特征进行 L2 范数归一化处理
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        # 将归一化后的文本特征重塑为指定形状，更新 self.txt_feats
        self.txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        # 更新模型最后一层的类别数为文本的长度
        self.model[-1].nc = len(text)
    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            txt_feats (torch.Tensor): The text features, use it if it's given. Defaults to None.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        # Convert txt_feats to device and dtype of input tensor x, defaulting to self.txt_feats if txt_feats is None
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        
        # If txt_feats has different length than x, repeat txt_feats to match the length of x
        if len(txt_feats) != len(x):
            txt_feats = txt_feats.repeat(len(x), 1, 1)
        
        # Create a deep copy of txt_feats for potential use later
        ori_txt_feats = txt_feats.clone()
        
        y, dt, embeddings = [], [], []  # Initialize lists for outputs
        
        # Iterate through each module in self.model (except the head part)
        for m in self.model:
            # Check if m.f is not -1, meaning it's not from a previous layer
            if m.f != -1:
                # Determine input x based on m.f, which can be an int or a list of ints
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            
            # If profiling is enabled, profile the computation time for the current layer
            if profile:
                self._profile_one_layer(m, x, dt)
            
            # Apply specific operations based on module type
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)  # Apply attention module with text features
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)  # Apply world detection module with original text features
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)  # Apply image pooling attention module to text features
            else:
                x = m(x)  # Perform standard forward pass for other module types
            
            # Save the output of the current module
            y.append(x if m.i in self.save else None)
            
            # If visualization is enabled, save feature maps for visualization
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            
            # If embeddings are requested and the current module index is in embed list, compute embeddings
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # Flatten embeddings
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)  # Return concatenated embeddings
        
        # Return the final output tensor
        return x


    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        # Initialize the criterion if it's not already initialized
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        
        # If preds is None, compute predictions using forward pass with batch["img"] and optional txt_feats
        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        
        # Compute and return the loss using initialized criterion
        return self.criterion(preds, batch)
class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLO network's final layer."""
        # 对每个模型进行前向传播，获取输出列表
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # 将每个模型的输出按第2维度（channel维度）拼接起来，用于非极大值抑制
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # 返回输出以及空对象，用于推断和训练输出


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.

    Example:
        ```python
        with temporary_modules({'old.module': 'new.module'}, {'old.module.attribute': 'new.module.attribute'}):
            import old.module  # this will now import new.module
            from old.module import attribute  # this will now import new.module.attribute
        ```py

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """

    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # 将新的模块路径设置在sys.modules下的旧名称
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # 将新的模块设置在sys.modules下的旧名称
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # 清除临时添加的模块路径
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


def torch_safe_load(weight):
    """
    This function attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised,
    ```
    # 导入下载相关的函数
    from ultralytics.utils.downloads import attempt_download_asset
    
    # 检查文件后缀是否为 ".pt"，如果不是则抛出异常
    check_suffix(file=weight, suffix=".pt")
    
    # 尝试从在线获取模型文件，如果本地不存在的话
    file = attempt_download_asset(weight)  # search online if missing locally
    
    try:
        # 使用临时的模块映射和属性映射加载模型检查点文件
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
            },
        ):
            # 使用 torch.load() 加载模型检查点文件到内存中，指定在 CPU 上加载
            ckpt = torch.load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # 如果捕获到模块未找到的异常
        if e.name == "models":
            # 抛出类型错误，提示用户模型不兼容，并提供建议
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolov8n.pt'"
                )
            ) from e
        # 记录警告信息，指出模型需要缺失的模块，并建议安装
        LOGGER.warning(
            f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolov8n.pt'"
        )
        # 安装缺失的模块
        check_requirements(e.name)
        # 再次尝试加载模型检查点文件到内存中，指定在 CPU 上加载
        ckpt = torch.load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # 如果加载的模型检查点不是字典类型，给出警告信息，并假设其格式不正确，尝试修复
        LOGGER.warning(
            f"WARNING ⚠️ The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        # 假设该文件是使用 torch.save(model, "saved_model.pt") 保存的 YOLO 实例，将其存入字典中
        ckpt = {"model": ckpt.model}

    # 返回加载的模型检查点及其文件路径
    return ckpt, file  # load
# 尝试加载多个模型的权重（模型集合或单个模型）
def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""

    # 创建一个 Ensemble 对象，用于存储加载的模型
    ensemble = Ensemble()

    # 遍历 weights 列表中的每个权重，如果 weights 是单个模型，则转换为列表再遍历
    for w in weights if isinstance(weights, list) else [weights]:
        # 使用 torch_safe_load 加载权重和检查点
        ckpt, w = torch_safe_load(w)  # load ckpt

        # 如果检查点中存在 "train_args"，则将其与默认参数 DEFAULT_CFG_DICT 合并作为 args
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None  # combined args

        # 从检查点中获取 EMA 模型或主模型，并将其转换为指定设备上的浮点数模型
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # 更新模型的一致性和其他属性
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = guess_model_task(model)

        # 如果模型没有 stride 属性，则设置默认 stride 为 [32.0]
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        # 如果开启了融合模式（fuse）并且模型具有 fuse 方法，则进行融合并设置模型为评估模式
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())  # model in eval mode

    # 遍历 ensemble 中的每个模型，并更新其模块的 inplace 属性或进行其他兼容性更新
    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # 如果 ensemble 中只有一个模型，则返回该模型
    if len(ensemble) == 1:
        return ensemble[-1]

    # 如果有多个模型，则返回整个 ensemble
    LOGGER.info(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"
    return ensemble


# 尝试加载单个模型的权重
def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""

    # 使用 torch_safe_load 加载单个模型的权重和检查点
    ckpt, weight = torch_safe_load(weight)  # load ckpt

    # 将模型的默认配置参数与检查点中的训练参数合并，优先使用模型参数
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args

    # 从检查点中获取 EMA 模型或主模型，并将其转换为指定设备上的浮点数模型
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

    # 更新模型的一致性和其他属性
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = guess_model_task(model)

    # 如果模型没有 stride 属性，则设置默认 stride 为 [32.0]
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    # 如果开启了融合模式（fuse）并且模型具有 fuse 方法，则进行融合并设置模型为评估模式
    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()  # model in eval mode

    # 遍历模型的所有模块，并更新 inplace 属性或进行其他兼容性更新
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # 返回加载的模型和检查点
    return model, ckpt


# 解析 YOLO 模型的模型字典，转换为 PyTorch 模型
def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    # 设置最大通道数为无穷大
    max_channels = float("inf")
    
    # 从字典 d 中获取 nc、act、scales 的值，并分别赋给 nc、act、scales
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    
    # 从字典 d 中获取 depth_multiple、width_multiple、kpt_shape 的值，如果不存在则使用默认值
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    
    # 如果 scales 存在
    if scales:
        # 从字典 d 中获取 scale 的值
        scale = d.get("scale")
        # 如果 scale 不存在，则选择 scales 字典的第一个键作为默认值，并发出警告信息
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        # 将 depth、width、max_channels 设置为 scales[scale] 中的值
        depth, width, max_channels = scales[scale]
    
    # 如果 act 存在
    if act:
        # 重新定义默认激活函数为 eval(act)，例如 Conv.default_act = nn.SiLU()
        Conv.default_act = eval(act)
        # 如果 verbose 为真，则打印激活函数信息
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")
    
    # 如果 verbose 为真
    if verbose:
        # 打印信息表头，显示模块的各种参数信息
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    
    # 将 ch 包装成列表
    ch = [ch]
    
    # 初始化 layers、save、c2 为空列表
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    
    # 返回 nn.Sequential(*layers) 和 sorted(save)
    return nn.Sequential(*layers), sorted(save)
def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    import re  # 导入正则表达式模块

    path = Path(path)  # 将路径转换为Path对象
    # 检查是否是YOLOv5或YOLOv8系列模型，若是，则重命名模型文件名
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    # 统一模型文件名格式，例如将yolov8x.yaml -> yolov8.yaml
    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))
    # 检查并加载YAML文件，优先使用unified_path，其次使用原始路径path
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # 加载YAML文件内容到字典d，表示模型
    d["scale"] = guess_model_scale(path)  # 猜测模型的规模大小，并存储在字典中
    d["yaml_file"] = str(path)  # 记录模型的YAML文件路径到字典中
    return d  # 返回模型的字典表示


def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale. The function
    uses regular expression matching to find the pattern of the model scale in the YAML file name, which is denoted by
    n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    with contextlib.suppress(AttributeError):
        import re  # 导入正则表达式模块

        return re.search(r"yolov\d+([nslmx])", Path(model_path).stem).group(1)  # 从模型路径中提取规模大小字符，如n, s, m, l, x
    return ""  # 若提取失败，则返回空字符串


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # 提取输出模块名称，并转换为小写
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"  # 分类任务
        if "detect" in m:
            return "detect"  # 目标检测任务
        if m == "segment":
            return "segment"  # 分割任务
        if m == "pose":
            return "pose"  # 姿态估计任务
        if m == "obb":
            return "obb"  # 方向边界框任务

    # 从模型配置中猜测任务类型
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)

    # 从PyTorch模型中猜测任务类型
    # 如果 model 是 nn.Module 的实例，表示这是一个 PyTorch 模型
    if isinstance(model, nn.Module):  # PyTorch model
        # 遍历可能包含任务信息的不同路径
        for x in "model.args", "model.model.args", "model.model.model.args":
            # 尝试从路径中获取任务信息，如果成功则返回任务名称
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        
        # 如果未能从上述路径中获取任务信息，尝试从 YAML 文件中解析任务信息
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            # 尝试解析 YAML 文件并转换为任务信息
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))

        # 遍历模型的所有模块
        for m in model.modules():
            # 根据模块类型判断任务类型
            if isinstance(m, Segment):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, v10Detect)):
                return "detect"

    # 如果 model 是字符串或路径的实例，尝试根据文件名猜测任务类型
    if isinstance(model, (str, Path)):
        model = Path(model)
        # 根据文件名的特定标识来猜测任务类型
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # 如果无法从模型中确定任务类型，则发出警告，并假设任务为检测 ("detect")
    LOGGER.warning(
        "WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect
```