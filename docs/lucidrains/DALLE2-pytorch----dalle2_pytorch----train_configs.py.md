# `.\lucidrains\DALLE2-pytorch\dalle2_pytorch\train_configs.py`

```py
# 导入所需的库
import json
from torchvision import transforms as T
from pydantic import BaseModel, validator, model_validator
from typing import List, Optional, Union, Tuple, Dict, Any, TypeVar

# 导入自定义的模块
from x_clip import CLIP as XCLIP
from open_clip import list_pretrained
from coca_pytorch import CoCa

# 导入自定义的模块中的类
from dalle2_pytorch.dalle2_pytorch import (
    CoCaAdapter,
    OpenAIClipAdapter,
    OpenClipAdapter,
    Unet,
    Decoder,
    DiffusionPrior,
    DiffusionPriorNetwork,
    XClipAdapter
)
from dalle2_pytorch.trackers import Tracker, create_loader, create_logger, create_saver

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 定义类型变量
InnerType = TypeVar('InnerType')
ListOrTuple = Union[List[InnerType], Tuple[InnerType]]
SingularOrIterable = Union[InnerType, ListOrTuple]

# 通用的 pydantic 类

# 训练集划分配置类
class TrainSplitConfig(BaseModel):
    train: float = 0.75
    val: float = 0.15
    test: float = 0.1

    # 验证所有参数的和是否为1
    @model_validator(mode = 'after')
    def validate_all(self, m):
        actual_sum = sum([*dict(self).values()])
        if actual_sum != 1.:
            raise ValueError(f'{dict(self).keys()} must sum to 1.0. Found: {actual_sum}')
        return self

# 日志追踪配置类
class TrackerLogConfig(BaseModel):
    log_type: str = 'console'
    resume: bool = False  # For logs that are saved to unique locations, resume a previous run
    auto_resume: bool = False  # If the process crashes and restarts, resume from the run that crashed
    verbose: bool = False

    class Config:
        # 每个日志类型都有自己的参数，将通过配置传递
        extra = "allow"

    # 创建日志记录器
    def create(self, data_path: str):
        kwargs = self.dict()
        return create_logger(self.log_type, data_path, **kwargs)

# 加载追踪配置类
class TrackerLoadConfig(BaseModel):
    load_from: Optional[str] = None
    only_auto_resume: bool = False  # Only attempt to load if the logger is auto-resuming

    class Config:
        extra = "allow"

    # 创建加载器
    def create(self, data_path: str):
        kwargs = self.dict()
        if self.load_from is None:
            return None
        return create_loader(self.load_from, data_path, **kwargs)

# 保存追踪配置类
class TrackerSaveConfig(BaseModel):
    save_to: str = 'local'
    save_all: bool = False
    save_latest: bool = True
    save_best: bool = True

    class Config:
        extra = "allow"

    # 创建保存器
    def create(self, data_path: str):
        kwargs = self.dict()
        return create_saver(self.save_to, data_path, **kwargs)

# 追踪配置类
class TrackerConfig(BaseModel):
    data_path: str = '.tracker_data'
    overwrite_data_path: bool = False
    log: TrackerLogConfig
    load: Optional[TrackerLoadConfig] = None
    save: Union[List[TrackerSaveConfig], TrackerSaveConfig]

    # 创建追踪器
    def create(self, full_config: BaseModel, extra_config: dict, dummy_mode: bool = False) -> Tracker:
        tracker = Tracker(self.data_path, dummy_mode=dummy_mode, overwrite_data_path=self.overwrite_data_path)
        # 添加日志记录器
        tracker.add_logger(self.log.create(self.data_path))
        # 添加加载器
        if self.load is not None:
            tracker.add_loader(self.load.create(self.data_path))
        # 添加保存器或保存器
        if isinstance(self.save, list):
            for save_config in self.save:
                tracker.add_saver(save_config.create(self.data_path))
        else:
            tracker.add_saver(self.save.create(self.data_path))
        # 初始化所有组件并验证所有数据是否有效
        tracker.init(full_config, extra_config)
        return tracker

# 扩散先验配置类

# 适配器配置类
class AdapterConfig(BaseModel):
    make: str = "openai"
    model: str = "ViT-L/14"
    base_model_kwargs: Optional[Dict[str, Any]] = None
    # 创建适配器对象的方法
    def create(self):
        # 如果适配器类型是 "openai"，则返回 OpenAIClipAdapter 对象
        if self.make == "openai":
            return OpenAIClipAdapter(self.model)
        # 如果适配器类型是 "open_clip"，则返回 OpenClipAdapter 对象
        elif self.make == "open_clip":
            # 获取预训练模型列表，并选择对应模型的检查点
            pretrained = dict(list_pretrained())
            checkpoint = pretrained[self.model]
            return OpenClipAdapter(name=self.model, pretrained=checkpoint)
        # 如果适配器类型是 "x-clip"，则返回 XClipAdapter 对象
        elif self.make == "x-clip":
            return XClipAdapter(XCLIP(**self.base_model_kwargs))
        # 如果适配器类型是 "coca"，则返回 CoCaAdapter 对象
        elif self.make == "coca":
            return CoCaAdapter(CoCa(**self.base_model_kwargs))
        # 如果适配器类型不匹配任何已知类型，则抛出属性错误异常
        else:
            raise AttributeError("No adapter with that name is available.")
# 定义 DiffusionPriorNetworkConfig 类，包含了模型的各种配置参数
class DiffusionPriorNetworkConfig(BaseModel):
    dim: int  # 模型维度
    depth: int  # 模型深度
    max_text_len: Optional[int] = None  # 最大文本长度
    num_timesteps: Optional[int] = None  # 时间步数
    num_time_embeds: int = 1  # 时间嵌入数量
    num_image_embeds: int = 1  # 图像嵌入数量
    num_text_embeds: int = 1  # 文本嵌入数量
    dim_head: int = 64  # 头部维度
    heads: int = 8  # 头部数量
    ff_mult: int = 4  # FeedForward 层倍数
    norm_in: bool = False  # 输入层是否进行归一化
    norm_out: bool = True  # 输出层是否进行归一化
    attn_dropout: float = 0.  # 注意力机制的 dropout 概率
    ff_dropout: float = 0.  # FeedForward 层的 dropout 概率
    final_proj: bool = True  # 是否进行最终投影
    normformer: bool = False  # 是否使用 Normformer
    rotary_emb: bool = True  # 是否使用旋转嵌入

    class Config:
        extra = "allow"

    # 创建 DiffusionPriorNetwork 对象
    def create(self):
        kwargs = self.dict()
        return DiffusionPriorNetwork(**kwargs)

# 定义 DiffusionPriorConfig 类，包含了模型的配置参数
class DiffusionPriorConfig(BaseModel):
    clip: Optional[AdapterConfig] = None  # 适配器配置
    net: DiffusionPriorNetworkConfig  # DiffusionPriorNetworkConfig 对象
    image_embed_dim: int  # 图像嵌入维度
    image_size: int  # 图像尺寸
    image_channels: int = 3  # 图像通道数
    timesteps: int = 1000  # 时间步数
    sample_timesteps: Optional[int] = None  # 采样时间步数
    cond_drop_prob: float = 0.  # 条件丢弃概率
    loss_type: str = 'l2'  # 损失类型
    predict_x_start: bool = True  # 是否预测 x 起始点
    beta_schedule: str = 'cosine'  # beta 调度
    condition_on_text_encodings: bool = True  # 是否在文本编码上进行条件

    class Config:
        extra = "allow"

    # 创建 DiffusionPrior 对象
    def create(self):
        kwargs = self.dict()

        has_clip = exists(kwargs.pop('clip'))
        kwargs.pop('net')

        clip = None
        if has_clip:
            clip = self.clip.create()

        diffusion_prior_network = self.net.create()
        return DiffusionPrior(net=diffusion_prior_network, clip=clip, **kwargs)

# 定义 DiffusionPriorTrainConfig 类，包含了训练配置参数
class DiffusionPriorTrainConfig(BaseModel):
    epochs: int = 1  # 训练轮数
    lr: float = 1.1e-4  # 学习率
    wd: float = 6.02e-2  # 权重衰减
    max_grad_norm: float = 0.5  # 最大梯度范数
    use_ema: bool = True  # 是否使用指数移动平均
    ema_beta: float = 0.99  # 指数移动平均的 beta
    amp: bool = False  # 是否使用混合精度训练
    warmup_steps: Optional[int] = None  # 热身步数
    save_every_seconds: int = 3600  # 保存模型的时间间隔
    eval_timesteps: List[int] = [64]  # 评估时间步数
    best_validation_loss: float = 1e9  # 最佳验证损失
    current_epoch: int = 0  # 当前轮数
    num_samples_seen: int = 0  # 当前样本数
    random_seed: int = 0  # 随机种子

# 定义 DiffusionPriorDataConfig 类，包含了数据配置参数
class DiffusionPriorDataConfig(BaseModel):
    image_url: str  # 嵌入文件夹路径
    meta_url: str  # 图像元数据（标题）路径
    splits: TrainSplitConfig  # 数据集的训练、验证、测试拆分
    batch_size: int  # 每个 GPU 的批量大小
    num_data_points: int = 25e7  # 训练数据点总数
    eval_every_seconds: int = 3600  # 多久进行一次验证统计

# 定义 TrainDiffusionPriorConfig 类，包含了训练配置参数
class TrainDiffusionPriorConfig(BaseModel):
    prior: DiffusionPriorConfig  # DiffusionPriorConfig 对象
    data: DiffusionPriorDataConfig  # DiffusionPriorDataConfig 对象
    train: DiffusionPriorTrainConfig  # DiffusionPriorTrainConfig 对象
    tracker: TrackerConfig  # 跟踪器配置

    # 从 JSON 路径加载配置
    @classmethod
    def from_json_path(cls, json_path):
        with open(json_path) as f:
            config = json.load(f)
        return cls(**config)

# 解码器 Pydantic 类

# 定义 UnetConfig 类，包含了 Unet 模型的配置参数
class UnetConfig(BaseModel):
    dim: int  # 维度
    dim_mults: ListOrTuple[int]  # 维度倍增列表
    image_embed_dim: Optional[int] = None  # 图像嵌入维度
    text_embed_dim: Optional[int] = None  # 文本嵌入维度
    cond_on_text_encodings: Optional[bool] = None  # 是否在文本编码上进行条件
    cond_dim: Optional[int] = None  # 条件维度
    channels: int = 3  # 通道数
    self_attn: SingularOrIterable[bool] = False  # 自注意力机制
    attn_dim_head: int = 32  # 注意力头部维度
    attn_heads: int = 16  # 注意力头部数量
    init_cross_embed: bool = True  # 是否初始化交叉嵌入

    class Config:
        extra = "allow"

# 定义 DecoderConfig 类，包含了解码器的配置参数
class DecoderConfig(BaseModel):
    unets: ListOrTuple[UnetConfig]  # UnetConfig 列表
    image_size: Optional[int] = None  # 图像尺寸
    image_sizes: ListOrTuple[int] = None  # 图像尺寸列表
    clip: Optional[AdapterConfig] = None  # 适配器配置（如果未提供嵌入，则使用 clip 模型）
    channels: int = 3  # 通道数
    timesteps: int = 1000  # 时间步数
    sample_timesteps: Optional[SingularOrIterable[Optional[int]]] = None  # 采样时间步数
    loss_type: str = 'l2'  # 损失类型
    beta_schedule: Optional[ListOrTuple[str]] = None  # beta 调度（None 表示所有余弦）
    # 定义学习方差的参数，默认为 True
    learned_variance: SingularOrIterable[bool] = True
    # 定义图像条件下的丢弃概率，默认为 0.1
    image_cond_drop_prob: float = 0.1
    # 定义文本条件下的丢弃概率，默认为 0.5

    def create(self):
        # 从参数中提取解码器的参数
        decoder_kwargs = self.dict()

        # 从解码器参数中提取 UNet 的配置
        unet_configs = decoder_kwargs.pop('unets')
        # 根据 UNet 的配置创建 UNet 对象列表
        unets = [Unet(**config) for config in unet_configs]

        # 检查是否存在剪辑参数
        has_clip = exists(decoder_kwargs.pop('clip'))
        clip = None
        # 如果存在剪辑参数，则创建剪辑对象
        if has_clip:
            clip = self.clip.create()

        # 返回解码器对象，传入 UNet 对象列表和剪辑对象
        return Decoder(unets, clip=clip, **decoder_kwargs)

    # 验证器，用于检查图像大小参数
    @validator('image_sizes')
    def check_image_sizes(cls, image_sizes, values):
        # 如果 image_size 和 image_sizes 中只有一个存在，则抛出异常
        if exists(values.get('image_size')) ^ exists(image_sizes):
            return image_sizes
        raise ValueError('either image_size or image_sizes is required, but not both')

    # 类配置，允许额外参数
    class Config:
        extra = "allow"
# 定义一个配置类，用于存储解码器的训练配置信息
class DecoderDataConfig(BaseModel):
    webdataset_base_url: str                     # 存储包含jpg图像的webdataset的路径
    img_embeddings_url: Optional[str] = None     # 存储包含嵌入向量的.npy文件的路径
    text_embeddings_url: Optional[str] = None    # 存储包含嵌入向量的.npy文件的路径
    num_workers: int = 4                         # 工作进程数
    batch_size: int = 64                         # 批量大小
    start_shard: int = 0                         # 起始分片
    end_shard: int = 9999999                     # 结束分片
    shard_width: int = 6                         # 分片宽度
    index_width: int = 4                         # 索引宽度
    splits: TrainSplitConfig                     # 训练数据集拆分配置
    shuffle_train: bool = True                    # 是否对训练数据进行洗牌
    resample_train: bool = False                  # 是否重新采样训练数据
    preprocessing: Dict[str, Any] = {'ToTensor': True}  # 预处理步骤配置

    @property
    def img_preproc(self):
        # 获取图像预处理转换函数
        def _get_transformation(transformation_name, **kwargs):
            if transformation_name == "RandomResizedCrop":
                return T.RandomResizedCrop(**kwargs)
            elif transformation_name == "RandomHorizontalFlip":
                return T.RandomHorizontalFlip()
            elif transformation_name == "ToTensor":
                return T.ToTensor()

        transforms = []
        # 遍历预处理配置，生成转换函数列表
        for transform_name, transform_kwargs_or_bool in self.preprocessing.items():
            transform_kwargs = {} if not isinstance(transform_kwargs_or_bool, dict) else transform_kwargs_or_bool
            transforms.append(_get_transformation(transform_name, **transform_kwargs))
        return T.Compose(transforms)

# 定义一个配置类，用于存储解码器的训练配置信息
class DecoderTrainConfig(BaseModel):
    epochs: int = 20                             # 训练轮数
    lr: SingularOrIterable[float] = 1e-4         # 学习率
    wd: SingularOrIterable[float] = 0.01         # 权重衰减
    warmup_steps: Optional[SingularOrIterable[int]] = None  # 预热步数
    find_unused_parameters: bool = True          # 是否查找未使用的参数
    static_graph: bool = True                    # 是否使用静态图
    max_grad_norm: SingularOrIterable[float] = 0.5  # 最大梯度范数
    save_every_n_samples: int = 100000           # 每隔多少样本保存一次模型
    n_sample_images: int = 6                     # 在采样训练和测试数���集时生成的示例图像数量
    cond_scale: Union[float, List[float]] = 1.0  # 条件缩放
    device: str = 'cuda:0'                       # 设备
    epoch_samples: Optional[int] = None          # 每轮样本数限制
    validation_samples: Optional[int] = None     # 验证集样本数限制
    save_immediately: bool = False                # 是否立即保存
    use_ema: bool = True                         # 是否使用指数移动平均
    ema_beta: float = 0.999                      # 指数移动平均的beta值
    amp: bool = False                            # 是否使用混合精度训练
    unet_training_mask: Optional[ListOrTuple[bool]] = None  # UNet训练掩码

# 定义一个配置类，用于存储解码器的评估配置信息
class DecoderEvaluateConfig(BaseModel):
    n_evaluation_samples: int = 1000             # 评估样本数
    FID: Optional[Dict[str, Any]] = None         # FID评估配置
    IS: Optional[Dict[str, Any]] = None          # IS评估配置
    KID: Optional[Dict[str, Any]] = None         # KID评估配置
    LPIPS: Optional[Dict[str, Any]] = None       # LPIPS评估配置

# 定义一个配置类，用于存储训练解码器的完整配置信息
class TrainDecoderConfig(BaseModel):
    decoder: DecoderConfig                      # 解码器配置
    data: DecoderDataConfig                      # 数据配置
    train: DecoderTrainConfig                    # 训练配置
    evaluate: DecoderEvaluateConfig              # 评估配置
    tracker: TrackerConfig                      # 追踪器配置
    seed: int = 0                                # 随机种子

    @classmethod
    def from_json_path(cls, json_path):
        with open(json_path) as f:
            config = json.load(f)                 # 从JSON文件中加载配置
            print(config)
        return cls(**config)

    @model_validator(mode = 'after')             # 模型验证器
    # 检查是否提供了足够的信息来获取指定用于训练的嵌入
    def check_has_embeddings(self, m):
        # 将self转换为字典
        values = dict(self)

        # 获取data和decoder配置
        data_config, decoder_config = values.get('data'), values.get('decoder')

        # 如果data_config或decoder_config不存在
        if not exists(data_config) or not exists(decoder_config):
            # 则发生了其他错误，应该直接返回values
            return values

        # 检查decoder是否使用文本嵌入
        using_text_embeddings = any([unet.cond_on_text_encodings for unet in decoder_config.unets])
        # 检查是否使用了clip
        using_clip = exists(decoder_config.clip)
        # 获取图片嵌入和文本嵌入的URL
        img_emb_url = data_config.img_embeddings_url
        text_emb_url = data_config.text_embeddings_url

        # 如果使用了文本嵌入
        if using_text_embeddings:
            # 需要一种方法来获取嵌入
            assert using_clip or exists(text_emb_url), 'If text conditioning, either clip or text_embeddings_url must be provided'

        # 如果使用了clip
        if using_clip:
            # 如果同时使用了文本嵌入和图片嵌入的URL
            if using_text_embeddings:
                assert not exists(text_emb_url) or not exists(img_emb_url), 'Loaded clip, but also provided text_embeddings_url and img_embeddings_url. This is redundant. Remove the clip model or the text embeddings'
            else:
                assert not exists(img_emb_url), 'Loaded clip, but also provided img_embeddings_url. This is redundant. Remove the clip model or the embeddings'

        # 如果存在文本嵌入的URL
        if text_emb_url:
            assert using_text_embeddings, "Text embeddings are being loaded, but text embeddings are not being conditioned on. This will slow down the dataloader for no reason."

        # 返回m
        return m
```