# `.\lucidrains\imagen-pytorch\imagen_pytorch\configs.py`

```
# 导入必要的模块和类
from pydantic import BaseModel, model_validator
from typing import List, Optional, Union, Tuple
from enum import Enum

# 导入自定义模块中的类和函数
from imagen_pytorch.imagen_pytorch import Imagen, Unet, Unet3D, NullUnet
from imagen_pytorch.trainer import ImagenTrainer
from imagen_pytorch.elucidated_imagen import ElucidatedImagen
from imagen_pytorch.t5 import DEFAULT_T5_NAME, get_encoded_dim

# 定义一些辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 定义一个接受内部类型的列表或元组
def ListOrTuple(inner_type):
    return Union[List[inner_type], Tuple[inner_type]]

# 定义一个接受内部类型的单个值或列表
def SingleOrList(inner_type):
    return Union[inner_type, ListOrTuple(inner_type)]

# 噪声调度

# 定义一个枚举类，表示噪声调度的类型
class NoiseSchedule(Enum):
    cosine = 'cosine'
    linear = 'linear'

# 允许额外字段的基础模型类
class AllowExtraBaseModel(BaseModel):
    class Config:
        extra = "allow"
        use_enum_values = True

# imagen pydantic 类

# 空 Unet 配置类
class NullUnetConfig(BaseModel):
    is_null:            bool

    def create(self):
        return NullUnet()

# Unet 配置类
class UnetConfig(AllowExtraBaseModel):
    dim:                int
    dim_mults:          ListOrTuple(int)
    text_embed_dim:     int = get_encoded_dim(DEFAULT_T5_NAME)
    cond_dim:           Optional[int] = None
    channels:           int = 3
    attn_dim_head:      int = 32
    attn_heads:         int = 16

    def create(self):
        return Unet(**self.dict())

# Unet3D 配置类
class Unet3DConfig(AllowExtraBaseModel):
    dim:                int
    dim_mults:          ListOrTuple(int)
    text_embed_dim:     int = get_encoded_dim(DEFAULT_T5_NAME)
    cond_dim:           Optional[int] = None
    channels:           int = 3
    attn_dim_head:      int = 32
    attn_heads:         int = 16

    def create(self):
        return Unet3D(**self.dict())

# Imagen 配置类
class ImagenConfig(AllowExtraBaseModel):
    unets:                  ListOrTuple(Union[UnetConfig, Unet3DConfig, NullUnetConfig])
    image_sizes:            ListOrTuple(int)
    video:                  bool = False
    timesteps:              SingleOrList(int) = 1000
    noise_schedules:        SingleOrList(NoiseSchedule) = 'cosine'
    text_encoder_name:      str = DEFAULT_T5_NAME
    channels:               int = 3
    loss_type:              str = 'l2'
    cond_drop_prob:         float = 0.5

    @model_validator(mode="after")
    def check_image_sizes(self):
        if len(self.image_sizes) != len(self.unets):
            raise ValueError(f'image sizes length {len(self.image_sizes)} must be equivalent to the number of unets {len(self.unets)}')
        return self

    def create(self):
        decoder_kwargs = self.dict()
        unets_kwargs = decoder_kwargs.pop('unets')
        is_video = decoder_kwargs.pop('video', False)

        unets = []

        for unet, unet_kwargs in zip(self.unets, unets_kwargs):
            if isinstance(unet, NullUnetConfig):
                unet_klass = NullUnet
            elif is_video:
                unet_klass = Unet3D
            else:
                unet_klass = Unet

            unets.append(unet_klass(**unet_kwargs))

        imagen = Imagen(unets, **decoder_kwargs)

        imagen._config = self.dict().copy()
        return imagen

# ElucidatedImagen 配置类
class ElucidatedImagenConfig(AllowExtraBaseModel):
    unets:                  ListOrTuple(Union[UnetConfig, Unet3DConfig, NullUnetConfig])
    image_sizes:            ListOrTuple(int)
    video:                  bool = False
    text_encoder_name:      str = DEFAULT_T5_NAME
    channels:               int = 3
    cond_drop_prob:         float = 0.5
    num_sample_steps:       SingleOrList(int) = 32
    sigma_min:              SingleOrList(float) = 0.002
    sigma_max:              SingleOrList(int) = 80
    sigma_data:             SingleOrList(float) = 0.5
    rho:                    SingleOrList(int) = 7
    P_mean:                 SingleOrList(float) = -1.2
    P_std:                  SingleOrList(float) = 1.2
    S_churn:                SingleOrList(int) = 80
    S_tmin:                 SingleOrList(float) = 0.05
    S_tmax:                 SingleOrList(int) = 50
    # 定义 S_tmax 变量，类型为 int 或 int 列表，默认值为 50
    S_noise:                SingleOrList(float) = 1.003
    # 定义 S_noise 变量，类型为 float 或 float 列表，默认值为 1.003

    @model_validator(mode="after")
    # 使用 model_validator 装饰器，指定 mode 参数为 "after"
    def check_image_sizes(self):
        # 检查图像大小是否与 unets 数量相等
        if len(self.image_sizes) != len(self.unets):
            raise ValueError(f'image sizes length {len(self.image_sizes)} must be equivalent to the number of unets {len(self.unets)}')
        return self
        # 返回当前对象

    def create(self):
        # 创建方法 create
        decoder_kwargs = self.dict()
        # 获取当前对象的字典形式
        unets_kwargs = decoder_kwargs.pop('unets')
        # 从字典中弹出键为 'unets' 的值，并赋给 unets_kwargs
        is_video = decoder_kwargs.pop('video', False)
        # 从字典中弹出键为 'video' 的值，如果不存在则默认为 False

        unet_klass = Unet3D if is_video else Unet
        # 根据 is_video 的值选择 Unet3D 或 Unet 类

        unets = []

        for unet, unet_kwargs in zip(self.unets, unets_kwargs):
            # 遍历 self.unets 和 unets_kwargs
            if isinstance(unet, NullUnetConfig):
                unet_klass = NullUnet
            elif is_video:
                unet_klass = Unet3D
            else:
                unet_klass = Unet

            unets.append(unet_klass(**unet_kwargs))
            # 根据条件选择 Unet 类型，并将实例添加到 unets 列表中

        imagen = ElucidatedImagen(unets, **decoder_kwargs)
        # 创建 ElucidatedImagen 实例，传入 unets 和 decoder_kwargs

        imagen._config = self.dict().copy()
        # 将当前对象的字典形式复制给 imagen 的 _config 属性
        return imagen
        # 返回 imagen 实例
# 定义一个配置类 ImagenTrainerConfig，继承自 AllowExtraBaseModel
class ImagenTrainerConfig(AllowExtraBaseModel):
    # 定义属性 imagen，类型为字典
    imagen:                 dict
    # 定义属性 elucidated，默认值为 False
    elucidated:             bool = False
    # 定义属性 video，默认值为 False
    video:                  bool = False
    # 定义属性 use_ema，默认值为 True
    use_ema:                bool = True
    # 定义属性 lr，默认值为 1e-4
    lr:                     SingleOrList(float) = 1e-4
    # 定义属性 eps，默认值为 1e-8
    eps:                    SingleOrList(float) = 1e-8
    # 定义属性 beta1，默认值为 0.9
    beta1:                  float = 0.9
    # 定义属性 beta2，默认值为 0.99
    beta2:                  float = 0.99
    # 定义属性 max_grad_norm，默认值为 None
    max_grad_norm:          Optional[float] = None
    # 定义属性 group_wd_params，默认值为 True
    group_wd_params:        bool = True
    # 定义属性 warmup_steps，默认值为 None
    warmup_steps:           SingleOrList(Optional[int]) = None
    # 定义属性 cosine_decay_max_steps，默认值为 None
    cosine_decay_max_steps: SingleOrList(Optional[int]) = None

    # 定义一个方法 create，用于创建 ImagenTrainer 对象
    def create(self):
        # 将配置参数转换为字典
        trainer_kwargs = self.dict()

        # 弹出并获取 imagen 属性的值
        imagen_config = trainer_kwargs.pop('imagen')
        # 弹出并获取 elucidated 属性的值
        elucidated = trainer_kwargs.pop('elucidated')

        # 根据 elucidated 属性的值选择不同的配置类
        imagen_config_klass = ElucidatedImagenConfig if elucidated else ImagenConfig
        # 创建 imagen 对象，根据 video 属性的值选择不同的配置
        imagen = imagen_config_klass(**{**imagen_config, 'video': video}).create()

        # 返回创建的 ImagenTrainer 对象
        return ImagenTrainer(imagen, **trainer_kwargs)
```