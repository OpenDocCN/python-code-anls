# `.\YOLO-World\yolo_world\models\backbones\mm_backbone.py`

```
# 导入所需的库
import itertools
from typing import List, Sequence, Tuple
import torch
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmdet.utils import OptMultiConfig, ConfigType
from transformers import (AutoTokenizer, AutoModel, CLIPTextConfig)
from transformers import CLIPTextModelWithProjection as CLIPTP

# 注册模型类到模型注册表中
@MODELS.register_module()
class HuggingVisionBackbone(BaseModule):
    # 初始化函数
    def __init__(self,
                 model_name: str,
                 out_indices: Sequence[int] = (0, 1, 2, 3),
                 norm_eval: bool = True,
                 frozen_modules: Sequence[str] = (),
                 init_cfg: OptMultiConfig = None) -> None:

        # 调用父类的初始化函数
        super().__init__(init_cfg=init_cfg)

        # 初始化属性
        self.norm_eval = norm_eval
        self.frozen_modules = frozen_modules
        self.model = AutoModel.from_pretrained(model_name)

        # 冻结指定模块
        self._freeze_modules()

    # 前向传播函数
    def forward(self, image: Tensor) -> Tuple[Tensor]:
        # 获取图像的编码字典
        encoded_dict = self.image_model(pixel_values=image,
                                        output_hidden_states=True)
        hidden_states = encoded_dict.hidden_states
        img_feats = encoded_dict.get('reshaped_hidden_states', hidden_states)
        img_feats = [img_feats[i] for i in self.image_out_indices]
        return tuple(img_feats)

    # 冻结指定模块的参数
    def _freeze_modules(self):
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break
    # 定义一个训练方法，设置模式为训练或评估
    def train(self, mode=True):
        # 调用父类的train方法，设置当前模型为训练或评估模式
        super().train(mode)
        # 冻结模型的参数
        self._freeze_modules()
        # 如果是训练模式并且开启了norm_eval
        if mode and self.norm_eval:
            # 遍历模型的所有子模块
            for m in self.modules():
                # 如果当前模块是BatchNorm类型
                if isinstance(m, _BatchNorm):
                    # 将当前BatchNorm模块设置为评估模式
                    m.eval()
# 注册 HuggingCLIPLanguageBackbone 类到 MODELS 模块
@MODELS.register_module()
class HuggingCLIPLanguageBackbone(BaseModule):
    # 初始化方法，接受模型名称、冻结模块、dropout 等参数
    def __init__(self,
                 model_name: str,
                 frozen_modules: Sequence[str] = (),
                 dropout: float = 0.0,
                 training_use_cache: bool = False,
                 init_cfg: OptMultiConfig = None) -> None:
        # 调用父类的初始化方法
        super().__init__(init_cfg=init_cfg)
        
        # 设置冻结模块和是否使用缓存的属性
        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        # 根据模型名称创建 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 根据模型名称和 dropout 创建 CLIPTextConfig 对象
        clip_config = CLIPTextConfig.from_pretrained(model_name, attention_dropout=dropout)
        # 根据模型名称和配置创建 CLIPTP 模型
        self.model = CLIPTP.from_pretrained(model_name, config=clip_config)
        # 冻结指定模块
        self._freeze_modules()

    # 前向传播方法，用于缓存文本数据
    def forward_cache(self, text: List[List[str]]) -> Tensor:
        # 如果不存在缓存，则调用 forward_text 方法生成缓存
        if not hasattr(self, "cache"):
            self.cache = self.forward_text(text)
        return self.cache

    # 前向传播方法，根据训练状态选择使用缓存或者重新计算
    def forward(self, text: List[List[str]]) -> Tensor:
        # 如果处于训练状态，则重新计算文本数据
        if self.training:
            return self.forward_text(text)
        # 否则使用缓存数据
        else:
            return self.forward_cache(text)

    # 前向传播方法，用于处理文本数据并返回处理后的数据
    def forward_tokenizer(self, texts):
        # 如果不存在文本数据，则处理文本数据
        if not hasattr(self, 'text'):
            # 将多个文本列表合并成一个文本列表
            text = list(itertools.chain(*texts))
            # 使用 tokenizer 处理文本数据并转换为 PyTorch 张量
            text = self.tokenizer(text=text, return_tensors='pt', padding=True)
            # 将处理后的文本数据保存到对象属性中
            self.text = text.to(device=self.model.device)
        return self.text
    # 前向传播文本数据，返回文本特征张量
    def forward_text(self, text: List[List[str]]) -> Tensor:
        # 计算每个批次中的序列数量
        num_per_batch = [len(t) for t in text]
        # 断言每个批次中的序列数量相等
        assert max(num_per_batch) == min(num_per_batch), (
            'number of sequences not equal in batch')
        # 将文本列表展开为一维列表
        text = list(itertools.chain(*text))
        # 使用分词器对文本进行处理
        text = self.tokenizer(text=text, return_tensors='pt', padding=True)
        # 将文本数据移动到指定设备上
        text = text.to(device=self.model.device)
        # 获取文本输出
        txt_outputs = self.model(**text)
        # 获取文本特征
        txt_feats = txt_outputs.text_embeds
        # 对文本特征进行归一化处理
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        # 重新调整文本特征的形状
        txt_feats = txt_feats.reshape(-1, num_per_batch[0], txt_feats.shape[-1])
        return txt_feats

    # 冻结指定模块
    def _freeze_modules(self):

        if len(self.frozen_modules) == 0:
            # 如果没有需要冻结的模块，则直接返回
            return
        if self.frozen_modules[0] == "all":
            # 如果需要冻结所有模块，则将所有模块设为评估模式并冻结参数
            self.model.eval()
            for _, module in self.model.named_modules():
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            return
        # 遍历模型的所有模块，冻结指定的模块
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    # 训练模型，设置模式并冻结指定模块
    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()
# 注册PseudoLanguageBackbone类到MODELS模块
@MODELS.register_module()
class PseudoLanguageBackbone(BaseModule):
    """Pseudo Language Backbone
    Args:
        text_embed_path (str): path to the text embedding file
    """
    # 初始化函数，接受文本嵌入文件路径和初始化配置
    def __init__(self,
                 text_embed_path: str = "",
                 test_embed_path: str = None,
                 init_cfg: OptMultiConfig = None):
        # 调用父类的初始化函数
        super().__init__(init_cfg)
        # 加载文本嵌入文件，存储为{text:embed}形式
        self.text_embed = torch.load(text_embed_path, map_location='cpu')
        # 如果测试嵌入文件路径为空，则使用文本嵌入文件
        if test_embed_path is None:
            self.test_embed = self.text_embed
        else:
            self.test_embed = torch.load(test_embed_path)
        # 注册缓冲区
        self.register_buffer("buff", torch.zeros([
            1,
        ]))

    # 缓存前向传播结果
    def forward_cache(self, text: List[List[str]]) -> Tensor:
        if not hasattr(self, "cache"):
            self.cache = self.forward_text(text)
        return self.cache

    # 前向传播函数
    def forward(self, text: List[List[str]]) -> Tensor:
        if self.training:
            return self.forward_text(text)
        else:
            return self.forward_cache(text)

    # 文本前向传播函数
    def forward_text(self, text: List[List[str]]) -> Tensor:
        # 计算每个批次的序列数量
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), (
            'number of sequences not equal in batch')
        # 将文本列表展平
        text = list(itertools.chain(*text))
        # 根据训练状态选择文本嵌入字典
        if self.training:
            text_embed_dict = self.text_embed
        else:
            text_embed_dict = self.test_embed
        # 根据文本获取对应的嵌入向量
        text_embeds = torch.stack(
            [text_embed_dict[x.split("/")[0]] for x in text])
        # 设置梯度为False，转换为浮点型
        text_embeds = text_embeds.to(
            self.buff.device).requires_grad_(False).float()
        # 重塑嵌入向量形状
        text_embeds = text_embeds.reshape(-1, num_per_batch[0],
                                          text_embeds.shape[-1])
        return text_embeds


# 注册MultiModalYOLOBackbone类到MODELS模块
@MODELS.register_module()
class MultiModalYOLOBackbone(BaseModule):
    # 初始化函数，接受图像模型、文本模型、冻结阶段和初始化配置作为参数
    def __init__(self,
                 image_model: ConfigType,
                 text_model: ConfigType,
                 frozen_stages: int = -1,
                 init_cfg: OptMultiConfig = None) -> None:
        
        # 调用父类的初始化函数
        super().__init__(init_cfg)
        
        # 使用传入的配置构建图像模型和文本模型
        self.image_model = MODELS.build(image_model)
        self.text_model = MODELS.build(text_model)
        self.frozen_stages = frozen_stages
        # 冻结指定阶段的参数
        self._freeze_stages()

    # 冻结指定阶段的参数
    def _freeze_stages(self):
        """Freeze the parameters of the specified stage so that they are no
        longer updated."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                # 获取指定阶段的模型层
                m = getattr(self.image_model, self.image_model.layers[i])
                # 将模型设置为评估模式
                m.eval()
                # 冻结模型参数
                for param in m.parameters():
                    param.requires_grad = False

    # 将模型转换为训练模式，同时保持归一化层冻结
    def train(self, mode: bool = True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        # 调用父类的训练函数
        super().train(mode)
        # 冻结指定阶段的参数
        self._freeze_stages()

    # 前向传播函数，接受图像和文本作为输入，返回图像特征和文本特征
    def forward(self, image: Tensor,
                text: List[List[str]]) -> Tuple[Tuple[Tensor], Tensor]:
        # 获取图像特征
        img_feats = self.image_model(image)
        # 获取文本特征
        txt_feats = self.text_model(text)
        # 返回图像特征和文本特征
        return img_feats, txt_feats
```