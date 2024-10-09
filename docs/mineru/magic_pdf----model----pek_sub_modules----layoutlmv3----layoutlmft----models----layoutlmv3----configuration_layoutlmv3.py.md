# `.\MinerU\magic_pdf\model\pek_sub_modules\layoutlmv3\layoutlmft\models\layoutlmv3\configuration_layoutlmv3.py`

```
# coding=utf-8  # 指定文件的编码为 UTF-8，以支持多种字符
from transformers.models.bert.configuration_bert import BertConfig  # 从 transformers 导入 BertConfig 类
from transformers.utils import logging  # 从 transformers 导入 logging 模块

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

LAYOUTLMV3_PRETRAINED_CONFIG_ARCHIVE_MAP = {  # 定义一个字典，映射模型名称到其配置文件的 URL
    "layoutlmv3-base": "https://huggingface.co/microsoft/layoutlmv3-base/resolve/main/config.json",  # 基础模型的配置文件 URL
    "layoutlmv3-large": "https://huggingface.co/microsoft/layoutlmv3-large/resolve/main/config.json",  # 大型模型的配置文件 URL
    # See all LayoutLMv3 models at https://huggingface.co/models?filter=layoutlmv3  # 备注：查看所有 LayoutLMv3 模型的链接
}

class LayoutLMv3Config(BertConfig):  # 定义 LayoutLMv3Config 类，继承自 BertConfig
    model_type = "layoutlmv3"  # 设置模型类型为 "layoutlmv3"

    def __init__(  # 定义初始化方法
        self,  # 当前实例
        pad_token_id=1,  # 填充标记的 ID
        bos_token_id=0,  # 句子开始标记的 ID
        eos_token_id=2,  # 句子结束标记的 ID
        max_2d_position_embeddings=1024,  # 最大二维位置嵌入的数量
        coordinate_size=None,  # 坐标的大小，默认为 None
        shape_size=None,  # 形状的大小，默认为 None
        has_relative_attention_bias=False,  # 是否具有相对注意力偏差
        rel_pos_bins=32,  # 相对位置的箱数
        max_rel_pos=128,  # 最大相对位置
        has_spatial_attention_bias=False,  # 是否具有空间注意力偏差
        rel_2d_pos_bins=64,  # 二维相对位置的箱数
        max_rel_2d_pos=256,  # 最大二维相对位置
        visual_embed=True,  # 是否使用视觉嵌入
        mim=False,  # 是否启用 MIM
        wpa_task=False,  # 是否启用 WPA 任务
        discrete_vae_weight_path='',  # 离散 VAE 权重路径
        discrete_vae_type='dall-e',  # 离散 VAE 类型
        input_size=224,  # 输入大小
        second_input_size=112,  # 第二输入大小
        device='cuda',  # 设备类型，默认为 CUDA
        **kwargs  # 其他关键字参数
    ):
        """Constructs RobertaConfig."""  # 构造函数的文档字符串
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)  # 调用父类初始化方法
        self.max_2d_position_embeddings = max_2d_position_embeddings  # 设置最大二维位置嵌入
        self.coordinate_size = coordinate_size  # 设置坐标大小
        self.shape_size = shape_size  # 设置形状大小
        self.has_relative_attention_bias = has_relative_attention_bias  # 设置相对注意力偏差标志
        self.rel_pos_bins = rel_pos_bins  # 设置相对位置箱数
        self.max_rel_pos = max_rel_pos  # 设置最大相对位置
        self.has_spatial_attention_bias = has_spatial_attention_bias  # 设置空间注意力偏差标志
        self.rel_2d_pos_bins = rel_2d_pos_bins  # 设置二维相对位置箱数
        self.max_rel_2d_pos = max_rel_2d_pos  # 设置最大二维相对位置
        self.visual_embed = visual_embed  # 设置视觉嵌入标志
        self.mim = mim  # 设置 MIM 标志
        self.wpa_task = wpa_task  # 设置 WPA 任务标志
        self.discrete_vae_weight_path = discrete_vae_weight_path  # 设置离散 VAE 权重路径
        self.discrete_vae_type = discrete_vae_type  # 设置离散 VAE 类型
        self.input_size = input_size  # 设置输入大小
        self.second_input_size = second_input_size  # 设置第二输入大小
        self.device = device  # 设置设备类型
```