# `.\VisualGLM-6B\model\visualglm.py`

```py
# 导入 torch 库
import torch
# 从 sat.model.official 模块中导入 ChatGLMModel 类
from sat.model.official import ChatGLMModel
# 从 sat.model.base_model 模块中导入 BaseMixin 类
from sat.model.base_model import BaseMixin
# 从 copy 模块中导入 deepcopy 函数
from copy import deepcopy
# 导入 json 库
import json
# 从当前目录下的 blip2 模块中导入 BLIP2 类
from .blip2 import BLIP2
# 从 sat.resources.urls 模块中导入 MODEL_URLS 字典
from sat.resources.urls import MODEL_URLS
# 将 'visualglm-6b' 对应的值设置为 'r2://visualglm-6b.zip'
MODEL_URLS['visualglm-6b'] = 'r2://visualglm-6b.zip'

# 定义 ImageMixin 类，继承自 BaseMixin 类
class ImageMixin(BaseMixin):
    # 初始化方法
    def __init__(self, args):
        # 调用父类的初始化方法
        super().__init__()
        # 深拷贝参数 args
        self.args = deepcopy(args)
        # 如果 args 中有 'model_parallel_size' 属性
        if hasattr(args, 'model_parallel_size'):
            # 将 'model_parallel_size' 分别赋值给 args.eva_args 和 args.qformer_args
            args.eva_args['model_parallel_size'] = args.model_parallel_size
            args.qformer_args['model_parallel_size'] = args.model_parallel_size
        # 创建 BLIP2 模型对象
        self.model = BLIP2(args.eva_args, args.qformer_args)

    # 定义 word_embedding_forward 方法
    def word_embedding_forward(self, input_ids, output_cross_layer, **kw_args):
        # 如果 kw_args 中的 "pre_image" 大于 input_ids 的列数 或者 "image" 为 None
        if kw_args["pre_image"] > input_ids.shape[1] or kw_args.get("image", None) is None:
            # 返回 transformer 的 word_embeddings 方法对 input_ids 的结果
            return self.transformer.word_embeddings(input_ids)
        # 调用 self.model 方法，传入 kw_args，得到 image_emb
        image_emb = self.model(**kw_args)
        # 将 input_ids 按照指定位置分割为 pre_id, pads, post_id
        pre_id, pads, post_id = torch.tensor_split(input_ids, [kw_args["pre_image"], kw_args["pre_image"]+self.args.image_length], dim=1)
        # 分别对 pre_id 和 post_id 进行 word_embeddings，得到 pre_txt_emb 和 post_txt_emb
        pre_txt_emb = self.transformer.word_embeddings(pre_id)
        post_txt_emb = self.transformer.word_embeddings(post_id)
        # 拼接 pre_txt_emb, image_emb, post_txt_emb，按列拼接
        return torch.cat([pre_txt_emb, image_emb, post_txt_emb], dim=1)

# 定义 VisualGLMModel 类，继承自 ChatGLMModel 类
class VisualGLMModel(ChatGLMModel):
    # 初始化方法
    def __init__(self, args, transformer=None, **kwargs):
        # 调用父类的初始化方法
        super().__init__(args, transformer=transformer, **kwargs)
        # 将 args 中的 image_length 赋值给 self.image_length
        self.image_length = args.image_length
        # 添加名为 "eva" 的 ImageMixin 实例到模型中
        self.add_mixin("eva", ImageMixin(args))

    # 类方法，用于添加特定于模型的参数
    @classmethod
    def add_model_specific_args(cls, parser):
        # 添加名为 'VisualGLM' 的参数组，描述为 'VisualGLM Configurations'
        group = parser.add_argument_group('VisualGLM', 'VisualGLM Configurations')
        # 添加类型为 int，默认值为 32 的参数 '--image_length'
        group.add_argument('--image_length', type=int, default=32)
        # 添加类型为 json.loads，默认值为空字典的参数 '--eva_args'
        group.add_argument('--eva_args', type=json.loads, default={})
        # 添加类型为 json.loads，默认值为空字典的参数 '--qformer_args'
        group.add_argument('--qformer_args', type=json.loads, default={})
        # 返回父类的 add_model_specific_args 方法
        return super().add_model_specific_args(parser)
```