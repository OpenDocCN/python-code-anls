# `.\PaddleOCR\ppocr\modeling\architectures\base_model.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按"原样"分发
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 paddle.nn 模块
from paddle import nn
# 导入 ppocr.modeling.transforms 模块中的 build_transform 函数
from ppocr.modeling.transforms import build_transform
# 导入 ppocr.modeling.backbones 模块中的 build_backbone 函数
from ppocr.modeling.backbones import build_backbone
# 导入 ppocr.modeling.necks 模块中的 build_neck 函数
from ppocr.modeling.necks import build_neck
# 导入 ppocr.modeling.heads 模块中的 build_head 函数
from ppocr.modeling.heads import build_head

# 定义 BaseModel 类
__all__ = ['BaseModel']


class BaseModel(nn.Layer):
    # 前向传播函数，接受输入 x 和数据 data
    def forward(self, x, data=None):

        # 初始化输出字典 y
        y = dict()
        # 如果使用变换，则对输入进行变换
        if self.use_transform:
            x = self.transform(x)
        # 如果使用主干网络，则对输入进行主干网络处理
        if self.use_backbone:
            x = self.backbone(x)
        # 如果 x 是字典类型，则更新输出字典 y
        if isinstance(x, dict):
            y.update(x)
        else:
            y["backbone_out"] = x
        # 初始化最终输出名称为 "backbone_out"
        final_name = "backbone_out"
        # 如果使用颈部网络，则对输入进行颈部网络处理
        if self.use_neck:
            x = self.neck(x)
            # 如果 x 是字典类型，则更新输出字典 y
            if isinstance(x, dict):
                y.update(x)
            else:
                y["neck_out"] = x
            # 更新最终输出名称为 "neck_out"
            final_name = "neck_out"
        # 如果使用头部网络，则对输入进行头部网络处理
        if self.use_head:
            x = self.head(x, targets=data)
            # 对于多头网络，保存 CTC 颈部输出用于 UDML
            if isinstance(x, dict) and 'ctc_neck' in x.keys():
                y["neck_out"] = x["ctc_neck"]
                y["head_out"] = x
            elif isinstance(x, dict):
                y.update(x)
            else:
                y["head_out"] = x
            # 更新最终输出名称为 "head_out"
            final_name = "head_out"
        # 如果需要返回所有特征
        if self.return_all_feats:
            # 如果处于训练状态，则返回输出字典 y
            if self.training:
                return y
            # 如果 x 是字典类型，则返回 x
            elif isinstance(x, dict):
                return x
            # 否则返回包含最终输出名称的字典
            else:
                return {final_name: x}
        else:
            return x
```