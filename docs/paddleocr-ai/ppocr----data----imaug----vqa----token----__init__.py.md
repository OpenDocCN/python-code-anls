# `.\PaddleOCR\ppocr\data\imaug\vqa\token\__init__.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 导入 VQASerTokenChunk 和 VQAReTokenChunk 类
from .vqa_token_chunk import VQASerTokenChunk, VQAReTokenChunk
# 导入 VQATokenPad 类
from .vqa_token_pad import VQATokenPad
# 导入 VQAReTokenRelation 类
from .vqa_token_relation import VQAReTokenRelation
# 导入 TensorizeEntitiesRelations 类
from .vqa_re_convert import TensorizeEntitiesRelations
```