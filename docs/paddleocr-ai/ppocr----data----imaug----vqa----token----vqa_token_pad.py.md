# `.\PaddleOCR\ppocr\data\imaug\vqa\token\vqa_token_pad.py`

```
# 版权声明
#
# 版权所有 (c) 2021 PaddlePaddle 作者。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言下的权限和限制。
#
# 导入 paddle 库
import paddle
# 导入 numpy 库，并重命名为 np
import numpy as np

# 定义 VQATokenPad 类
class VQATokenPad(object):
    # 初始化方法
    def __init__(self,
                 max_seq_len=512,
                 pad_to_max_seq_len=True,
                 return_attention_mask=True,
                 return_token_type_ids=True,
                 truncation_strategy="longest_first",
                 return_overflowing_tokens=False,
                 return_special_tokens_mask=False,
                 infer_mode=False,
                 **kwargs):
        # 设置最大序列长度
        self.max_seq_len = max_seq_len
        # 是否填充到最大序列长度
        self.pad_to_max_seq_len = max_seq_len
        # 是否返回注意力掩码
        self.return_attention_mask = return_attention_mask
        # 是否返回 token 类型 ID
        self.return_token_type_ids = return_token_type_ids
        # 截断策略
        self.truncation_strategy = truncation_strategy
        # 是否返回溢出的 token
        self.return_overflowing_tokens = return_overflowing_tokens
        # 是否返回特殊 token 掩码
        self.return_special_tokens_mask = return_special_tokens_mask
        # 填充 token 标签 ID
        self.pad_token_label_id = paddle.nn.CrossEntropyLoss().ignore_index
        # 推断模式
        self.infer_mode = infer_mode
```