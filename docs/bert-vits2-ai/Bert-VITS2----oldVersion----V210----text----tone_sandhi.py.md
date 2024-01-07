# `Bert-VITS2\oldVersion\V210\text\tone_sandhi.py`

```

# 2021 PaddlePaddle作者版权所有。
#
# 根据Apache许可证2.0版（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“按原样”分发，没有任何明示或暗示的保证或条件。
# 请参阅许可证以了解特定语言管理权限和
# 许可证下的限制。
from typing import List  # 导入List类型提示
from typing import Tuple  # 导入Tuple类型提示

import jieba  # 导入jieba分词库
from pypinyin import lazy_pinyin  # 从pypinyin库中导入lazy_pinyin函数
from pypinyin import Style  # 从pypinyin库中导入Style类

```