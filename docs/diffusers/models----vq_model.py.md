# `.\diffusers\models\vq_model.py`

```py
# 版权声明，表明版权归 HuggingFace 团队所有
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 按照 Apache 许可证 2.0 版本进行许可
# Licensed under the Apache License, Version 2.0 (the "License");
# 你只能在符合许可证的情况下使用此文件
# you may not use this file except in compliance with the License.
# 你可以在以下网址获取许可证副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件以 "原样" 基础分发，不附带任何明示或暗示的担保或条件
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以了解特定的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.
# 从 utils 模块导入 deprecate 函数
from ..utils import deprecate
# 从 autoencoders.vq_model 模块导入 VQEncoderOutput 和 VQModel 类
from .autoencoders.vq_model import VQEncoderOutput, VQModel


# 定义 VQEncoderOutput 类，继承自 VQEncoderOutput
class VQEncoderOutput(VQEncoderOutput):
    # 初始化方法，接收可变参数
    def __init__(self, *args, **kwargs):
        # 定义过时警告信息，提示用户导入路径已过时
        deprecation_message = "Importing `VQEncoderOutput` from `diffusers.models.vq_model` is deprecated and this will be removed in a future version. Please use `from diffusers.models.autoencoders.vq_model import VQEncoderOutput`, instead."
        # 调用 deprecate 函数，记录过时警告
        deprecate("VQEncoderOutput", "0.31", deprecation_message)
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)


# 定义 VQModel 类，继承自 VQModel
class VQModel(VQModel):
    # 初始化方法，接收可变参数
    def __init__(self, *args, **kwargs):
        # 定义过时警告信息，提示用户导入路径已过时
        deprecation_message = "Importing `VQModel` from `diffusers.models.vq_model` is deprecated and this will be removed in a future version. Please use `from diffusers.models.autoencoders.vq_model import VQModel`, instead."
        # 调用 deprecate 函数，记录过时警告
        deprecate("VQModel", "0.31", deprecation_message)
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
```