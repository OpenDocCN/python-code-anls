# `.\PaddleOCR\StyleText\tools\synth_dataset.py`

```
#   版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权使用该文件；
# 除非符合许可证的规定，否则不得使用该文件；
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于“按原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 导入必要的库
import os
import sys

# 获取当前文件所在目录的绝对路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在目录添加到系统路径中
sys.path.append(__dir__)
# 将当前文件所在目录的上级目录添加到系统路径中
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

# 导入自定义模块 DatasetSynthesiser 中的类 DatasetSynthesiser
from engine.synthesisers import DatasetSynthesiser

# 定义函数 synth_dataset
def synth_dataset():
    # 创建 DatasetSynthesiser 类的实例
    dataset_synthesiser = DatasetSynthesiser()
    # 调用实例的 synth_dataset 方法
    dataset_synthesiser.synth_dataset()

# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 调用 synth_dataset 函数
    synth_dataset()
```