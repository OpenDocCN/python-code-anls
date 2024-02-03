# `.\PaddleOCR\ppocr\utils\stats.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权使用本文件
# 除非符合许可证的规定，否则不得使用本文件
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按“原样”分发，不附带任何担保或条件，无论是明示还是暗示的
# 请查看许可证以获取特定语言的权限和限制

# 导入所需的库
import collections
import numpy as np
import datetime

# 定义可以导出的类
__all__ = ['TrainingStats', 'Time']

# 定义 SmoothedValue 类，用于跟踪一系列值并提供对窗口或全局系列平均值的访问
class SmoothedValue(object):
    def __init__(self, window_size):
        self.deque = collections.deque(maxlen=window_size)

    def add_value(self, value):
        self.deque.append(value)

    def get_median_value(self):
        return np.median(self.deque)

# 定义 Time 函数，返回当前时间的字符串表示
def Time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

# 定义 TrainingStats 类，用于跟踪平滑的损失和指标值
class TrainingStats(object):
    def __init__(self, window_size, stats_keys):
        self.window_size = window_size
        self.smoothed_losses_and_metrics = {
            key: SmoothedValue(window_size)
            for key in stats_keys
        }

    def update(self, stats):
        for k, v in stats.items():
            if k not in self.smoothed_losses_and_metrics:
                self.smoothed_losses_and_metrics[k] = SmoothedValue(
                    self.window_size)
            self.smoothed_losses_and_metrics[k].add_value(v)
    # 获取统计信息，以有序字典的形式返回
    def get(self, extras=None):
        # 创建一个空的有序字典
        stats = collections.OrderedDict()
        # 如果有额外信息传入
        if extras:
            # 遍历额外信息字典，将其添加到统计信息字典中
            for k, v in extras.items():
                stats[k] = v
        # 遍历平滑后的损失和指标字典
        for k, v in self.smoothed_losses_and_metrics.items():
            # 将每个指标的中位数值取到小数点后6位，添加到统计信息字典中
            stats[k] = round(v.get_median_value(), 6)

        # 返回统计信息字典
        return stats

    # 记录日志信息
    def log(self, extras=None):
        # 获取统计信息字典
        d = self.get(extras)
        # 创建一个空列表用于存储字符串
        strs = []
        # 遍历统计信息字典
        for k, v in d.items():
            # 格式化每个键值对的字符串，保留6位小数，添加到字符串列表中
            strs.append('{}: {:x<6f}'.format(k, v))
        # 将字符串列表中的元素用逗号连接成一个字符串
        strs = ', '.join(strs)
        # 返回拼接后的字符串
        return strs
```