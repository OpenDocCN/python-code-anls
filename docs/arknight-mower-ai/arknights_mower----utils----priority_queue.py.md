# `arknights-mower\arknights_mower\utils\priority_queue.py`

```py
# 导入 heapq 模块，用于实现优先队列
import heapq

# 定义优先队列类
class PriorityQueue(object):
    """
    基于 heapq 实现的优先队列
    """

    # 初始化方法，创建一个空的队列
    def __init__(self):
        self.queue = []

    # 入队方法，使用 heapq 的 heappush 方法将数据加入队列
    def push(self, data):
        heapq.heappush(self.queue, data)

    # 出队方法，如果队列为空则返回 None，否则使用 heapq 的 heappop 方法弹出队列中的最小值
    def pop(self):
        if len(self.queue) == 0:
            return None
        return heapq.heappop(self.queue)
```