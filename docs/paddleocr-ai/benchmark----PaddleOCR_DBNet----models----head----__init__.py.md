# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\models\head\__init__.py`

```py
# 设置文件编码为 UTF-8
# 设置时间为 2020 年 6 月 5 日 11 点 35 分
# 设置作者为 zhoujun
# 导入 DBHead 模块
from .DBHead import DBHead

# 定义模块中可导出的内容
__all__ = ['build_head']
# 支持的头部类型列表
support_head = ['DBHead']

# 构建头部函数，根据头部名称和参数构建对应的头部对象
def build_head(head_name, **kwargs):
    # 确保头部名称在支持的头部列表中
    assert head_name in support_head, f'all support head is {support_head}'
    # 根据头部名称动态创建对应的头部对象
    head = eval(head_name)(**kwargs)
    # 返回创建的头部对象
    return head
```