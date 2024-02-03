# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\models\neck\__init__.py`

```
# 设置文件编码为 UTF-8
# 设置时间为 2020年6月5日11点34分
# 设置作者为 zhoujun
from .FPN import FPN

# 导出所有的 neck 模块
__all__ = ['build_neck']
# 支持的 neck 模块列表
support_neck = ['FPN']

# 构建 neck 模块
def build_neck(neck_name, **kwargs):
    # 断言所选的 neck 模块在支持的列表中
    assert neck_name in support_neck, f'all support neck is {support_neck}'
    # 使用 eval 函数根据 neck 名称创建 neck 对象
    neck = eval(neck_name)(**kwargs)
    # 返回创建的 neck 对象
    return neck
```