# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\post_processing\__init__.py`

```
# 设置文件编码为 UTF-8
# 设置时间为 2019年12月5日15时17分
# 设置作者为周军

# 从seg_detector_representer模块中导入SegDetectorRepresenter类
from .seg_detector_representer import SegDetectorRepresenter

# 定义一个函数，根据配置获取后处理对象
def get_post_processing(config):
    # 尝试根据配置中的'type'键值对应的类实例化对象，并传入配置中的'args'参数
    try:
        cls = eval(config['type'])(**config['args'])
        # 返回实例化后的对象
        return cls
    # 如果出现异常，则返回None
    except:
        return None
```