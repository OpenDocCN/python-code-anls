# `so-vits-svc\vdecoder\nsf_hifigan\env.py`

```
# 导入 os 模块
import os
# 导入 shutil 模块
import shutil

# 创建一个继承自 dict 的类，用于将字典的键作为属性访问
class AttrDict(dict):
    # 初始化方法，接受任意数量的位置参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super(AttrDict, self).__init__(*args, **kwargs)
        # 将当前对象的 __dict__ 属性指向自身，实现将字典的键作为属性访问

# 构建环境的函数，接受配置、配置名称和路径作为参数
def build_env(config, config_name, path):
    # 拼接配置文件路径
    t_path = os.path.join(path, config_name)
    # 如果配置文件不在目标路径下
    if config != t_path:
        # 创建目标路径，如果路径已存在则不报错
        os.makedirs(path, exist_ok=True)
        # 复制配置文件到目标路径下，并命名为配置名称
        shutil.copyfile(config, os.path.join(path, config_name))
```