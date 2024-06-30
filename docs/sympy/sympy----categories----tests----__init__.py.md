# `D:\src\scipysrc\sympy\sympy\categories\tests\__init__.py`

```
# 定义一个名为 `parse_config` 的函数，接受一个参数 `config_file`
def parse_config(config_file):
    # 打开配置文件，读取所有内容并存储在 `config_lines` 列表中
    config_lines = open(config_file).readlines()
    
    # 初始化一个空的配置字典 `config_dict`
    config_dict = {}
    
    # 遍历 `config_lines` 列表中的每一行
    for line in config_lines:
        # 去除每行两端的空白符并判断是否为空行
        if line.strip():  # 如果不是空行
            # 分割每行的键值对，并移除两端的空白符
            key, value = map(str.strip, line.split('=', 1))
            # 将键值对添加到 `config_dict` 字典中
            config_dict[key] = value
    
    # 返回解析后的配置字典 `config_dict`
    return config_dict
```