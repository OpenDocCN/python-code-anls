# `KubiScan\engine\capabilities\capabilities.py`

```
# 定义一个包含能力名称和对应数值的字典
caps_list = {
  "CHOWN": 1,  # 更改文件所有者
  "DAC_OVERRIDE": 2,  # 忽略文件权限
  "DAC_READ_SEARCH": 3,  # 忽略目录读取权限和搜索权限
  # ... 其他能力
}

# 定义一个包含默认能力名称和对应数值的字典
default_caps = {
  "CAP_CHOWN": 1,  # 更改文件所有者
  "DAC_OVERRIDE": 2,  # 忽略文件权限
  # ... 其他默认能力
}

# 定义一个包含危险能力名称和对应数值的字典
dangerous_caps = {
  "DAC_READ_SEARCH": 3,  # 忽略目录读取权限和搜索权限
  "LINUX_IMMUTABLE": 10,  # 设置文件不可修改
  # ... 其他危险能力
}

# 下面的代码是注释掉的，不会执行
# indexes = get_indexes_with_one(0x10)
# indexes = get_indexes_with_one(0x3fffffffff)
# print_decoded_capabilities(indexes)
```