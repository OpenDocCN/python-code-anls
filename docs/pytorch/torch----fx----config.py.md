# `.\pytorch\torch\fx\config.py`

```
# 是否禁用编译过程中的进度显示
# 如果在这里导入 dynamo 配置，会导致循环导入，因此需要添加一个新的配置
disable_progress = True

# 如果设置为 True，还会在每个编译步骤中显示节点名称，对于小型模型很有帮助，但对于大型模型则可能会导致噪音过多
verbose_progress = False
```