# `ZeroNet\plugins\disabled-StemPort\__init__.py`

```
# 尝试导入 stem.control 模块，如果成功则将 stem_found 设置为 True
try:
    from stem.control import Controller
    stem_found = True
# 如果导入失败，则打印错误信息并将 stem_found 设置为 False
except Exception as err:
    print(("STEM NOT FOUND! %s" % err))
    stem_found = False

# 如果 stem_found 为 True，则打印提示信息
if stem_found:
    print("Starting Stem plugin...")
    # 导入 StemPortPlugin 模块
    from . import StemPortPlugin
```