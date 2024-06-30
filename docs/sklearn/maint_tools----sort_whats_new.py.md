# `D:\src\scipysrc\scikit-learn\maint_tools\sort_whats_new.py`

```
#!/usr/bin/env python
# 定义了一个脚本，用于对给定的新功能条目进行排序，并按模块分组显示。
# 从标准输入读取新功能条目。

import re  # 导入正则表达式模块
import sys  # 导入系统相关的模块
from collections import defaultdict  # 导入默认字典模块

LABEL_ORDER = ["MajorFeature", "Feature", "Efficiency", "Enhancement", "Fix", "API"]

# 定义排序函数，根据条目的类型来确定排序顺序
def entry_sort_key(s):
    if s.startswith("- |"):
        return LABEL_ORDER.index(s.split("|")[1])
    else:
        return -1

# 从标准输入中获取条目文本，并丢弃标题和其他非条目行
text = "".join(l for l in sys.stdin if l.startswith("- ") or l.startswith(" "))

bucketed = defaultdict(list)  # 创建一个默认字典，用于按类别存储条目

# 使用正则表达式将文本分割成条目，并根据模块名称进行分类存储
for entry in re.split("\n(?=- )", text.strip()):
    modules = re.findall(
        r":(?:func|meth|mod|class):" r"`(?:[^<`]*<|~)?(?:sklearn.)?([a-z]\w+)", entry
    )
    modules = set(modules)
    if len(modules) > 1:
        key = "Multiple modules"
    elif modules:
        key = ":mod:`sklearn.%s`" % next(iter(modules))
    else:
        key = "Miscellaneous"
    bucketed[key].append(entry.strip() + "\n")

everything = []  # 创建一个空列表，用于存储排序后的结果

# 对存储的条目按类别排序并格式化输出
for key, bucket in sorted(bucketed.items()):
    everything.append(key + "\n" + "." * len(key))
    bucket.sort(key=entry_sort_key)  # 对每个类别的条目进行排序
    everything.extend(bucket)

# 打印最终的排序结果，类别之间使用双空行分隔
print("\n\n".join(everything))
```