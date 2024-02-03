# `stable-diffusion-webui\modules\localization.py`

```py
# 导入 json 和 os 模块
import json
import os

# 从 modules 模块中导入 errors 和 scripts 模块
from modules import errors, scripts

# 创建空的本地化字典
localizations = {}

# 列出指定目录中的本地化文件
def list_localizations(dirname):
    # 清空本地化字典
    localizations.clear()

    # 遍历目录中的文件
    for file in os.listdir(dirname):
        # 获取文件名和扩展名
        fn, ext = os.path.splitext(file)
        # 如果扩展名不是 .json，则跳过
        if ext.lower() != ".json":
            continue

        # 将文件名和路径添加到本地化字典中
        localizations[fn] = [os.path.join(dirname, file)]

    # 遍历脚本中的本地化文件
    for file in scripts.list_scripts("localizations", ".json"):
        # 获取文件名和扩展名
        fn, ext = os.path.splitext(file.filename)
        # 如果文件名不在本地化字典中，则添加到字典中
        if fn not in localizations:
            localizations[fn] = []
        localizations[fn].append(file.path)

# 生成本地化的 JavaScript 代码
def localization_js(current_localization_name: str) -> str:
    # 获取当前本地化名称对应的文件列表
    fns = localizations.get(current_localization_name, None)
    # 创建空的数据字典
    data = {}
    # 如果文件列表不为空
    if fns is not None:
        # 遍历文件列表
        for fn in fns:
            try:
                # 尝试打开文件并加载数据到数据字典中
                with open(fn, "r", encoding="utf8") as file:
                    data.update(json.load(file))
            except Exception:
                # 报告加载本地化文件时的错误
                errors.report(f"Error loading localization from {fn}", exc_info=True)

    # 返回包含本地化数据的 JavaScript 代码
    return f"window.localization = {json.dumps(data)}"
```