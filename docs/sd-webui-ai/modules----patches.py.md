# `stable-diffusion-webui\modules\patches.py`

```py
# 导入 defaultdict 模块
from collections import defaultdict

# 替换模块或类中的函数，并将原始函数存储在此模块中，可以通过 original(key, obj, field) 检索
# 如果函数已被此调用者（key）替换，则会引发异常 - 在此之前使用 undo()
def patch(key, obj, field, replacement):
    # 生成用于标识替换的键
    patch_key = (obj, field)
    # 如果 patch_key 在 originals[key] 中，则引发异常
    if patch_key in originals[key]:
        raise RuntimeError(f"patch for {field} is already applied")

    # 获取对象中的原始函数
    original_func = getattr(obj, field)
    # 将原始函数存储在 originals[key] 中
    originals[key][patch_key] = original_func

    # 替换对象中的函数为新函数
    setattr(obj, field, replacement)

    # 返回原始函数
    return original_func

# 撤销 patch() 函数替换的函数
# 如果函数未被替换，则引发异常
def undo(key, obj, field):
    # 生成用于标识替换的键
    patch_key = (obj, field)

    # 如果 patch_key 不在 originals[key] 中，则引发异常
    if patch_key not in originals[key]:
        raise RuntimeError(f"there is no patch for {field} to undo")

    # 弹出原始函数并将其设置回对象中
    original_func = originals[key].pop(patch_key)
    setattr(obj, field, original_func)

    # 总是返回 None
    return None

# 返回由 patch() 函数创建的 patch 的原始函数
def original(key, obj, field):
    # 生成用于标识替换的键
    patch_key = (obj, field)

    # 获取原始函数，如果不存在则返回 None
    return originals[key].get(patch_key, None)

# 创建一个 defaultdict 用于存储原始函数
originals = defaultdict(dict)
```