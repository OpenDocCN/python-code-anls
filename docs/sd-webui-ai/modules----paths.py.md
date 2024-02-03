# `stable-diffusion-webui\modules\paths.py`

```py
# 导入必要的模块
import os
import sys
from modules.paths_internal import models_path, script_path, data_path, extensions_dir, extensions_builtin_dir, cwd  # noqa: F401

# 导入安全模块
import modules.safe  # noqa: F401

# 创建虚拟模块，用于替代 SDXL 想要导入但实际上我们不需要的模块
def mute_sdxl_imports():
    class Dummy:
        pass

    # 创建一个空的 LPIPS 模块
    module = Dummy()
    module.LPIPS = None
    sys.modules['taming.modules.losses.lpips'] = module

    # 创建一个空的 StableDataModuleFromConfig 模块
    module = Dummy()
    module.StableDataModuleFromConfig = None
    sys.modules['sgm.data'] = module

# 将脚本路径添加到系统路径中
sys.path.insert(0, script_path)

# 在以下位置搜索稳定扩散的目录
sd_path = None
possible_sd_paths = [os.path.join(script_path, 'repositories/stable-diffusion-stability-ai'), '.', os.path.dirname(script_path)]
for possible_sd_path in possible_sd_paths:
    # 检查是否存在稳定扩散的关键文件
    if os.path.exists(os.path.join(possible_sd_path, 'ldm/models/diffusion/ddpm.py')):
        sd_path = os.path.abspath(possible_sd_path)
        break

# 确保找到稳定扩散的路径
assert sd_path is not None, f"Couldn't find Stable Diffusion in any of: {possible_sd_paths}"

# 屏蔽 SDXL 的导入
mute_sdxl_imports()

# 定义路径目录列表
path_dirs = [
    (sd_path, 'ldm', 'Stable Diffusion', []),
    (os.path.join(sd_path, '../generative-models'), 'sgm', 'Stable Diffusion XL', ["sgm"]),
    (os.path.join(sd_path, '../CodeFormer'), 'inference_codeformer.py', 'CodeFormer', []),
    (os.path.join(sd_path, '../BLIP'), 'models/blip.py', 'BLIP', []),
    (os.path.join(sd_path, '../k-diffusion'), 'k_diffusion/sampling.py', 'k_diffusion', ["atstart"]),
]

# 初始化路径字典
paths = {}

# 遍历路径目录列表
for d, must_exist, what, options in path_dirs:
    # 获取必须存在的路径
    must_exist_path = os.path.abspath(os.path.join(script_path, d, must_exist))
    # 如果路径不存在，则打印警告信息
    if not os.path.exists(must_exist_path):
        print(f"Warning: {what} not found at path {must_exist_path}", file=sys.stderr)
    # 如果条件不成立，获取绝对路径
    else:
        d = os.path.abspath(d)
        # 如果选项中包含"atstart"，将路径插入到 sys.path 的开头
        if "atstart" in options:
            sys.path.insert(0, d)
        # 如果选项中包含"sgm"，处理 Stable Diffusion XL 仓库的特殊情况
        elif "sgm" in options:
            # 在 sys.path 的开头插入路径
            sys.path.insert(0, d)
            # 导入 sgm 模块，但不使用 F401 警告
            import sgm  # noqa: F401
            # 移除刚才插入的路径，避免影响其他扩展的脚本目录
            sys.path.pop(0)
        # 如果选项中不包含"atstart"或"sgm"，将路径追加到 sys.path
        else:
            sys.path.append(d)
        # 将路径添加到 paths 字典中
        paths[what] = d
```