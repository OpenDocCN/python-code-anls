# `.\pytorch\torch\_inductor\codegen\aoti_hipify_utils.py`

```py
# mypy: allow-untyped-defs
# 引入正则表达式模块
import re

# 引入 PyTorch 模块
import torch

# 从 torch.utils.hipify.hipify_python 中导入需要的常量
from torch.utils.hipify.hipify_python import PYTORCH_MAP, PYTORCH_TRIE

# 不建议直接对代码生成应用 hipify_torch，因为这样会容易受到以下情况的影响：
#   "...
#    from ..codecache import CudaKernelParamCache
#   ..."
# 在这些情况下，不需要对代码生成/codecache中的原始类/文件名进行 hipify_torch 处理

# 定义一个函数，用于可能进行 HIP 转换的代码包装
def maybe_hipify_code_wrapper(source_codes: str, force_hipify: bool = False) -> str:
    # 如果当前 PyTorch 版本不支持 HIP 或者没有强制进行 HIP 转换，则直接返回原始代码
    if torch.version.hip is None and not force_hipify:
        return source_codes

    # 定义一个函数，用于替换匹配到的字符串
    def c2_repl(m):
        return PYTORCH_MAP[m.group(0)]

    # 在 hipify_torch 中，我们需要重新定义 RE_PYTORCH_PREPROCESSOR，
    # 因为它会对模式应用正向后查找 (?<=\W)，以避免在代码生成中匹配行首关键字。
    # 但是在代码生成中，这种情况可能发生，会导致模式不匹配。

    # 需要注意的是，前瞻查找 (?=\W) 仍然需要保留以保持 HIP 转换的一部分，
    # 例如，我们需要跳过在 "getStreamFromExternalMasqueradingAsCUDA" 中的 "getStreamFromExternal" 的替换。
    RE_PYTORCH_PREPROCESSOR = re.compile(rf"({PYTORCH_TRIE.export_to_regex()})(?=\W)")

    # 对源代码应用 RE_PYTORCH_PREPROCESSOR 中定义的替换函数 c2_repl
    source_codes = RE_PYTORCH_PREPROCESSOR.sub(c2_repl, source_codes)
    
    # 返回经过可能的 HIP 转换处理后的源代码
    return source_codes
```