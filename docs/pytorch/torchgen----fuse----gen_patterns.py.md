# `.\pytorch\torchgen\fuse\gen_patterns.py`

```py
#!/usr/bin/env python3
# 导入操作系统相关的模块
import os

# 导入模式匹配器和关联的图形处理模块
from torch._inductor import pattern_matcher
from torch._inductor.fx_passes import joint_graph

# 如果这个脚本是主程序
if __name__ == "__main__":
    # 开始删除所有现有的模式文件。
    # 遍历模式匹配器中的序列化模式路径下的所有条目。
    for path in pattern_matcher.SERIALIZED_PATTERN_PATH.iterdir():
        # 如果路径名是这些特定的值之一，则跳过。
        if path.name in {"__init__.py", "__pycache__"}:
            continue
        # 如果路径是文件，则删除它。
        if path.is_file():
            path.unlink()

    # 现在让联合图形加载所有已知模式，并告知模式匹配器在处理过程中序列化这些模式。
    # 设置环境变量以启用模式生成标志。
    os.environ["PYTORCH_GEN_PATTERNS"] = "1"
    # 执行联合图形的延迟初始化。
    joint_graph.lazy_init()
```