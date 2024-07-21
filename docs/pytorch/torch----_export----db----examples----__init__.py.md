# `.\pytorch\torch\_export\db\examples\__init__.py`

```py
# 添加类型检查允许未声明的函数
# 导入必要的库和模块
import dataclasses          # 导入数据类支持
import glob                 # 用于文件名模式匹配
import inspect              # 用于检查对象
from os.path import basename, dirname, isfile, join   # 导入路径相关的函数

import torch               # 导入PyTorch库
from torch._export.db.case import (  # 导入特定的导出案例相关模块
    _EXAMPLE_CASES,
    _EXAMPLE_CONFLICT_CASES,
    _EXAMPLE_REWRITE_CASES,
    SupportLevel,
    export_case,
    ExportCase,
)


def _collect_examples():
    # 获取当前目录下所有扩展名为.py的文件名列表
    case_names = glob.glob(join(dirname(__file__), "*.py"))
    # 过滤掉非文件或以__init__.py结尾的文件，获取纯文件名列表
    case_names = [
        basename(f)[:-3] for f in case_names if isfile(f) and not f.endswith("__init__.py")
    ]

    # 获取ExportCase数据类的字段名集合
    case_fields = {f.name for f in dataclasses.fields(ExportCase)}
    # 遍历每个文件名，动态导入模块
    for case_name in case_names:
        case = __import__(case_name, globals(), locals(), [], 1)
        # 获取模块中与ExportCase字段名匹配的变量名列表
        variables = [name for name in dir(case) if name in case_fields]
        # 调用export_case函数，传入模块变量名和对应的属性值
        export_case(**{v: getattr(case, v) for v in variables})(case.model)

# 收集所有案例示例
_collect_examples()

# 返回所有示例案例
def all_examples():
    return _EXAMPLE_CASES


# 如果存在冲突的导出案例，则引发运行时错误
if len(_EXAMPLE_CONFLICT_CASES) > 0:

    def get_name(case):
        # 获取案例关联的模型类名
        model = case.model
        if isinstance(model, torch.nn.Module):
            model = type(model)
        return model.__name__

    # 构建错误消息字符串
    msg = "Error on conflict export case name.\n"
    for case_name, cases in _EXAMPLE_CONFLICT_CASES.items():
        msg += f"Case name {case_name} is associated with multiple cases:\n  "
        msg += f"[{','.join(map(get_name, cases))}]\n"

    # 抛出运行时异常，包含错误消息
    raise RuntimeError(msg)


# 根据支持级别过滤示例案例
def filter_examples_by_support_level(support_level: SupportLevel):
    return {
        key: val
        for key, val in all_examples().items()
        if val.support_level == support_level
    }


# 获取重写案例
def get_rewrite_cases(case):
    return _EXAMPLE_REWRITE_CASES.get(case.name, [])
```