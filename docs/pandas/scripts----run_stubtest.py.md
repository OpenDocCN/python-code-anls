# `D:\src\scipysrc\pandas\scripts\run_stubtest.py`

```
# 导入标准库中的模块和函数
import os
from pathlib import Path
import sys
import tempfile
import warnings

# 导入第三方库中的模块
from mypy import stubtest

# 导入 pandas 库并获取其版本信息
import pandas as pd

# 获取 pandas 的版本信息，如果未安装 pandas，则 pd_version 为空字符串
pd_version = getattr(pd, "__version__", "")

# 如果未安装 pandas，则发出警告并终止程序运行
if not pd_version:
    warnings.warn("You need to install the development version of pandas")
    # 如果在持续集成环境中（CI），则以失败的状态退出
    if pd.compat.is_ci_environment():
        sys.exit(1)
    # 否则在本地开发环境中软性失败
    else:
        sys.exit(0)

# 如果 pandas 版本不包含 "dev"，则发出警告
if "dev" not in pd_version:
    warnings.warn(
        f"stubtest may fail as {pd_version} is not a dev version. "
        f"Please install a pandas dev version or see https://pandas.pydata.org/"
        f"pandas-docs/stable/development/contributing_codebase.html"
        f"#validating-type-hints on how to skip the stubtest"
    )

# _ALLOWLIST 列表，用于存放允许的方法和属性名称
_ALLOWLIST = [  # should be empty
    # TODO (child classes implement these methods)
    "pandas._libs.hashtable.HashTable.__contains__",
    "pandas._libs.hashtable.HashTable.__len__",
    "pandas._libs.hashtable.HashTable.factorize",
    "pandas._libs.hashtable.HashTable.get_item",
    "pandas._libs.hashtable.HashTable.get_labels",
    "pandas._libs.hashtable.HashTable.get_na",
    "pandas._libs.hashtable.HashTable.get_state",
    "pandas._libs.hashtable.HashTable.lookup",
    "pandas._libs.hashtable.HashTable.map_locations",
    "pandas._libs.hashtable.HashTable.set_item",
    "pandas._libs.hashtable.HashTable.set_na",
    "pandas._libs.hashtable.HashTable.sizeof",
    "pandas._libs.hashtable.HashTable.unique",
    "pandas._libs.hashtable.HashTable.hash_inner_join",
    # stubtest might be too sensitive
    "pandas._libs.lib.NoDefault",
    "pandas._libs.lib._NoDefault.no_default",
    # stubtest/Cython is not recognizing the default value for the dtype parameter
    "pandas._libs.lib.map_infer_mask",
    # internal type alias (should probably be private)
    "pandas._libs.lib.ndarray_obj_2d",
    # runtime argument "owner" has a default value but stub argument does not
    "pandas._libs.properties.AxisProperty.__get__",
    "pandas._libs.properties.cache_readonly.deleter",
    "pandas._libs.properties.cache_readonly.getter",
    "pandas._libs.properties.cache_readonly.setter",
    # TODO (child classes implement these methods)
    "pandas._libs.sparse.SparseIndex.__init__",
    "pandas._libs.sparse.SparseIndex.equals",
    "pandas._libs.sparse.SparseIndex.indices",
    "pandas._libs.sparse.SparseIndex.intersect",
    "pandas._libs.sparse.SparseIndex.lookup",
    "pandas._libs.sparse.SparseIndex.lookup_array",
    "pandas._libs.sparse.SparseIndex.make_union",
    "pandas._libs.sparse.SparseIndex.nbytes",
    "pandas._libs.sparse.SparseIndex.ngaps",
    "pandas._libs.sparse.SparseIndex.to_block_index",
    "pandas._libs.sparse.SparseIndex.to_int_index",
    # TODO (decorator changes argument names)
    "pandas._libs.tslibs.offsets.BusinessHour.rollback",
    "pandas._libs.tslibs.offsets.BusinessHour.rollforward ",
    # type alias
    # 字符串定义，表示要导入的模块或对象名称
    "pandas._libs.tslibs.timedeltas.UnitChoices",
# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 获取当前工作目录的路径对象
    root = Path.cwd()
    
    # 查找所有位于 pandas 子目录及其子目录下的 .pyi 文件
    pyi_modules = [
        # 生成每个 .pyi 文件相对于当前工作目录的路径，并去除文件扩展名，将路径分隔符替换为 "."
        str(pyi.relative_to(root).with_suffix("")).replace(os.sep, ".")
        for pyi in root.glob("pandas/**/*.pyi")
    ]

    # 创建临时文件作为 allowlist
    with tempfile.NamedTemporaryFile(mode="w+t") as allow:
        # 将 _ALLOWLIST 中的内容写入临时文件
        allow.write("\n".join(_ALLOWLIST))
        allow.flush()

        # 构造命令行参数列表
        args = pyi_modules + [
            "--ignore-missing-stub",  # 忽略缺失的类型提示
            "--concise",  # 使用简洁模式输出
            "--mypy-config-file",  # 指定 Mypy 配置文件为 pyproject.toml
            "--allowlist",  # 指定 allowlist 文件为之前创建的临时文件名
            allow.name,  # 传递临时文件的路径作为参数
        ]
        
        # 使用 stubtest 模块执行测试，并退出程序
        sys.exit(stubtest.test_stubs(stubtest.parse_options(args)))
```