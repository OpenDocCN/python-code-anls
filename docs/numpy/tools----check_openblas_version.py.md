# `.\numpy\tools\check_openblas_version.py`

```py
"""
usage: check_openblas_version.py <min_version>

Check the blas version is blas from scipy-openblas and is higher than
min_version
example: check_openblas_version.py 0.3.26
"""

# 导入必要的库
import numpy
import pprint
import sys

# 从命令行参数获取所需的最低版本号
version = sys.argv[1]

# 获取 numpy 的构建依赖配置信息
deps = numpy.show_config('dicts')['Build Dependencies']

# 确保依赖中包含名为 "blas" 的信息
assert "blas" in deps

# 输出提示信息
print("Build Dependencies: blas")

# 打印 blas 的详细信息
pprint.pprint(deps["blas"])

# 确保 blas 的版本号高于等于指定的最低版本号
assert deps["blas"]["version"].split(".") >= version.split(".")

# 确保 blas 的名称为 "scipy-openblas"
assert deps["blas"]["name"] == "scipy-openblas"
```