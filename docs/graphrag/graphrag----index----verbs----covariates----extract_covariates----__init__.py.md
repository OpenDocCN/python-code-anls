# `.\graphrag\graphrag\index\verbs\covariates\extract_covariates\__init__.py`

```py
# 版权声明，标明此部分代码版权归 2024 年的 Microsoft Corporation 所有，并遵循 MIT 许可协议
# 此模块用于索引引擎的文本提取索赔包的根目录

# 导入从 extract_covariates 模块中引入的 ExtractClaimsStrategyType 类和 extract_covariates 函数
from .extract_covariates import ExtractClaimsStrategyType, extract_covariates

# __all__ 列表，用于声明模块中公开的接口，即在 from package import * 时会被导入的对象
__all__ = ["ExtractClaimsStrategyType", "extract_covariates"]
```