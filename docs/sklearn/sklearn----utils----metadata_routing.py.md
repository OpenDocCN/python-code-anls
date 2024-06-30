# `D:\src\scipysrc\scikit-learn\sklearn\utils\metadata_routing.py`

```
"""Utilities to route metadata within scikit-learn estimators."""

# This module is not a separate sub-folder since that would result in a circular
# import issue.
#
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 从 _metadata_requests 模块导入以下标识符，使用 noqa 来忽略 PEP 8 的导入检查
from ._metadata_requests import WARN, UNUSED, UNCHANGED  # noqa
# 从 _metadata_requests 模块导入 get_routing_for_object 函数，使用 noqa 来忽略 PEP 8 的导入检查
from ._metadata_requests import get_routing_for_object  # noqa
# 从 _metadata_requests 模块导入 MetadataRouter 类，使用 noqa 来忽略 PEP 8 的导入检查
from ._metadata_requests import MetadataRouter  # noqa
# 从 _metadata_requests 模块导入 MetadataRequest 类，使用 noqa 来忽略 PEP 8 的导入检查
from ._metadata_requests import MetadataRequest  # noqa
# 从 _metadata_requests 模块导入 MethodMapping 类，使用 noqa 来忽略 PEP 8 的导入检查
from ._metadata_requests import MethodMapping  # noqa
# 从 _metadata_requests 模块导入 process_routing 函数，使用 noqa 来忽略 PEP 8 的导入检查
from ._metadata_requests import process_routing  # noqa
# 从 _metadata_requests 模块导入 _MetadataRequester 类，使用 noqa 来忽略 PEP 8 的导入检查
from ._metadata_requests import _MetadataRequester  # noqa
# 从 _metadata_requests 模块导入 _routing_enabled 函数，使用 noqa 来忽略 PEP 8 的导入检查
from ._metadata_requests import _routing_enabled  # noqa
# 从 _metadata_requests 模块导入 _raise_for_params 函数，使用 noqa 来忽略 PEP 8 的导入检查
from ._metadata_requests import _raise_for_params  # noqa
# 从 _metadata_requests 模块导入 _RoutingNotSupportedMixin 类，使用 noqa 来忽略 PEP 8 的导入检查
from ._metadata_requests import _RoutingNotSupportedMixin  # noqa
# 从 _metadata_requests 模块导入 _raise_for_unsupported_routing 函数，使用 noqa 来忽略 PEP 8 的导入检查
from ._metadata_requests import _raise_for_unsupported_routing  # noqa
```