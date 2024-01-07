# `.\kubehunter\kube_hunter\modules\report\plain.py`

```

# 导入未来版本的 print 函数
from __future__ import print_function

# 导入 prettytable 库中的 ALL 和 PrettyTable
from prettytable import ALL, PrettyTable

# 导入自定义模块中的 BaseReporter 类
from kube_hunter.modules.report.base import BaseReporter

# 导入自定义模块中的 collector 模块中的各种子模块
from kube_hunter.modules.report.collector import (
    services,
    vulnerabilities,
    hunters,
    services_lock,
    vulnerabilities_lock,
)

# 设置预览证据的长度
EVIDENCE_PREVIEW = 40

# 设置表格的最大宽度
MAX_TABLE_WIDTH = 20

# 设置链接地址
KB_LINK = "https://github.com/aquasecurity/kube-hunter/tree/master/docs/_kb"

```