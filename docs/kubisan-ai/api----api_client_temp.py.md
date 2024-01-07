# `KubiScan\api\api_client_temp.py`

```
# 这是 API 客户端模块的临时部分，用于绕过 bug https://github.com/kubernetes-client/python/issues/577
# 当 bug 被解决后，可以删除这部分代码，并在 utils.py 中使用原始的 API 调用
# 它仅用于 list_cluster_role_binding()

# 编码声明，指定文件编码为 utf-8
# Kubernetes
# 未提供描述（由 Swagger Codegen https://github.com/swagger-api/swagger-codegen 生成）

# 导入必要的模块
from __future__ import absolute_import
import os
import re
import json
import mimetypes
import tempfile
from multiprocessing.pool import ThreadPool
from datetime import date, datetime

# 导入兼容 Python 2 和 Python 3 的库
from six import PY3, integer_types, iteritems, text_type
from six.moves.urllib.parse import quote

# 导入 Kubernetes 客户端相关模块
from kubernetes.client import models, V1ObjectMeta, V1RoleRef, V1Subject, V1ClusterRoleBinding, V1ClusterRole, V1ClusterRoleList, V1ClusterRoleBindingList, V1PolicyRule
from kubernetes.client.configuration import Configuration
from kubernetes.client.rest import ApiException, RESTClientObject
```