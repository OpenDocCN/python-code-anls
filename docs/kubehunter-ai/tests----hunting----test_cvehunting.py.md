# `kubehunter\tests\hunting\test_cvehunting.py`

```
# 导入时间模块
import time

# 从 kube_hunter.core.events 模块中导入 handler
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 K8sVersionDisclosure 类
from kube_hunter.core.events.types import K8sVersionDisclosure
# 从 kube_hunter.modules.hunting.cves 模块中导入 K8sClusterCveHunter, ServerApiVersionEndPointAccessPE, ServerApiVersionEndPointAccessDos, CveUtils 类
from kube_hunter.modules.hunting.cves import (
    K8sClusterCveHunter,
    ServerApiVersionEndPointAccessPE,
    ServerApiVersionEndPointAccessDos,
    CveUtils,
)

# 初始化 CVE 计数器
cve_counter = 0

# 定义测试函数 test_K8sCveHunter
def test_K8sCveHunter():
    global cve_counter
    # 由于猎人自行注销，我们手动删除此选项，以便进行测试
    K8sClusterCveHunter.__new__ = lambda self, cls: object.__new__(self)

    # 创建 K8sVersionDisclosure 对象
    e = K8sVersionDisclosure(version="1.10.1", from_endpoint="/version")
    # 创建 K8sClusterCveHunter 对象
    h = K8sClusterCveHunter(e)
    # 执行猎人的执行方法
    h.execute()

    # 等待一段时间
    time.sleep(0.01)
    # 断言 CVE 计数器的值为 2
    assert cve_counter == 2
    # 重置 CVE 计数器
    cve_counter = 0

    # 测试已修补的版本
    e = K8sVersionDisclosure(version="v1.13.6-gke.13", from_endpoint="/version")
    h = K8sClusterCveHunter(e)
    h.execute()

    time.sleep(0.01)
    # 断言 CVE 计数器的值为 0
    assert cve_counter == 0
    # 重置 CVE 计数器
    cve_counter = 0

# 订阅 ServerApiVersionEndPointAccessPE 事件的处理器
@handler.subscribe(ServerApiVersionEndPointAccessPE)
class test_CVE_2018_1002105(object):
    # 初始化函数
    def __init__(self, event):
        global cve_counter
        # CVE 计数器加一
        cve_counter += 1

# 订阅 ServerApiVersionEndPointAccessDos 事件的处理器
@handler.subscribe(ServerApiVersionEndPointAccessDos)
class test_CVE_2019_1002100:
    # 初始化函数
    def __init__(self, event):
        global cve_counter
        # CVE 计数器加一
        cve_counter += 1

# 定义 TestCveUtils 类
class TestCveUtils:
    # 定义测试函数，用于测试是否为下游版本
    def test_is_downstream(self):
        # 定义测试用例
        test_cases = (
            ("1", False),  # 版本号为"1"，预期结果为False
            ("1.2", False),  # 版本号为"1.2"，预期结果为False
            ("1.2-3", True),  # 版本号为"1.2-3"，预期结果为True
            ("1.2-r3", True),  # 版本号为"1.2-r3"，预期结果为True
            ("1.2+3", True),  # 版本号为"1.2+3"，预期结果为True
            ("1.2~3", True),  # 版本号为"1.2~3"，预期结果为True
            ("1.2+a3f5cb2", True),  # 版本号为"1.2+a3f5cb2"，预期结果为True
            ("1.2-9287543", True),  # 版本号为"1.2-9287543"，预期结果为True
            ("v1", False),  # 版本号为"v1"，预期结果为False
            ("v1.2", False),  # 版本号为"v1.2"，预期结果为False
            ("v1.2-3", True),  # 版本号为"v1.2-3"，预期结果为True
            ("v1.2-r3", True),  # 版本号为"v1.2-r3"，预期结果为True
            ("v1.2+3", True),  # 版本号为"v1.2+3"，预期结果为True
            ("v1.2~3", True),  # 版本号为"v1.2~3"，预期结果为True
            ("v1.2+a3f5cb2", True),  # 版本号为"v1.2+a3f5cb2"，预期结果为True
            ("v1.2-9287543", True),  # 版本号为"v1.2-9287543"，预期结果为True
            ("v1.13.9-gke.3", True),  # 版本号为"v1.13.9-gke.3"，预期结果为True
        )

        # 遍历测试用例
        for version, expected in test_cases:
            # 调用函数获取实际结果
            actual = CveUtils.is_downstream_version(version)
            # 断言实际结果与预期结果相等
            assert actual == expected

    # 定义测试函数，用于测试是否忽略下游版本
    def test_ignore_downstream(self):
        # 定义测试用例
        test_cases = (
            ("v2.2-abcd", ["v1.1", "v2.3"], False),  # 版本号为"v2.2-abcd"，修复版本为["v1.1", "v2.3"]，预期结果为False
            ("v2.2-abcd", ["v1.1", "v2.2"], False),  # 版本号为"v2.2-abcd"，修复版本为["v1.1", "v2.2"]，预期结果为False
            ("v1.13.9-gke.3", ["v1.14.8"], False),  # 版本号为"v1.13.9-gke.3"，修复版本为["v1.14.8"]，预期结果为False
        )

        # 遍历测试用例
        for check_version, fix_versions, expected in test_cases:
            # 调用函数获取实际结果
            actual = CveUtils.is_vulnerable(fix_versions, check_version, True)
            # 断言实际结果与预期结果相等
            assert actual == expected
```