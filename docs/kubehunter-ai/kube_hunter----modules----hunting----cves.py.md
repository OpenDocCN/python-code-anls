# `kubehunter\kube_hunter\modules\hunting\cves.py`

```
# 导入 logging 模块，用于记录日志
import logging
# 导入 packaging 模块中的 version 类，用于处理版本信息
from packaging import version
# 导入 kube_hunter.conf 模块中的 config 对象
from kube_hunter.conf import config
# 导入 kube_hunter.core.events 模块中的 handler 函数
from kube_hunter.core.events import handler
# 导入 kube_hunter.core.events.types 模块中的 Vulnerability、Event、K8sVersionDisclosure 类
from kube_hunter.core.events.types import Vulnerability, Event, K8sVersionDisclosure
# 导入 kube_hunter.core.types 模块中的 Hunter、KubernetesCluster、RemoteCodeExec、PrivilegeEscalation、DenialOfService、KubectlClient 类
from kube_hunter.core.types import (
    Hunter,
    KubernetesCluster,
    RemoteCodeExec,
    PrivilegeEscalation,
    DenialOfService,
    KubectlClient,
)
# 导入 kube_hunter.modules.discovery.kubectl 模块中的 KubectlClientEvent 类
from kube_hunter.modules.discovery.kubectl import KubectlClientEvent
# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 定义 ServerApiVersionEndPointAccessPE 类，继承自 Vulnerability 和 Event 类
class ServerApiVersionEndPointAccessPE(Vulnerability, Event):
# 定义一个名为Node的类，该类对CVE-2018-1002105存在漏洞
class Node:
    # 初始化方法，接受evidence作为参数
    def __init__(self, evidence):
        # 调用Vulnerability类的初始化方法，传入KubernetesCluster、name、category和vid作为参数
        Vulnerability.__init__(
            self,
            KubernetesCluster,
            name="Critical Privilege Escalation CVE",
            category=PrivilegeEscalation,
            vid="KHV022",
        )
        # 将evidence赋值给self.evidence
        self.evidence = evidence


# 定义一个名为ServerApiVersionEndPointAccessDos的类，该类对CVE-2019-1002100存在漏洞
class ServerApiVersionEndPointAccessDos(Vulnerability, Event):
    # 初始化方法，接受evidence作为参数
    def __init__(self, evidence):
        # 调用Vulnerability类的初始化方法，传入self、name作为参数
        Vulnerability.__init__(
            self,
# 定义一个名为KubernetesCluster的类，表示Kubernetes集群
# 定义一个名为Denial of Service to Kubernetes API Server的漏洞，属于DenialOfService类别，编号为KHV023
# 将证据赋值给self.evidence
class PingFloodHttp2Implementation(Vulnerability, Event):
    """Node not patched for CVE-2019-9512. an attacker could cause a
    Denial of Service by sending specially crafted HTTP requests."""

    # 初始化函数，接受evidence作为参数
    def __init__(self, evidence):
        # 调用Vulnerability类的初始化函数，设置漏洞类型为KubernetesCluster，名称为Possible Ping Flood Attack，类别为DenialOfService，编号为KHV024
        Vulnerability.__init__(
            self, KubernetesCluster, name="Possible Ping Flood Attack", category=DenialOfService, vid="KHV024",
        )
        # 将证据赋值给self.evidence
        self.evidence = evidence


# 定义一个名为ResetFloodHttp2Implementation的类，表示重置FloodHttp2Implementation
class ResetFloodHttp2Implementation(Vulnerability, Event):
# 定义一个名为 Node 的类，该类表示一个漏洞，该漏洞未修补 CVE-2019-9514。攻击者可以通过发送特制的 HTTP 请求导致拒绝服务。
class Node(Vulnerability):
    def __init__(self, evidence):
        # 调用父类的初始化方法，传入 KubernetesCluster 类型的实例，漏洞名称为 "Possible Reset Flood Attack"，类别为 DenialOfService，漏洞编号为 "KHV025"
        Vulnerability.__init__(
            self, KubernetesCluster, name="Possible Reset Flood Attack", category=DenialOfService, vid="KHV025",
        )
        # 设置漏洞的证据
        self.evidence = evidence


# 定义一个名为 ServerApiClusterScopedResourcesAccess 的类，该类表示一个漏洞，该漏洞未修补 CVE-2019-11247。API 服务器允许通过错误的范围访问自定义资源。
class ServerApiClusterScopedResourcesAccess(Vulnerability, Event):
    def __init__(self, evidence):
        # 调用父类的初始化方法，传入 KubernetesCluster 类型的实例，漏洞名称为 "Arbitrary Access To Cluster Scoped Resources"，类别为 PrivilegeEscalation
        Vulnerability.__init__(
            self,
            KubernetesCluster,
            name="Arbitrary Access To Cluster Scoped Resources",
            category=PrivilegeEscalation,
        )
# 定义一个名为IncompleteFixToKubectlCpVulnerability的类，继承自Vulnerability和Event类
class IncompleteFixToKubectlCpVulnerability(Vulnerability, Event):
    # 类的说明文档，描述kubectl客户端存在CVE-2019-11246漏洞，攻击者可能在客户端机器上执行任意代码
    """The kubectl client is vulnerable to CVE-2019-11246,
    an attacker could potentially execute arbitrary code on the client's machine"""

    # 类的初始化方法，接受binary_version参数
    def __init__(self, binary_version):
        # 调用Vulnerability类的初始化方法，传入KubectlClient、"Kubectl Vulnerable To CVE-2019-11246"等参数
        Vulnerability.__init__(
            self, KubectlClient, "Kubectl Vulnerable To CVE-2019-11246", category=RemoteCodeExec, vid="KHV027",
        )
        # 设置binary_version属性
        self.binary_version = binary_version
        # 设置evidence属性，描述kubectl版本信息
        self.evidence = "kubectl version: {}".format(self.binary_version)


# 定义一个名为KubectlCpVulnerability的类，继承自Vulnerability和Event类
class KubectlCpVulnerability(Vulnerability, Event):
    # 类的说明文档，描述kubectl客户端存在CVE-2019-1002101漏洞，攻击者可能在客户端机器上执行任意代码
    """The kubectl client is vulnerable to CVE-2019-1002101,
    an attacker could potentially execute arbitrary code on the client's machine"""
# 初始化函数，接受一个二进制版本参数
def __init__(self, binary_version):
    # 调用父类的初始化函数，传入 KubectlClient、漏洞描述、漏洞类别和漏洞 ID
    Vulnerability.__init__(
        self, KubectlClient, "Kubectl Vulnerable To CVE-2019-1002101", category=RemoteCodeExec, vid="KHV028",
    )
    # 设置对象的二进制版本属性
    self.binary_version = binary_version
    # 设置对象的证据属性，记录 kubectl 版本信息
    self.evidence = "kubectl version: {}".format(self.binary_version)

# CVE 工具类
class CveUtils:
    # 静态方法，获取基础版本号
    @staticmethod
    def get_base_release(full_ver):
        # 如果是 LegacyVersion 类型的版本号，手动转换为基础版本号
        if type(full_ver) == version.LegacyVersion:
            return version.parse(".".join(full_ver._version.split(".")[:2]))
        # 否则，将完整版本号转换为基础版本号
        return version.parse(".".join(map(str, full_ver._version.release[:2])))

    # 静态方法，将版本号转换为 LegacyVersion 类型
    @staticmethod
    def to_legacy(full_ver):
        # 将版本号转换为 LegacyVersion 类型
    # 将版本号转换为遗留版本对象
    @staticmethod
    def to_legacy_version(v):
        # 如果输入的版本号不是遗留版本对象，则将其转换为字符串形式
        if type(v) != version.LegacyVersion:
            return ".".join(map(str, v._version.release))
        # 如果输入的版本号已经是遗留版本对象，则直接返回其版本号
        return v._version

    # 比较两个版本号，处理版本号转换为遗留版本对象的差异
    @staticmethod
    def version_compare(v1, v2):
        """Function compares two versions, handling differences with conversion to LegacyVersion"""
        # 获取原始版本号，去除开头的 'v' 字符（如果存在的话），以便安全比较两个版本号
        v1_raw = CveUtils.to_legacy_version(v1).strip("v")
        v2_raw = CveUtils.to_legacy_version(v2).strip("v")
        # 创建新的遗留版本对象
        new_v1 = version.LegacyVersion(v1_raw)
        new_v2 = version.LegacyVersion(v2_raw)

        # 调用基本比较函数，比较两个版本号
        return CveUtils.basic_compare(new_v1, new_v2)
    # 静态方法，用于比较两个版本号的大小，返回值为-1、0、1
    @staticmethod
    def basic_compare(v1, v2):
        return (v1 > v2) - (v1 < v2)

    # 静态方法，用于判断版本号是否为下游版本
    @staticmethod
    def is_downstream_version(version):
        return any(c in version for c in "+-~")

    # 静态方法，用于判断给定版本是否存在漏洞
    @staticmethod
    def is_vulnerable(fix_versions, check_version, ignore_downstream=False):
        """Function determines if a version is vulnerable,
        by comparing to given fix versions by base release"""
        # 如果忽略下游版本并且检查版本是下游版本，则返回False
        if ignore_downstream and CveUtils.is_downstream_version(check_version):
            return False

        vulnerable = False
        # 解析检查版本号
        check_v = version.parse(check_version)
        # 获取基本发布版本的检查版本号
        base_check_v = CveUtils.get_base_release(check_v)

        # 默认使用经典比较，除非检查版本是传统版本。
# 设置版本比较函数为 CveUtils.basic_compare
version_compare_func = CveUtils.basic_compare
# 如果 check_v 的类型为 LegacyVersion，则将版本比较函数设置为 CveUtils.version_compare
if type(check_v) == version.LegacyVersion:
    version_compare_func = CveUtils.version_compare

# 如果 check_version 不在 fix_versions 中
if check_version not in fix_versions:
    # 遍历 fix_versions 中的版本
    for fix_v in fix_versions:
        # 将 fix_v 解析为版本对象
        fix_v = version.parse(fix_v)
        # 获取 fix_v 的基础版本
        base_fix_v = CveUtils.get_base_release(fix_v)

        # 如果 check_version 和当前 fix_v 具有相同的基础版本
        if base_check_v == base_fix_v:
            # 当 check_version 是 legacy 时，使用自定义比较函数处理版本之间的差异
            if version_compare_func(check_v, fix_v) == -1:
                # 如果 check_v 小于 fix_v 并且具有相同的基础版本，则标记为 vulnerable，并跳出循环
                vulnerable = True
                break

# 如果在 fix releases 中找不到修复版本，检查版本是否小于第一个修复版本
if not vulnerable and version_compare_func(check_v, version.parse(fix_versions[0])) == -1:
# 设置一个标志变量为True
vulnerable = True

# 返回标志变量的值
return vulnerable

# 订阅K8sVersionDisclosure事件的处理程序
@handler.subscribe_once(K8sVersionDisclosure)
class K8sClusterCveHunter(Hunter):
    """K8s CVE Hunter
    检查节点是否运行了特定重要CVE漏洞的Kubernetes版本
    """

    def __init__(self, event):
        self.event = event

    def execute(self):
        logger.debug(f"Checking known CVEs for k8s API version: {self.event.version}")
        # 定义CVE与Kubernetes版本的映射关系
        cve_mapping = {
            ServerApiVersionEndPointAccessPE: ["1.10.11", "1.11.5", "1.12.3"],
            ServerApiVersionEndPointAccessDos: ["1.11.8", "1.12.6", "1.13.4"],
# 定义一个字典，键为漏洞名称，值为修复版本号列表
cve_mapping = {
    ResetFloodHttp2Implementation: ["1.13.10", "1.14.6", "1.15.3"],
    PingFloodHttp2Implementation: ["1.13.10", "1.14.6", "1.15.3"],
    ServerApiClusterScopedResourcesAccess: ["1.13.9", "1.14.5", "1.15.2"],
}
# 遍历漏洞字典，检查当前版本是否存在漏洞，如果存在则发布事件
for vulnerability, fix_versions in cve_mapping.items():
    if CveUtils.is_vulnerable(fix_versions, self.event.version, not config.include_patched_versions):
        self.publish_event(vulnerability(self.event.version))
```

```
@handler.subscribe(KubectlClientEvent)
class KubectlCVEHunter(Hunter):
    """Kubectl CVE Hunter
    Checks if the kubectl client is vulnerable to specific important CVEs
    """

    def __init__(self, event):
        self.event = event

    def execute(self):
        # 定义一个漏洞与修复版本的映射字典
        cve_mapping = {
# 定义一个包含 Kubernetes kubectl 命令漏洞及其修复版本的字典
cve_mapping = {
    KubectlCpVulnerability: ["1.11.9", "1.12.7", "1.13.5", "1.14.0"],
    IncompleteFixToKubectlCpVulnerability: ["1.12.9", "1.13.6", "1.14.2"],
}
# 记录调试信息，检查 kubectl 版本是否存在已知的 CVE 漏洞
logger.debug(f"Checking known CVEs for kubectl version: {self.event.version}")
# 遍历漏洞字典，检查当前 kubectl 版本是否存在漏洞
for vulnerability, fix_versions in cve_mapping.items():
    # 如果当前 kubectl 版本存在漏洞，则根据配置发布相应的事件
    if CveUtils.is_vulnerable(fix_versions, self.event.version, not config.include_patched_versions):
        self.publish_event(vulnerability(binary_version=self.event.version))
```