# `kubehunter\kube_hunter\modules\hunting\cves.py`

```
# 导入 logging 模块
import logging
# 从 packaging 模块中导入 version 类
from packaging import version
# 从 kube_hunter.conf 模块中导入 config 变量
from kube_hunter.conf import config
# 从 kube_hunter.core.events 模块中导入 handler 函数
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Vulnerability, Event, K8sVersionDisclosure 类
from kube_hunter.core.events.types import Vulnerability, Event, K8sVersionDisclosure
# 从 kube_hunter.core.types 模块中导入 Hunter, KubernetesCluster, RemoteCodeExec, PrivilegeEscalation, DenialOfService, KubectlClient 类
from kube_hunter.core.types import (
    Hunter,
    KubernetesCluster,
    RemoteCodeExec,
    PrivilegeEscalation,
    DenialOfService,
    KubectlClient,
)
# 从 kube_hunter.modules.discovery.kubectl 模块中导入 KubectlClientEvent 类
from kube_hunter.modules.discovery.kubectl import KubectlClientEvent
# 获取 logger 对象
logger = logging.getLogger(__name__)

# 定义 ServerApiVersionEndPointAccessPE 类，继承自 Vulnerability, Event 类
class ServerApiVersionEndPointAccessPE(Vulnerability, Event):
    """Node is vulnerable to critical CVE-2018-1002105"""

    # 初始化方法
    def __init__(self, evidence):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self,
            KubernetesCluster,
            name="Critical Privilege Escalation CVE",
            category=PrivilegeEscalation,
            vid="KHV022",
        )
        # 设置 evidence 属性
        self.evidence = evidence

# 定义 ServerApiVersionEndPointAccessDos 类，继承自 Vulnerability, Event 类
class ServerApiVersionEndPointAccessDos(Vulnerability, Event):
    """Node not patched for CVE-2019-1002100. Depending on your RBAC settings,
     a crafted json-patch could cause a Denial of Service."""

    # 初始化方法
    def __init__(self, evidence):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self,
            KubernetesCluster,
            name="Denial of Service to Kubernetes API Server",
            category=DenialOfService,
            vid="KHV023",
        )
        # 设置 evidence 属性
        self.evidence = evidence

# 定义 PingFloodHttp2Implementation 类，继承自 Vulnerability, Event 类
class PingFloodHttp2Implementation(Vulnerability, Event):
    """Node not patched for CVE-2019-9512. an attacker could cause a
    Denial of Service by sending specially crafted HTTP requests."""

    # 初始化方法
    def __init__(self, evidence):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self, KubernetesCluster, name="Possible Ping Flood Attack", category=DenialOfService, vid="KHV024",
        )
        # 设置 evidence 属性
        self.evidence = evidence

# 定义 ResetFloodHttp2Implementation 类，继承自 Vulnerability, Event 类
class ResetFloodHttp2Implementation(Vulnerability, Event):
    """Node not patched for CVE-2019-9514. an attacker could cause a
    Denial of Service by sending specially crafted HTTP requests."""
    # 定义初始化方法，接受一个参数 evidence
    def __init__(self, evidence):
        # 调用父类Vulnerability的初始化方法，传入参数 KubernetesCluster, name="Possible Reset Flood Attack", category=DenialOfService, vid="KHV025"
        Vulnerability.__init__(
            self, KubernetesCluster, name="Possible Reset Flood Attack", category=DenialOfService, vid="KHV025",
        )
        # 设置实例属性 evidence 为传入的参数
        self.evidence = evidence
class ServerApiClusterScopedResourcesAccess(Vulnerability, Event):
    """Api Server not patched for CVE-2019-11247.
    API server allows access to custom resources via wrong scope"""

    def __init__(self, evidence):
        # 调用父类的初始化方法，传入漏洞相关信息
        Vulnerability.__init__(
            self,
            KubernetesCluster,
            name="Arbitrary Access To Cluster Scoped Resources",
            category=PrivilegeEscalation,
            vid="KHV026",
        )
        # 保存证据信息
        self.evidence = evidence


class IncompleteFixToKubectlCpVulnerability(Vulnerability, Event):
    """The kubectl client is vulnerable to CVE-2019-11246,
    an attacker could potentially execute arbitrary code on the client's machine"""

    def __init__(self, binary_version):
        # 调用父类的初始化方法，传入漏洞相关信息
        Vulnerability.__init__(
            self, KubectlClient, "Kubectl Vulnerable To CVE-2019-11246", category=RemoteCodeExec, vid="KHV027",
        )
        # 保存二进制版本信息
        self.binary_version = binary_version
        # 保存证据信息
        self.evidence = "kubectl version: {}".format(self.binary_version)


class KubectlCpVulnerability(Vulnerability, Event):
    """The kubectl client is vulnerable to CVE-2019-1002101,
    an attacker could potentially execute arbitrary code on the client's machine"""

    def __init__(self, binary_version):
        # 调用父类的初始化方法，传入漏洞相关信息
        Vulnerability.__init__(
            self, KubectlClient, "Kubectl Vulnerable To CVE-2019-1002101", category=RemoteCodeExec, vid="KHV028",
        )
        # 保存二进制版本信息
        self.binary_version = binary_version
        # 保存证据信息
        self.evidence = "kubectl version: {}".format(self.binary_version)


class CveUtils:
    @staticmethod
    def get_base_release(full_ver):
        # 如果是 LegacyVersion 类型，手动转换为基本版本
        if type(full_ver) == version.LegacyVersion:
            return version.parse(".".join(full_ver._version.split(".")[:2]))
        # 否则，将完整版本转换为基本版本
        return version.parse(".".join(map(str, full_ver._version.release[:2])))

    @staticmethod
    # 将完整版本号转换为 version.LegacyVersion 类型的版本号
    def to_legacy(full_ver):
        # 将版本号转换为 version.LegacyVersion 类型的对象
        return version.LegacyVersion(".".join(map(str, full_ver._version.release)))

    # 将版本号转换为原始版本号
    @staticmethod
    def to_raw_version(v):
        # 如果版本号不是 version.LegacyVersion 类型，则将其转换为字符串形式
        if type(v) != version.LegacyVersion:
            return ".".join(map(str, v._version.release)
        # 如果是 version.LegacyVersion 类型，则直接返回原始版本号
        return v._version

    # 比较两个版本号，处理差异并转换为 LegacyVersion 类型
    @staticmethod
    def version_compare(v1, v2):
        """Function compares two versions, handling differences with conversion to LegacyVersion"""
        # 获取原始版本号，去除开头的 'v' 字符（如果存在的话），以便安全比较两个版本号
        v1_raw = CveUtils.to_raw_version(v1).strip("v")
        v2_raw = CveUtils.to_raw_version(v2).strip("v")
        # 创建 LegacyVersion 类型的新版本号
        new_v1 = version.LegacyVersion(v1_raw)
        new_v2 = version.LegacyVersion(v2_raw)

        # 调用 basic_compare 方法比较两个版本号
        return CveUtils.basic_compare(new_v1, new_v2)

    # 基本的版本号比较方法
    @staticmethod
    def basic_compare(v1, v2):
        return (v1 > v2) - (v1 < v2)

    # 判断是否为下游版本号
    @staticmethod
    def is_downstream_version(version):
        # 判断版本号中是否包含 '+', '-', '~' 中的任意一个字符
        return any(c in version for c in "+-~")

    @staticmethod
    # 判断一个版本是否存在漏洞，通过与给定的修复版本进行比较
    def is_vulnerable(fix_versions, check_version, ignore_downstream=False):
        """Function determines if a version is vulnerable,
        by comparing to given fix versions by base release"""
        # 如果忽略下游版本并且检查版本是下游版本，则返回 False
        if ignore_downstream and CveUtils.is_downstream_version(check_version):
            return False

        vulnerable = False
        # 解析要检查的版本号
        check_v = version.parse(check_version)
        # 获取检查版本的基本发布版本
        base_check_v = CveUtils.get_base_release(check_v)

        # 默认使用经典比较，除非检查版本是传统版本。
        version_compare_func = CveUtils.basic_compare
        if type(check_v) == version.LegacyVersion:
            version_compare_func = CveUtils.version_compare

        # 如果检查版本不在修复版本列表中
        if check_version not in fix_versions:
            # 比较修复版本的基本发布版本
            for fix_v in fix_versions:
                fix_v = version.parse(fix_v)
                base_fix_v = CveUtils.get_base_release(fix_v)

                # 如果检查版本和当前修复版本具有相同的基本发布版本
                if base_check_v == base_fix_v:
                    # 当检查版本是传统版本时，我们使用自定义比较函数来处理版本之间的差异。
                    if version_compare_func(check_v, fix_v) == -1:
                        # 如果较小且具有相同的基本版本，则确定存在漏洞
                        vulnerable = True
                        break

        # 如果在修复版本中找不到修复，检查版本是否小于第一个修复版本
        if not vulnerable and version_compare_func(check_v, version.parse(fix_versions[0])) == -1:
            vulnerable = True

        return vulnerable
# 订阅 K8sVersionDisclosure 事件，并定义 K8sClusterCveHunter 类
@handler.subscribe_once(K8sVersionDisclosure)
class K8sClusterCveHunter(Hunter):
    """K8s CVE Hunter
    Checks if Node is running a Kubernetes version vulnerable to
    specific important CVEs
    """

    # 初始化方法，接收事件对象
    def __init__(self, event):
        self.event = event

    # 执行方法，检查已知 CVE 是否适用于 k8s API 版本
    def execute(self):
        # 记录调试信息，检查已知 CVE 是否适用于 k8s API 版本
        logger.debug(f"Checking known CVEs for k8s API version: {self.event.version}")
        # 定义 CVE 映射关系，包含漏洞和修复版本
        cve_mapping = {
            ServerApiVersionEndPointAccessPE: ["1.10.11", "1.11.5", "1.12.3"],
            ServerApiVersionEndPointAccessDos: ["1.11.8", "1.12.6", "1.13.4"],
            ResetFloodHttp2Implementation: ["1.13.10", "1.14.6", "1.15.3"],
            PingFloodHttp2Implementation: ["1.13.10", "1.14.6", "1.15.3"],
            ServerApiClusterScopedResourcesAccess: ["1.13.9", "1.14.5", "1.15.2"],
        }
        # 遍历 CVE 映射关系，检查是否存在漏洞适用于当前 k8s API 版本
        for vulnerability, fix_versions in cve_mapping.items():
            if CveUtils.is_vulnerable(fix_versions, self.event.version, not config.include_patched_versions):
                # 发布漏洞事件
                self.publish_event(vulnerability(self.event.version))


# 订阅 KubectlClientEvent 事件，并定义 KubectlCVEHunter 类
@handler.subscribe(KubectlClientEvent)
class KubectlCVEHunter(Hunter):
    """Kubectl CVE Hunter
    Checks if the kubectl client is vulnerable to specific important CVEs
    """

    # 初始化方法，接收事件对象
    def __init__(self, event):
        self.event = event

    # 执行方法，检查已知 CVE 是否适用于 kubectl 版本
    def execute(self):
        # 定义 CVE 映射关系，包含漏洞和修复版本
        cve_mapping = {
            KubectlCpVulnerability: ["1.11.9", "1.12.7", "1.13.5", "1.14.0"],
            IncompleteFixToKubectlCpVulnerability: ["1.12.9", "1.13.6", "1.14.2"],
        }
        # 记录调试信息，检查已知 CVE 是否适用于 kubectl 版本
        logger.debug(f"Checking known CVEs for kubectl version: {self.event.version}")
        # 遍历 CVE 映射关系，检查是否存在漏洞适用于当前 kubectl 版本
        for vulnerability, fix_versions in cve_mapping.items():
            if CveUtils.is_vulnerable(fix_versions, self.event.version, not config.include_patched_versions):
                # 发布漏洞事件
                self.publish_event(vulnerability(binary_version=self.event.version))
```