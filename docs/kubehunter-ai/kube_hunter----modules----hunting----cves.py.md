# `.\kubehunter\kube_hunter\modules\hunting\cves.py`

```

# 导入日志模块
import logging
# 导入版本模块
from packaging import version
# 导入配置模块
from kube_hunter.conf import config
# 导入事件处理模块
from kube_hunter.core.events import handler
# 导入事件类型模块
from kube_hunter.core.events.types import Vulnerability, Event, K8sVersionDisclosure
# 导入类型模块
from kube_hunter.core.types import (
    Hunter,
    KubernetesCluster,
    RemoteCodeExec,
    PrivilegeEscalation,
    DenialOfService,
    KubectlClient,
)
# 导入kubectl模块
from kube_hunter.modules.discovery.kubectl import KubectlClientEvent

# 获取logger对象
logger = logging.getLogger(__name__)

# 定义ServerApiVersionEndPointAccessPE类，继承Vulnerability和Event类
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
        self.evidence = evidence

# 定义ServerApiVersionEndPointAccessDos类，继承Vulnerability和Event类
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
        self.evidence = evidence

# 定义PingFloodHttp2Implementation类，继承Vulnerability和Event类
class PingFloodHttp2Implementation(Vulnerability, Event):
    """Node not patched for CVE-2019-9512. an attacker could cause a
    Denial of Service by sending specially crafted HTTP requests."""

    # 初始化方法
    def __init__(self, evidence):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self, KubernetesCluster, name="Possible Ping Flood Attack", category=DenialOfService, vid="KHV024",
        )
        self.evidence = evidence

# 定义ResetFloodHttp2Implementation类，继承Vulnerability和Event类
class ResetFloodHttp2Implementation(Vulnerability, Event):
    """Node not patched for CVE-2019-9514. an attacker could cause a
    Denial of Service by sending specially crafted HTTP requests."""

    # 初始化方法
    def __init__(self, evidence):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self, KubernetesCluster, name="Possible Reset Flood Attack", category=DenialOfService, vid="KHV025",
        )
        self.evidence = evidence

# 定义ServerApiClusterScopedResourcesAccess类，继承Vulnerability和Event类
class ServerApiClusterScopedResourcesAccess(Vulnerability, Event):
    """Api Server not patched for CVE-2019-11247.
    API server allows access to custom resources via wrong scope"""

    # 初始化方法
    def __init__(self, evidence):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self,
            KubernetesCluster,
            name="Arbitrary Access To Cluster Scoped Resources",
            category=PrivilegeEscalation,
            vid="KHV026",
        )
        self.evidence = evidence

# 定义IncompleteFixToKubectlCpVulnerability类，继承Vulnerability和Event类
class IncompleteFixToKubectlCpVulnerability(Vulnerability, Event):
    """The kubectl client is vulnerable to CVE-2019-11246,
    an attacker could potentially execute arbitrary code on the client's machine"""

    # 初始化方法
    def __init__(self, binary_version):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self, KubectlClient, "Kubectl Vulnerable To CVE-2019-11246", category=RemoteCodeExec, vid="KHV027",
        )
        self.binary_version = binary_version
        self.evidence = "kubectl version: {}".format(self.binary_version)

# 定义KubectlCpVulnerability类，继承Vulnerability和Event类
class KubectlCpVulnerability(Vulnerability, Event):
    """The kubectl client is vulnerable to CVE-2019-1002101,
    an attacker could potentially execute arbitrary code on the client's machine"""

    # 初始化方法
    def __init__(self, binary_version):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self, KubectlClient, "Kubectl Vulnerable To CVE-2019-1002101", category=RemoteCodeExec, vid="KHV028",
        )
        self.binary_version = binary_version
        self.evidence = "kubectl version: {}".format(self.binary_version)

# 定义CveUtils类
class CveUtils:
    # 获取基础版本
    @staticmethod
    def get_base_release(full_ver):
        # 如果是LegacyVersion，手动转换为基础版本
        if type(full_ver) == version.LegacyVersion:
            return version.parse(".".join(full_ver._version.split(".")[:2]))
        return version.parse(".".join(map(str, full_ver._version.release[:2])))

    # 转换为LegacyVersion
    @staticmethod
    def to_legacy(full_ver):
        return version.LegacyVersion(".".join(map(str, full_ver._version.release)))

    # 转换为原始版本
    @staticmethod
    def to_raw_version(v):
        if type(v) != version.LegacyVersion:
            return ".".join(map(str, v._version.release))
        return v._version

    # 版本比较
    @staticmethod
    def version_compare(v1, v2):
        """Function compares two versions, handling differences with conversion to LegacyVersion"""
        v1_raw = CveUtils.to_raw_version(v1).strip("v")
        v2_raw = CveUtils.to_raw_version(v2).strip("v")
        new_v1 = version.LegacyVersion(v1_raw)
        new_v2 = version.LegacyVersion(v2_raw)

        return CveUtils.basic_compare(new_v1, new_v2)

    # 基本比较
    @staticmethod
    def basic_compare(v1, v2):
        return (v1 > v2) - (v1 < v2)

    # 是否是下游版本
    @staticmethod
    def is_downstream_version(version):
        return any(c in version for c in "+-~")

    # 是否有漏洞
    @staticmethod
    def is_vulnerable(fix_versions, check_version, ignore_downstream=False):
        """Function determines if a version is vulnerable,
        by comparing to given fix versions by base release"""
        if ignore_downstream and CveUtils.is_downstream_version(check_version):
            return False

        vulnerable = False
        check_v = version.parse(check_version)
        base_check_v = CveUtils.get_base_release(check_v)

        version_compare_func = CveUtils.basic_compare
        if type(check_v) == version.LegacyVersion:
            version_compare_func = CveUtils.version_compare

        if check_version not in fix_versions:
            for fix_v in fix_versions:
                fix_v = version.parse(fix_v)
                base_fix_v = CveUtils.get_base_release(fix_v)

                if base_check_v == base_fix_v:
                    if version_compare_func(check_v, fix_v) == -1:
                        vulnerable = True
                        break

        if not vulnerable and version_compare_func(check_v, version.parse(fix_versions[0])) == -1:
            vulnerable = True

        return vulnerable

# 订阅K8sVersionDisclosure事件
@handler.subscribe_once(K8sVersionDisclosure)
class K8sClusterCveHunter(Hunter):
    """K8s CVE Hunter
    Checks if Node is running a Kubernetes version vulnerable to
    specific important CVEs
    """

    def __init__(self, event):
        self.event = event

    def execute(self):
        logger.debug(f"Checking known CVEs for k8s API version: {self.event.version}")
        cve_mapping = {
            ServerApiVersionEndPointAccessPE: ["1.10.11", "1.11.5", "1.12.3"],
            ServerApiVersionEndPointAccessDos: ["1.11.8", "1.12.6", "1.13.4"],
            ResetFloodHttp2Implementation: ["1.13.10", "1.14.6", "1.15.3"],
            PingFloodHttp2Implementation: ["1.13.10", "1.14.6", "1.15.3"],
            ServerApiClusterScopedResourcesAccess: ["1.13.9", "1.14.5", "1.15.2"],
        }
        for vulnerability, fix_versions in cve_mapping.items():
            if CveUtils.is_vulnerable(fix_versions, self.event.version, not config.include_patched_versions):
                self.publish_event(vulnerability(self.event.version))

# 订阅KubectlClientEvent事件
@handler.subscribe(KubectlClientEvent)
class KubectlCVEHunter(Hunter):
    """Kubectl CVE Hunter
    Checks if the kubectl client is vulnerable to specific important CVEs
    """

    def __init__(self, event):
        self.event = event

    def execute(self):
        cve_mapping = {
            KubectlCpVulnerability: ["1.11.9", "1.12.7", "1.13.5", "1.14.0"],
            IncompleteFixToKubectlCpVulnerability: ["1.12.9", "1.13.6", "1.14.2"],
        }
        logger.debug(f"Checking known CVEs for kubectl version: {self.event.version}")
        for vulnerability, fix_versions in cve_mapping.items():
            if CveUtils.is_vulnerable(fix_versions, self.event.version, not config.include_patched_versions):
                self.publish_event(vulnerability(binary_version=self.event.version))

```