# `kubehunter\tests\core\test_handler.py`

```py
# 从 kube_hunter.core.events.handler 模块导入 handler 函数
from kube_hunter.core.events.handler import handler
# 从 kube_hunter.modules.discovery.apiserver 模块导入 ApiServiceDiscovery 类
from kube_hunter.modules.discovery.apiserver import ApiServiceDiscovery
# 从 kube_hunter.modules.discovery.dashboard 模块导入 KubeDashboard 类作为 KubeDashboardDiscovery
from kube_hunter.modules.discovery.dashboard import KubeDashboard as KubeDashboardDiscovery
# 从 kube_hunter.modules.discovery.etcd 模块导入 EtcdRemoteAccess 类作为 EtcdRemoteAccessDiscovery
from kube_hunter.modules.discovery.etcd import EtcdRemoteAccess as EtcdRemoteAccessDiscovery
# 从 kube_hunter.modules.discovery.hosts 模块导入 FromPodHostDiscovery 和 HostDiscovery 类
from kube_hunter.modules.discovery.hosts import FromPodHostDiscovery, HostDiscovery
# 从 kube_hunter.modules.discovery.kubectl 模块导入 KubectlClientDiscovery 类
from kube_hunter.modules.discovery.kubectl import KubectlClientDiscovery
# 从 kube_hunter.modules.discovery.kubelet 模块导入 KubeletDiscovery 类
from kube_hunter.modules.discovery.kubelet import KubeletDiscovery
# 从 kube_hunter.modules.discovery.ports 模块导入 PortDiscovery 类
from kube_hunter.modules.discovery.ports import PortDiscovery
# 从 kube_hunter.modules.discovery.proxy 模块导入 KubeProxy 类作为 KubeProxyDiscovery
from kube_hunter.modules.discovery.proxy import KubeProxy as KubeProxyDiscovery
# 从 kube_hunter.modules.hunting.aks 模块导入 AzureSpnHunter 和 ProveAzureSpnExposure 类
from kube_hunter.modules.hunting.aks import AzureSpnHunter, ProveAzureSpnExposure
# 从 kube_hunter.modules.hunting.apiserver 模块导入多个类
from kube_hunter.modules.hunting.apiserver import (
    AccessApiServer,
    ApiVersionHunter,
    AccessApiServerActive,
    AccessApiServerWithToken,
)
# 从 kube_hunter.modules.hunting.arp 模块导入 ArpSpoofHunter 类
from kube_hunter.modules.hunting.arp import ArpSpoofHunter
# 从 kube_hunter.modules.hunting.capabilities 模块导入 PodCapabilitiesHunter 类
from kube_hunter.modules.hunting.capabilities import PodCapabilitiesHunter
# 从 kube_hunter.modules.hunting.certificates 模块导入 CertificateDiscovery 类
from kube_hunter.modules.hunting.certificates import CertificateDiscovery
# 从 kube_hunter.modules.hunting.cves 模块导入 K8sClusterCveHunter 和 KubectlCVEHunter 类
from kube_hunter.modules.hunting.cves import K8sClusterCveHunter, KubectlCVEHunter
# 从 kube_hunter.modules.hunting.dashboard 模块导入 KubeDashboard 类
from kube_hunter.modules.hunting.dashboard import KubeDashboard
# 从 kube_hunter.modules.hunting.dns 模块导入 DnsSpoofHunter 类
from kube_hunter.modules.hunting.dns import DnsSpoofHunter
# 从 kube_hunter.modules.hunting.etcd 模块导入 EtcdRemoteAccess 和 EtcdRemoteAccessActive 类
from kube_hunter.modules.hunting.etcd import EtcdRemoteAccess, EtcdRemoteAccessActive
# 从 kube_hunter.modules.hunting.kubelet 模块导入多个类
from kube_hunter.modules.hunting.kubelet import (
    ReadOnlyKubeletPortHunter,
    SecureKubeletPortHunter,
    ProveRunHandler,
    ProveContainerLogsHandler,
    ProveSystemLogs,
)
# 从 kube_hunter.modules.hunting.mounts 模块导入 VarLogMountHunter 和 ProveVarLogMount 类
from kube_hunter.modules.hunting.mounts import VarLogMountHunter, ProveVarLogMount
# 从 kube_hunter.modules.hunting.proxy 模块导入 KubeProxy 和 ProveProxyExposed 类
from kube_hunter.modules.hunting.proxy import KubeProxy, ProveProxyExposed, K8sVersionDisclosureProve
# 从 kube_hunter.modules.hunting.secrets 模块导入 AccessSecrets 类

from kube_hunter.modules.hunting.secrets import AccessSecrets

# 定义 PASSIVE_HUNTERS 集合，包含多个发现和猎取的类
PASSIVE_HUNTERS = {
    ApiServiceDiscovery,
    KubeDashboardDiscovery,
    EtcdRemoteAccessDiscovery,
    FromPodHostDiscovery,
    HostDiscovery,
    KubectlClientDiscovery,  # 发现 Kubectl 客户端
    KubeletDiscovery,  # 发现 Kubelet
    PortDiscovery,  # 发现端口
    KubeProxyDiscovery,  # 发现 Kube 代理
    AzureSpnHunter,  # Azure SPN 搜索
    AccessApiServer,  # 访问 API 服务器
    AccessApiServerWithToken,  # 使用令牌访问 API 服务器
    ApiVersionHunter,  # API 版本搜索
    PodCapabilitiesHunter,  # Pod 能力搜索
    CertificateDiscovery,  # 证书搜索
    K8sClusterCveHunter,  # K8s 集群 CVE 搜索
    KubectlCVEHunter,  # Kubectl CVE 搜索
    KubeDashboard,  # Kube 仪表板
    EtcdRemoteAccess,  # Etcd 远程访问
    ReadOnlyKubeletPortHunter,  # 只读 Kubelet 端口搜索
    SecureKubeletPortHunter,  # 安全 Kubelet 端口搜索
    VarLogMountHunter,  # VarLog 挂载搜索
    KubeProxy,  # Kube 代理
    AccessSecrets,  # 访问 Secrets
# 定义活跃猎人的集合，包括多个猎人类
ACTIVE_HUNTERS = {
    ProveAzureSpnExposure,
    AccessApiServerActive,
    ArpSpoofHunter,
    DnsSpoofHunter,
    EtcdRemoteAccessActive,
    ProveRunHandler,
    ProveContainerLogsHandler,
    ProveSystemLogs,
    ProveVarLogMount,
    ProveProxyExposed,
    K8sVersionDisclosureProve,
}

# 从猎人集合中移除测试猎人
def remove_test_hunters(hunters):
    return {hunter for hunter in hunters if not hunter.__module__.startswith("test")}

# 测试被动猎人是否注册
def test_passive_hunters_registered():
    expected_missing = set()
    expected_odd = set()

    # 从处理程序中移除测试猎人，得到已注册的被动猎人集合
    registered_passive = remove_test_hunters(handler.passive_hunters.keys())
    # 计算缺失的被动猎人
    actual_missing = PASSIVE_HUNTERS - registered_passive
    # 计算意外注册的被动猎人
    actual_odd = registered_passive - PASSIVE_HUNTERS

    # 断言缺失的被动猎人是否为空，否则输出错误信息
    assert expected_missing == actual_missing, "Passive hunters are missing"
    # 断言意外注册的被动猎人是否为空，否则输出错误信息
    assert expected_odd == actual_odd, "Unexpected passive hunters are registered"

# 测试所有猎人是否注册
def test_all_hunters_registered():
    # TODO: 在测试中启用活跃猎人模式
    # 期望的猎人集合包括被动猎人和活跃猎人
    expected = PASSIVE_HUNTERS
    # 实际已注册的猎人集合
    actual = remove_test_hunters(handler.all_hunters.keys())

    # 断言期望的猎人集合和实际的猎人集合是否相等
    assert expected == actual
```