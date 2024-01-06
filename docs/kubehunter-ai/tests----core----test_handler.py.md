# `kubehunter\tests\core\test_handler.py`

```
# 从 kube_hunter.core.events.handler 模块中导入 handler 函数
from kube_hunter.core.events.handler import handler
# 从 kube_hunter.modules.discovery.apiserver 模块中导入 ApiServiceDiscovery 类
from kube_hunter.modules.discovery.apiserver import ApiServiceDiscovery
# 从 kube_hunter.modules.discovery.dashboard 模块中导入 KubeDashboardDiscovery 类
from kube_hunter.modules.discovery.dashboard import KubeDashboard as KubeDashboardDiscovery
# 从 kube_hunter.modules.discovery.etcd 模块中导入 EtcdRemoteAccessDiscovery 类
from kube_hunter.modules.discovery.etcd import EtcdRemoteAccess as EtcdRemoteAccessDiscovery
# 从 kube_hunter.modules.discovery.hosts 模块中导入 FromPodHostDiscovery 和 HostDiscovery 类
from kube_hunter.modules.discovery.hosts import FromPodHostDiscovery, HostDiscovery
# 从 kube_hunter.modules.discovery.kubectl 模块中导入 KubectlClientDiscovery 类
from kube_hunter.modules.discovery.kubectl import KubectlClientDiscovery
# 从 kube_hunter.modules.discovery.kubelet 模块中导入 KubeletDiscovery 类
from kube_hunter.modules.discovery.kubelet import KubeletDiscovery
# 从 kube_hunter.modules.discovery.ports 模块中导入 PortDiscovery 类
from kube_hunter.modules.discovery.ports import PortDiscovery
# 从 kube_hunter.modules.discovery.proxy 模块中导入 KubeProxyDiscovery 类
from kube_hunter.modules.discovery.proxy import KubeProxy as KubeProxyDiscovery
# 从 kube_hunter.modules.hunting.aks 模块中导入 AzureSpnHunter 和 ProveAzureSpnExposure 类
from kube_hunter.modules.hunting.aks import AzureSpnHunter, ProveAzureSpnExposure
# 从 kube_hunter.modules.hunting.apiserver 模块中导入多个类
from kube_hunter.modules.hunting.apiserver import (
    AccessApiServer,
    ApiVersionHunter,
    AccessApiServerActive,
    AccessApiServerWithToken,
)
# 从 kube_hunter.modules.hunting.arp 模块中导入 ArpSpoofHunter 类
from kube_hunter.modules.hunting.arp import ArpSpoofHunter
# 从 kube_hunter.modules.hunting.capabilities 模块中导入 PodCapabilitiesHunter 类
from kube_hunter.modules.hunting.capabilities import PodCapabilitiesHunter
# 从 kube_hunter.modules.hunting.certificates 模块中导入 CertificateDiscovery 类
from kube_hunter.modules.hunting.certificates import CertificateDiscovery
# 从 kube_hunter.modules.hunting.cves 模块中导入 K8sClusterCveHunter 和 KubectlCVEHunter 类
from kube_hunter.modules.hunting.cves import K8sClusterCveHunter, KubectlCVEHunter
# 从 kube_hunter 模块中导入不同的猎手类
from kube_hunter.modules.hunting.dashboard import KubeDashboard
from kube_hunter.modules.hunting.dns import DnsSpoofHunter
from kube_hunter.modules.hunting.etcd import EtcdRemoteAccess, EtcdRemoteAccessActive
from kube_hunter.modules.hunting.kubelet import (
    ReadOnlyKubeletPortHunter,
    SecureKubeletPortHunter,
    ProveRunHandler,
    ProveContainerLogsHandler,
    ProveSystemLogs,
)
from kube_hunter.modules.hunting.mounts import VarLogMountHunter, ProveVarLogMount
from kube_hunter.modules.hunting.proxy import KubeProxy, ProveProxyExposed, K8sVersionDisclosureProve
from kube_hunter.modules.hunting.secrets import AccessSecrets

# 定义 PASSIVE_HUNTERS 集合，包含不同的猎手类
PASSIVE_HUNTERS = {
    ApiServiceDiscovery,
    KubeDashboardDiscovery,
    EtcdRemoteAccessDiscovery,
    FromPodHostDiscovery,
    HostDiscovery,
}
    KubectlClientDiscovery,  // 发现 Kubectl 客户端
    KubeletDiscovery,  // 发现 Kubelet
    PortDiscovery,  // 发现端口
    KubeProxyDiscovery,  // 发现 Kube 代理
    AzureSpnHunter,  // Azure SPN 搜索
    AccessApiServer,  // 访问 API 服务器
    AccessApiServerWithToken,  // 使用令牌访问 API 服务器
    ApiVersionHunter,  // API 版本搜索
    PodCapabilitiesHunter,  // Pod 能力搜索
    CertificateDiscovery,  // 证书发现
    K8sClusterCveHunter,  // K8s 集群 CVE 搜索
    KubectlCVEHunter,  // Kubectl CVE 搜索
    KubeDashboard,  // Kube 仪表板
    EtcdRemoteAccess,  // Etcd 远程访问
    ReadOnlyKubeletPortHunter,  // 只读 Kubelet 端口搜索
    SecureKubeletPortHunter,  // 安全 Kubelet 端口搜索
    VarLogMountHunter,  // VarLog 挂载搜索
    KubeProxy,  // Kube 代理
    AccessSecrets,  // 访问 Secrets
}
# 定义一个名为 ACTIVE_HUNTERS 的集合，包含了一系列活跃的猎手类
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

# 定义一个名为 remove_test_hunters 的函数，用于移除测试猎手类
def remove_test_hunters(hunters):
    # 使用集合推导式，遍历猎手类集合，移除名称以 "test" 开头的猎手类
    return {hunter for hunter in hunters if not hunter.__module__.startswith("test")}
# 测试被动猎手是否注册
def test_passive_hunters_registered():
    # 预期缺失的被动猎手集合
    expected_missing = set()
    # 预期奇怪的被动猎手集合
    expected_odd = set()

    # 从处理程序的被动猎手中移除测试猎手
    registered_passive = remove_test_hunters(handler.passive_hunters.keys())
    # 实际缺失的被动猎手集合
    actual_missing = PASSIVE_HUNTERS - registered_passive
    # 实际奇怪的被动猎手集合
    actual_odd = registered_passive - PASSIVE_HUNTERS

    # 断言预期缺失的被动猎手集合与实际缺失的被动猎手集合相等，否则抛出异常
    assert expected_missing == actual_missing, "Passive hunters are missing"
    # 断言预期奇怪的被动猎手集合与实际奇怪的被动猎手集合相等，否则抛出异常
    assert expected_odd == actual_odd, "Unexpected passive hunters are registered"

# TODO (#334): 无法测试主动猎手的注册，因为需要设置 `config.active`
# def test_active_hunters_registered():
#     expected_missing = set()
#     expected_odd = set()
#
#     registered_active = remove_test_hunters(handler.active_hunters.keys())
#     actual_missing = ACTIVE_HUNTERS - registered_active
#     actual_odd = registered_active - ACTIVE_HUNTERS
# 断言预期的缺失值与实际的缺失值相等，如果不相等则抛出"Active hunters are missing"的异常
# 断言预期的奇数与实际的奇数相等，如果不相等则抛出"Unexpected active hunters are registered"的异常

def test_all_hunters_registered():
    # TODO: 在测试中启用主动寻找模式
    # 预期值为被动猎人和主动猎人的并集
    expected = PASSIVE_HUNTERS
    # 从所有猎人中移除测试猎人，得到实际值
    actual = remove_test_hunters(handler.all_hunters.keys())

    # 断言预期值与实际值相等
    assert expected == actual
```