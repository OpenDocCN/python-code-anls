# `.\kubehunter\tests\core\test_handler.py`

```

# 从 kube_hunter.core.events.handler 模块中导入 handler 函数
from kube_hunter.core.events.handler import handler
# 从不同模块中导入相关的类
from kube_hunter.modules.discovery.apiserver import ApiServiceDiscovery
from kube_hunter.modules.discovery.dashboard import KubeDashboard as KubeDashboardDiscovery
from kube_hunter.modules.discovery.etcd import EtcdRemoteAccess as EtcdRemoteAccessDiscovery
from kube_hunter.modules.discovery.hosts import FromPodHostDiscovery, HostDiscovery
from kube_hunter.modules.discovery.kubectl import KubectlClientDiscovery
from kube_hunter.modules.discovery.kubelet import KubeletDiscovery
from kube_hunter.modules.discovery.ports import PortDiscovery
from kube_hunter.modules.discovery.proxy import KubeProxy as KubeProxyDiscovery
from kube_hunter.modules.hunting.aks import AzureSpnHunter, ProveAzureSpnExposure
from kube_hunter.modules.hunting.apiserver import (
    AccessApiServer,
    ApiVersionHunter,
    AccessApiServerActive,
    AccessApiServerWithToken,
)
from kube_hunter.modules.hunting.arp import ArpSpoofHunter
from kube_hunter.modules.hunting.capabilities import PodCapabilitiesHunter
from kube_hunter.modules.hunting.certificates import CertificateDiscovery
from kube_hunter.modules.hunting.cves import K8sClusterCveHunter, KubectlCVEHunter
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

# 定义 PASSIVE_HUNTERS 集合，包含 passvie 模式下的所有 hunter 类
PASSIVE_HUNTERS = {
    ApiServiceDiscovery,
    KubeDashboardDiscovery,
    EtcdRemoteAccessDiscovery,
    FromPodHostDiscovery,
    HostDiscovery,
    KubectlClientDiscovery,
    KubeletDiscovery,
    PortDiscovery,
    KubeProxyDiscovery,
    AzureSpnHunter,
    AccessApiServer,
    AccessApiServerWithToken,
    ApiVersionHunter,
    PodCapabilitiesHunter,
    CertificateDiscovery,
    K8sClusterCveHunter,
    KubectlCVEHunter,
    KubeDashboard,
    EtcdRemoteAccess,
    ReadOnlyKubeletPortHunter,
    SecureKubeletPortHunter,
    VarLogMountHunter,
    KubeProxy,
    AccessSecrets,
}

# 定义 ACTIVE_HUNTERS 集合，包含 active 模式下的所有 hunter 类
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

# 定义函数 remove_test_hunters，用于移除测试用例中的 hunter
def remove_test_hunters(hunters):
    return {hunter for hunter in hunters if not hunter.__module__.startswith("test")}

# 定义测试函数 test_passive_hunters_registered，用于测试 passive 模式下的 hunter 是否注册成功
def test_passive_hunters_registered():
    expected_missing = set()
    expected_odd = set()

    # 移除测试用例中的 hunter
    registered_passive = remove_test_hunters(handler.passive_hunters.keys())
    # 检查 PASSIVE_HUNTERS 中是否有未注册的 hunter
    actual_missing = PASSIVE_HUNTERS - registered_passive
    # 检查是否有意外注册的 hunter
    actual_odd = registered_passive - PASSIVE_HUNTERS

    # 断言，检查是否有未注册的 hunter 和意外注册的 hunter
    assert expected_missing == actual_missing, "Passive hunters are missing"
    assert expected_odd == actual_odd, "Unexpected passive hunters are registered"

# 定义测试函数 test_all_hunters_registered，用于测试所有模式下的 hunter 是否注册成功
def test_all_hunters_registered():
    # TODO: Enable active hunting mode in testing
    # expected = PASSIVE_HUNTERS | ACTIVE_HUNTERS
    expected = PASSIVE_HUNTERS
    # 移除测试用例中的 hunter
    actual = remove_test_hunters(handler.all_hunters.keys())

    # 断言，检查是否所有模式下的 hunter 都注册成功
    assert expected == actual

```