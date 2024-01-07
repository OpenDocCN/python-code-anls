# `KubiScan\unit_test.py`

```

# 导入所需的模块
import unittest
from engine import utils, privleged_containers
from engine.privleged_containers import get_privileged_containers
from api import api_client
from .KubiScan import get_all_affecting_cves_table_by_version
import json

# 定义一些容器和用户的列表
list_of_risky_containers = ["test1-yes", "test3-yes", "test5ac2-yes", "test6a-yes", "test6b-yes",
                            "test7c2-yes", "test8c-yes"]
list_of_not_risky_containers = ["test5ac1-no", "test1-no", "test2b-no", "test7c1-no"]

list_of_risky_users = ["kubiscan-sa"]
list_of_not_risky_users = ["kubiscan-sa2", "default"]

list_of_privileged_pods = ["etcd-minikube", "kube-apiserver-minikube", "kube-controller-manager-minikube",
                           "kube-scheduler-minikube", "storage-provisioner"]

# 定义版本和 CVE 相关的信息
version_dict = {"mid_version": "1.19.14",
                "above_all_version": "1.200.0",
                "under_all_version": "1.0.0"}

mid_version_cve = ["CVE-2021-25741", "CVE-2021-25749", "CVE-2022-3172"]

# 定义一系列函数来获取容器、用户和 CVE 相关的信息

# 定义测试类 TestKubiScan
class TestKubiScan(unittest.TestCase):
    # 初始化 API 客户端
    api_client.api_init()

    # 测试获取有风险的容器
    def test_get_risky_pods(self):
        risky_containers_by_name = get_containers_by_names()
        for container in list_of_risky_containers:
            self.assertIn(container, risky_containers_by_name)
        for container in list_of_not_risky_containers:
            self.assertNotIn(container, risky_containers_by_name)

    # 测试获取有风险的用户
    def test_get_all_risky_roles(self):
        risky_users_by_name = get_risky_users_by_name()
        for user in list_of_risky_users:
            self.assertIn(user, risky_users_by_name)
        for user in list_of_not_risky_users:
            self.assertNotIn(user, risky_users_by_name)

    # 测试获取特权容器
    def test_get_privileged_containers(self):
        pods = get_privileged_containers()
        string_list_of_privileged_pods = []
        for pod in pods:
            string_list_of_privileged_pods.append(pod.metadata.name)
        for pod_name in list_of_privileged_pods:
            self.assertIn(pod_name, string_list_of_privileged_pods)

    # 测试根据版本获取受影响的 CVE 表
    def test_get_all_affecting_cves_table_by_version(self):
        empty_table = get_all_affecting_cves_table_by_version(version_dict["above_all_version"])
        self.assertTrue(len(empty_table._rows) == 0)

        mid_cve_list_sorted = get_cve_list("mid_version")
        hard_coded_mid_version_cve_sorted = sorted(mid_version_cve)
        self.assertListEqual(hard_coded_mid_version_cve_sorted, mid_cve_list_sorted)

        all_cve_list_sorted = get_cve_list("under_all_version")
        all_cve_from_json = sorted(get_all_cve_from_json())
        self.assertListEqual(all_cve_list_sorted, all_cve_from_json)

# 运行测试
if __name__ == '__main__':
    unittest.main()

```