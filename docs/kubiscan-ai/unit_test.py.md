# `KubiScan\unit_test.py`

```py
# 导入单元测试模块
import unittest
# 从 engine 模块中导入 utils 和 privleged_containers
from engine import utils, privleged_containers
# 从 engine.privleged_containers 模块中导入 get_privileged_containers 函数
from engine.privleged_containers import get_privileged_containers
# 从 api 模块中导入 api_client
from api import api_client
# 从当前目录下的 KubiScan 模块中导入 get_all_affecting_cves_table_by_version 函数
from .KubiScan import get_all_affecting_cves_table_by_version
# 导入 json 模块

# 定义一个包含有风险容器名称的列表
list_of_risky_containers = ["test1-yes", "test3-yes", "test5ac2-yes", "test6a-yes", "test6b-yes",
                            "test7c2-yes", "test8c-yes"]
# 定义一个不包含有风险容器名称的列表
list_of_not_risky_containers = ["test5ac1-no", "test1-no", "test2b-no", "test7c1-no"]

# 定义一个包含有风险用户的列表
list_of_risky_users = ["kubiscan-sa"]
# 定义一个不包含有风险用户的列表
list_of_not_risky_users = ["kubiscan-sa2", "default"]

# 定义一个包含特权 pod 名称的列表
list_of_privileged_pods = ["etcd-minikube", "kube-apiserver-minikube", "kube-controller-manager-minikube",
                           "kube-scheduler-minikube", "storage-provisioner"]

# 定义一个版本字典
version_dict = {"mid_version": "1.19.14",
                "above_all_version": "1.200.0",
                "under_all_version": "1.0.0"}

# 定义一个中间版本的 CVE 列表
mid_version_cve = ["CVE-2021-25741", "CVE-2021-25749", "CVE-2022-3172"]

# 定义一个函数，获取风险容器的名称
def get_containers_by_names():
    # 获取所有风险 pod
    risky_pods = utils.get_risky_pods()
    risky_containers_by_name = []
    # 遍历每个风险 pod，并获取其容器名称
    for risky_pod in risky_pods or []:
        for container in risky_pod.containers:
            risky_containers_by_name.append(container.name)
    return risky_containers_by_name

# 定义一个函数，获取风险用户的名称
def get_risky_users_by_name():
    # 获取所有风险用户
    risky_users = utils.get_all_risky_subjects()
    risky_users_by_name = []
    # 遍历每个风险用户，并获取其名称
    for risky_user in risky_users:
        risky_users_by_name.append(risky_user.user_info.name)
    return risky_users_by_name

# 定义一个函数，获取特定版本的 CVE 列表
def get_cve_list(version_status):
    # 获取特定版本的 CVE 表格
    version_table = get_all_affecting_cves_table_by_version(version_dict[version_status])
    cve_list = []
    # 遍历表格中的每一行，并获取 CVE 号码
    for row in version_table:
        row.border = False
        row.header = False
        cve_list.append(row.get_string(fields=['CVE']).strip())
    return sorted(cve_list)

# 定义一个函数，从 JSON 文件中获取所有 CVE
def get_all_cve_from_json():
    with open('CVE.json', 'r') as f:
        data = json.load(f)
    all_cves = []
    # 遍历 JSON 数据中的每个 CVE，并获取其 CVE 号码
    for cve in data["CVES"]:
        all_cves.append(cve["CVENumber"])
    # 返回变量 all_cves 的值
    return all_cves
# 定义测试类 TestKubiScan，继承自 unittest.TestCase
class TestKubiScan(unittest.TestCase):
    # 调用 api_init 方法初始化 API 客户端
    api_client.api_init()

    # 测试获取风险容器的方法
    def test_get_risky_pods(self):
        # 调用 get_containers_by_names 方法获取风险容器的字典
        risky_containers_by_name = get_containers_by_names()
        # 遍历风险容器列表，检查是否在风险容器字典中
        for container in list_of_risky_containers:
            self.assertIn(container, risky_containers_by_name)
        # 遍历非风险容器列表，检查是否不在风险容器字典中
        for container in list_of_not_risky_containers:
            self.assertNotIn(container, risky_containers_by_name)

    # 测试获取所有风险角色的方法
    def test_get_all_risky_roles(self):
        # 调用 get_risky_users_by_name 方法获取风险用户的字典
        risky_users_by_name = get_risky_users_by_name()
        # 遍历风险用户列表，检查是否在风险用户字典中
        for user in list_of_risky_users:
            self.assertIn(user, risky_users_by_name)
        # 遍历非风险用户列表，检查是否不在风险用户字典中
        for user in list_of_not_risky_users:
            self.assertNotIn(user, risky_users_by_name)

    # 测试获取特权容器的方法
    def test_get_privileged_containers(self):
        # 调用 get_privileged_containers 方法获取特权容器列表
        pods = get_privileged_containers()
        # 将特权容器列表中的名称转换为字符串列表
        string_list_of_privileged_pods = []
        for pod in pods:
            string_list_of_privileged_pods.append(pod.metadata.name)
        # 遍历特权容器名称列表，检查是否在特权容器字符串列表中
        for pod_name in list_of_privileged_pods:
            self.assertIn(pod_name, string_list_of_privileged_pods)

    # 测试根据版本获取所有影响 CVE 表的方法
    def test_get_all_affecting_cves_table_by_version(self):
        # 根据版本字典中的版本号获取影响 CVE 表，检查是否为空
        empty_table = get_all_affecting_cves_table_by_version(version_dict["above_all_version"])
        self.assertTrue(len(empty_table._rows) == 0)

        # 获取中间版本的 CVE 列表并排序
        mid_cve_list_sorted = get_cve_list("mid_version")
        hard_coded_mid_version_cve_sorted = sorted(mid_version_cve)
        # 检查获取的中间版本 CVE 列表与硬编码的中间版本 CVE 列表是否相等
        self.assertListEqual(hard_coded_mid_version_cve_sorted, mid_cve_list_sorted)

        # 获取所有 CVE 列表并排序
        all_cve_list_sorted = get_cve_list("under_all_version")
        all_cve_from_json = sorted(get_all_cve_from_json())
        # 检查获取的所有 CVE 列表与从 JSON 中获取的所有 CVE 列表是否相等
        self.assertListEqual(all_cve_list_sorted, all_cve_from_json)

# 如果是主程序入口，则执行测试
if __name__ == '__main__':
    unittest.main()
```