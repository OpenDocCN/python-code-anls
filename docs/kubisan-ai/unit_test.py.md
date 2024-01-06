# `KubiScan\unit_test.py`

```
# 导入单元测试模块
import unittest
# 从 engine 模块中导入 utils 和 privleged_containers
from engine import utils, privleged_containers
# 从 engine.privleged_containers 模块中导入 get_privileged_containers
from engine.privleged_containers import get_privileged_containers
# 从 api 模块中导入 api_client
from api import api_client
# 从当前目录下的 KubiScan 模块中导入 get_all_affecting_cves_table_by_version
from .KubiScan import get_all_affecting_cves_table_by_version
# 导入 json 模块

# 定义一个包含有风险容器名称的列表
list_of_risky_containers = ["test1-yes", "test3-yes", "test5ac2-yes", "test6a-yes", "test6b-yes",
                            "test7c2-yes", "test8c-yes"]
# 定义一个不包含风险容器名称的列表
list_of_not_risky_containers = ["test5ac1-no", "test1-no", "test2b-no", "test7c1-no"]
# 定义一个包含有风险用户的列表
list_of_risky_users = ["kubiscan-sa"]
# 定义一个不包含风险用户的列表
list_of_not_risky_users = ["kubiscan-sa2", "default"]
# 定义一个包含有特权 pod 的列表
list_of_privileged_pods = ["etcd-minikube", "kube-apiserver-minikube", "kube-controller-manager-minikube",
                           "kube-scheduler-minikube", "storage-provisioner"]

# 定义一个版本字典，包含中间版本和最新版本
version_dict = {"mid_version": "1.19.14",
                "above_all_version": "1.200.0",
# 定义一个字典，包含了所有版本的软件及其对应的漏洞信息
all_version_cve = {
    "latest_version": "2.0.0",
    "mid_version": "1.5.0",
    "under_all_version": "1.0.0"
}

# 定义一个列表，包含了中间版本的软件对应的漏洞信息
mid_version_cve = ["CVE-2021-25741", "CVE-2021-25749", "CVE-2022-3172"]

# 定义一个函数，用于获取所有容器的名称
def get_containers_by_names():
    # 调用utils模块的get_risky_pods函数，获取所有风险容器
    risky_pods = utils.get_risky_pods()
    risky_containers_by_name = []
    # 遍历每个风险容器，获取其名称并添加到列表中
    for risky_pod in risky_pods or []:
        for container in risky_pod.containers:
            risky_containers_by_name.append(container.name)
    return risky_containers_by_name

# 定义一个函数，用于获取所有风险用户的名称
def get_risky_users_by_name():
    # 调用utils模块的get_all_risky_subjects函数，获取所有风险用户
    risky_users = utils.get_all_risky_subjects()
    risky_users_by_name = []
    # 遍历每个风险用户，获取其名称并添加到列表中
    for risky_user in risky_users:
        risky_users_by_name.append(risky_user.user_info.name)
    return risky_users_by_name
# 根据版本状态获取受影响的 CVE 列表
def get_cve_list(version_status):
    # 根据版本状态获取受影响 CVE 的表格
    version_table = get_all_affecting_cves_table_by_version(version_dict[version_status])
    # 初始化 CVE 列表
    cve_list = []
    # 遍历表格中的每一行
    for row in version_table:
        # 设置行的边框和标题为 False
        row.border = False
        row.header = False
        # 获取每行的 CVE 字段，并去除空格后添加到 CVE 列表中
        cve_list.append(row.get_string(fields=['CVE']).strip())
    # 返回排序后的 CVE 列表
    return sorted(cve_list)

# 从 JSON 文件中获取所有的 CVE
def get_all_cve_from_json():
    # 打开 CVE.json 文件，并读取其中的数据
    with open('CVE.json', 'r') as f:
        data = json.load(f)
    # 初始化所有 CVE 的列表
    all_cves = []
    # 遍历 JSON 数据中的每个 CVE
    for cve in data["CVES"]:
        # 将每个 CVE 的编号添加到所有 CVE 的列表中
        all_cves.append(cve["CVENumber"])
    # 返回所有 CVE 的列表
    return all_cves
# 定义一个测试类 TestKubiScan，继承自 unittest.TestCase
class TestKubiScan(unittest.TestCase):
    # 初始化 API 客户端
    api_client.api_init()

    # 测试获取风险容器的方法
    def test_get_risky_pods(self):
        # 通过容器名称获取风险容器
        risky_containers_by_name = get_containers_by_names()
        # 遍历风险容器列表，检查是否在获取的风险容器中
        for container in list_of_risky_containers:
            self.assertIn(container, risky_containers_by_name)
        # 遍历非风险容器列表，检查是否不在获取的风险容器中
        for container in list_of_not_risky_containers:
            self.assertNotIn(container, risky_containers_by_name)

    # 测试获取所有风险角色的方法
    def test_get_all_risky_roles(self):
        # 获取风险用户列表
        risky_users_by_name = get_risky_users_by_name()
        # 遍历风险用户列表，检查是否在获取的风险用户中
        for user in list_of_risky_users:
            self.assertIn(user, risky_users_by_name)
        # 遍历非风险用户列表，检查是否不在获取的风险用户中
        for user in list_of_not_risky_users:
            self.assertNotIn(user, risky_users_by_name)

    # 测试获取特权容器的方法
    def test_get_privileged_containers(self):
        # 获取特权容器列表
        pods = get_privileged_containers()
# 创建一个空列表来存储特权 pod 的名称
string_list_of_privileged_pods = []
# 遍历 pods 列表，将每个 pod 的名称添加到特权 pod 名称列表中
for pod in pods:
    string_list_of_privileged_pods.append(pod.metadata.name)
# 遍历特权 pod 名称列表，确保每个特权 pod 的名称都在 string_list_of_privileged_pods 中
for pod_name in list_of_privileged_pods:
    self.assertIn(pod_name, string_list_of_privileged_pods)

# 测试获取所有受影响 CVE 表格的版本
def test_get_all_affecting_cves_table_by_version(self):
    # 获取指定版本的受影响 CVE 表格，确保表格中没有行
    empty_table = get_all_affecting_cves_table_by_version(version_dict["above_all_version"])
    self.assertTrue(len(empty_table._rows) == 0)

    # 获取中间版本的 CVE 列表，并对其进行排序
    mid_cve_list_sorted = get_cve_list("mid_version")
    # 获取硬编码的中间版本 CVE 列表，并对其进行排序
    hard_coded_mid_version_cve_sorted = sorted(mid_version_cve)
    # 确保获取的中间版本 CVE 列表与硬编码的中间版本 CVE 列表相等
    self.assertListEqual(hard_coded_mid_version_cve_sorted, mid_cve_list_sorted)

    # 获取所有版本的 CVE 列表，并对其进行排序
    all_cve_list_sorted = get_cve_list("under_all_version")
    # 从 JSON 中获取所有 CVE，并对其进行排序
    all_cve_from_json = sorted(get_all_cve_from_json())
    # 确保获取的所有版本 CVE 列表与从 JSON 中获取的所有 CVE 列表相等
    self.assertListEqual(all_cve_list_sorted, all_cve_from_json)

# 如果是主程序入口
if __name__ == '__main__':
# 运行单元测试的主程序入口，用于执行所有测试用例
```