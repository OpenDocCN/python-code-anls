# `kubehunter\tests\hunting\test_dashboard.py`

```
# 导入所需的模块
import json
from types import SimpleNamespace
from requests_mock import Mocker
from kube_hunter.modules.hunting.dashboard import KubeDashboard

# 定义测试类 TestKubeDashboard
class TestKubeDashboard:
    # 定义静态方法 get_nodes_mock，用于模拟获取节点信息的请求
    @staticmethod
    def get_nodes_mock(result: dict, **kwargs):
        # 使用 requests_mock 模拟 HTTP GET 请求，返回指定的结果
        with Mocker() as m:
            m.get("http://mockdashboard:8000/api/v1/node", text=json.dumps(result), **kwargs)
            # 创建 KubeDashboard 对象
            hunter = KubeDashboard(SimpleNamespace(host="mockdashboard", port=8000))
            # 调用 KubeDashboard 对象的 get_nodes 方法，返回节点信息
            return hunter.get_nodes()

    # 定义静态方法 test_get_nodes_with_result，用于测试获取节点信息的方法
    @staticmethod
    def test_get_nodes_with_result():
        # 模拟的节点信息
        nodes = {"nodes": [{"objectMeta": {"name": "node1"}}]}
        # 期望的节点列表
        expected = ["node1"]
        # 调用 get_nodes_mock 方法，获取实际的节点列表
        actual = TestKubeDashboard.get_nodes_mock(nodes)
# 断言预期值和实际值相等
assert expected == actual

# 测试获取没有结果的节点
@staticmethod
def test_get_nodes_without_result():
    # 创建一个空的节点字典
    nodes = {"nodes": []}
    # 期望的结果是一个空列表
    expected = []
    # 调用 get_nodes_mock 方法，获取实际结果
    actual = TestKubeDashboard.get_nodes_mock(nodes)
    # 断言预期值和实际值相等
    assert expected == actual

# 测试获取无效结果的节点
@staticmethod
def test_get_nodes_invalid_result():
    # 期望的结果是 None
    expected = None
    # 调用 get_nodes_mock 方法，传入一个空的字典和状态码 404，获取实际结果
    actual = TestKubeDashboard.get_nodes_mock(dict(), status_code=404)
    # 断言预期值和实际值相等
    assert expected == actual
```