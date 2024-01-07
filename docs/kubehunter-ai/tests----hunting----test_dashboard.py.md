# `.\kubehunter\tests\hunting\test_dashboard.py`

```

# 导入所需的模块
import json
from types import SimpleNamespace
from requests_mock import Mocker
from kube_hunter.modules.hunting.dashboard import KubeDashboard

# 定义测试类 TestKubeDashboard
class TestKubeDashboard:
    # 定义静态方法 get_nodes_mock，用于模拟获取节点信息
    @staticmethod
    def get_nodes_mock(result: dict, **kwargs):
        # 使用 Mocker 创建一个模拟对象
        with Mocker() as m:
            # 使用模拟对象发送 GET 请求，返回模拟数据
            m.get("http://mockdashboard:8000/api/v1/node", text=json.dumps(result), **kwargs)
            # 创建 KubeDashboard 对象
            hunter = KubeDashboard(SimpleNamespace(host="mockdashboard", port=8000))
            # 调用 KubeDashboard 对象的 get_nodes 方法，返回节点信息
            return hunter.get_nodes()

    # 定义静态方法 test_get_nodes_with_result，测试获取节点信息并有结果的情况
    @staticmethod
    def test_get_nodes_with_result():
        # 模拟节点信息
        nodes = {"nodes": [{"objectMeta": {"name": "node1"}}]}
        expected = ["node1"]
        # 调用 get_nodes_mock 方法获取实际结果
        actual = TestKubeDashboard.get_nodes_mock(nodes)
        # 断言实际结果与预期结果相等
        assert expected == actual

    # 定义静态方法 test_get_nodes_without_result，测试获取节点信息并无结果的情况
    @staticmethod
    def test_get_nodes_without_result():
        # 模拟节点信息
        nodes = {"nodes": []}
        expected = []
        # 调用 get_nodes_mock 方法获取实际结果
        actual = TestKubeDashboard.get_nodes_mock(nodes)
        # 断言实际结果与预期结果相等
        assert expected == actual

    # 定义静态方法 test_get_nodes_invalid_result，测试获取节点信息并返回无效结果的情况
    @staticmethod
    def test_get_nodes_invalid_result():
        expected = None
        # 调用 get_nodes_mock 方法获取实际结果
        actual = TestKubeDashboard.get_nodes_mock(dict(), status_code=404)
        # 断言实际结果与预期结果相等
        assert expected == actual

```