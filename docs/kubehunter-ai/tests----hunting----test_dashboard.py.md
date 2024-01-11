# `kubehunter\tests\hunting\test_dashboard.py`

```
# 导入 json 模块
import json

# 从 types 模块中导入 SimpleNamespace 类
from types import SimpleNamespace
# 从 requests_mock 模块中导入 Mocker 类
from requests_mock import Mocker
# 从 kube_hunter.modules.hunting.dashboard 模块中导入 KubeDashboard 类
from kube_hunter.modules.hunting.dashboard import KubeDashboard


# 定义 TestKubeDashboard 类
class TestKubeDashboard:
    # 定义静态方法 get_nodes_mock，接受一个字典类型的 result 参数和任意关键字参数
    @staticmethod
    def get_nodes_mock(result: dict, **kwargs):
        # 使用 Mocker 创建一个上下文环境
        with Mocker() as m:
            # 使用 Mocker 模拟发送 GET 请求，返回 result 参数的 JSON 字符串形式
            m.get("http://mockdashboard:8000/api/v1/node", text=json.dumps(result), **kwargs)
            # 创建 KubeDashboard 对象
            hunter = KubeDashboard(SimpleNamespace(host="mockdashboard", port=8000))
            # 调用 KubeDashboard 对象的 get_nodes 方法
            return hunter.get_nodes()

    # 定义静态方法 test_get_nodes_with_result
    @staticmethod
    def test_get_nodes_with_result():
        # 定义 nodes 变量，包含一个包含一个字典的列表
        nodes = {"nodes": [{"objectMeta": {"name": "node1"}}]}
        # 定义期望结果列表
        expected = ["node1"]
        # 调用 get_nodes_mock 方法，传入 nodes 变量，获取实际结果
        actual = TestKubeDashboard.get_nodes_mock(nodes)

        # 断言期望结果和实际结果是否相等
        assert expected == actual

    # 定义静态方法 test_get_nodes_without_result
    @staticmethod
    def test_get_nodes_without_result():
        # 定义 nodes 变量，包含一个空列表
        nodes = {"nodes": []}
        # 定义期望结果为空列表
        expected = []
        # 调用 get_nodes_mock 方法，传入 nodes 变量，获取实际结果
        actual = TestKubeDashboard.get_nodes_mock(nodes)

        # 断言期望结果和实际结果是否相等
        assert expected == actual

    # 定义静态方法 test_get_nodes_invalid_result
    @staticmethod
    def test_get_nodes_invalid_result():
        # 定义期望结果为 None
        expected = None
        # 调用 get_nodes_mock 方法，传入空字典和 status_code 参数为 404，获取实际结果
        actual = TestKubeDashboard.get_nodes_mock(dict(), status_code=404)

        # 断言期望结果和实际结果是否相等
        assert expected == actual
```