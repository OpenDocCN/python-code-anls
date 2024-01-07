# `.\kubehunter\tests\core\test_cloud.py`

```

# 导入requests_mock模块和json模块
import requests_mock
import json

# 导入NewHostEvent事件类型
from kube_hunter.core.events.types import NewHostEvent

# 测试如果云类型已经设置，则不尝试运行get_cloud函数
# get_cloud(1.2.3.4)将导致错误
def test_presetcloud():
    # 预期的云类型为"AWS"
    expcted = "AWS"
    # 创建NewHostEvent对象，设置主机和云类型
    hostEvent = NewHostEvent(host="1.2.3.4", cloud=expcted)
    # 断言预期的云类型与实际的云类型相等
    assert expcted == hostEvent.cloud

# 测试获取云类型
def test_getcloud():
    # 设置假主机IP
    fake_host = "1.2.3.4"
    # 预期的云类型为"Azure"
    expected_cloud = "Azure"
    # 模拟API返回的结果
    result = {"cloud": expected_cloud}

    # 使用requests_mock模拟HTTP请求
    with requests_mock.mock() as m:
        # 模拟GET请求返回结果
        m.get(f"https://api.azurespeed.com/api/region?ipOrUrl={fake_host}", text=json.dumps(result))
        # 创建NewHostEvent对象，设置主机IP
        hostEvent = NewHostEvent(host=fake_host)
        # 断言主机的云类型与预期的云类型相等
        assert hostEvent.cloud == expected_cloud

```