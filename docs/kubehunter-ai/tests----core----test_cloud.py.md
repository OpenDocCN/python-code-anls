# `kubehunter\tests\core\test_cloud.py`

```
# 导入requests_mock和json模块
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
    # 假的主机IP地址
    fake_host = "1.2.3.4"
    # 预期的云类型为"Azure"
    expected_cloud = "Azure"
    # 模拟的结果数据
    result = {"cloud": expected_cloud}

    # 使用requests_mock模拟HTTP请求
    with requests_mock.mock() as m:
# 发送 GET 请求到指定的 API 地址，使用假的主机名作为参数，返回结果为 result
m.get(f"https://api.azurespeed.com/api/region?ipOrUrl={fake_host}", text=json.dumps(result))
# 创建一个新的主机事件对象，使用假的主机名作为参数
hostEvent = NewHostEvent(host=fake_host)
# 断言主机事件对象的云属性是否等于预期的云
assert hostEvent.cloud == expected_cloud
```