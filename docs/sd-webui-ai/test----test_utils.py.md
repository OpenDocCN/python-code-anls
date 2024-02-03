# `stable-diffusion-webui\test\test_utils.py`

```
# 导入 pytest 和 requests 模块
import pytest
import requests

# 测试写入选项的函数
def test_options_write(base_url):
    # 构建获取选项的 URL
    url_options = f"{base_url}/sdapi/v1/options"
    # 发送 GET 请求获取响应
    response = requests.get(url_options)
    # 断言响应状态码为 200
    assert response.status_code == 200

    # 获取先前的 send_seed 值
    pre_value = response.json()["send_seed"]

    # 发送 POST 请求更新 send_seed 值，并断言响应状态码为 200
    assert requests.post(url_options, json={'send_seed': (not pre_value)}).status_code == 200

    # 再次发送 GET 请求获取响应，断言状态码为 200，并检查 send_seed 值是否更新
    response = requests.get(url_options)
    assert response.status_code == 200
    assert response.json()['send_seed'] == (not pre_value)

    # 恢复原始的 send_seed 值
    requests.post(url_options, json={"send_seed": pre_value})

# 参数化测试获取 API URL 的函数
@pytest.mark.parametrize("url", [
    "sdapi/v1/cmd-flags",
    "sdapi/v1/samplers",
    "sdapi/v1/upscalers",
    "sdapi/v1/sd-models",
    "sdapi/v1/hypernetworks",
    "sdapi/v1/face-restorers",
    "sdapi/v1/realesrgan-models",
    "sdapi/v1/prompt-styles",
    "sdapi/v1/embeddings",
])
def test_get_api_url(base_url, url):
    # 发送 GET 请求获取 API URL 的响应，断言状态码为 200
    assert requests.get(f"{base_url}/{url}").status_code == 200
```