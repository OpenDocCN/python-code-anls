# `stable-diffusion-webui\test\test_extras.py`

```py
# 导入 requests 模块
import requests

# 测试简单的放大操作是否执行
def test_simple_upscaling_performed(base_url, img2img_basic_image_base64):
    # 构建请求参数 payload
    payload = {
        "resize_mode": 0,
        "show_extras_results": True,
        "gfpgan_visibility": 0,
        "codeformer_visibility": 0,
        "codeformer_weight": 0,
        "upscaling_resize": 2,
        "upscaling_resize_w": 128,
        "upscaling_resize_h": 128,
        "upscaling_crop": True,
        "upscaler_1": "Lanczos",
        "upscaler_2": "None",
        "extras_upscaler_2_visibility": 0,
        "image": img2img_basic_image_base64,
    }
    # 发送 POST 请求并断言状态码为 200
    assert requests.post(f"{base_url}/sdapi/v1/extra-single-image", json=payload).status_code == 200

# 测试获取 PNG 图片信息是否执行
def test_png_info_performed(base_url, img2img_basic_image_base64):
    # 构建请求参数 payload
    payload = {
        "image": img2img_basic_image_base64,
    }
    # 发送 POST 请求并断言状态码为 200
    assert requests.post(f"{base_url}/sdapi/v1/extra-single-image", json=payload).status_code == 200

# 测试获取图片信息是否执行
def test_interrogate_performed(base_url, img2img_basic_image_base64):
    # 构建请求参数 payload
    payload = {
        "image": img2img_basic_image_base64,
        "model": "clip",
    }
    # 发送 POST 请求并断言状态码为 200
    assert requests.post(f"{base_url}/sdapi/v1/extra-single-image", json=payload).status_code == 200
```