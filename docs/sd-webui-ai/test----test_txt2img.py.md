# `stable-diffusion-webui\test\test_txt2img.py`

```
# 导入 pytest 和 requests 模块
import pytest
import requests

# 定义一个 fixture，返回 txt2img 的 URL
@pytest.fixture()
def url_txt2img(base_url):
    return f"{base_url}/sdapi/v1/txt2img"

# 定义一个 fixture，返回简单的 txt2img 请求参数
@pytest.fixture()
def simple_txt2img_request():
    return {
        "batch_size": 1,
        "cfg_scale": 7,
        "denoising_strength": 0,
        "enable_hr": False,
        "eta": 0,
        "firstphase_height": 0,
        "firstphase_width": 0,
        "height": 64,
        "n_iter": 1,
        "negative_prompt": "",
        "prompt": "example prompt",
        "restore_faces": False,
        "s_churn": 0,
        "s_noise": 1,
        "s_tmax": 0,
        "s_tmin": 0,
        "sampler_index": "Euler a",
        "seed": -1,
        "seed_resize_from_h": -1,
        "seed_resize_from_w": -1,
        "steps": 3,
        "styles": [],
        "subseed": -1,
        "subseed_strength": 0,
        "tiling": False,
        "width": 64,
    }

# 测试简单的 txt2img 请求是否成功
def test_txt2img_simple_performed(url_txt2img, simple_txt2img_request):
    assert requests.post(url_txt2img, json=simple_txt2img_request).status_code == 200

# 测试带有负面提示的 txt2img 请求是否成功
def test_txt2img_with_negative_prompt_performed(url_txt2img, simple_txt2img_request):
    simple_txt2img_request["negative_prompt"] = "example negative prompt"
    assert requests.post(url_txt2img, json=simple_txt2img_request).status_code == 200

# 测试带有复杂提示的 txt2img 请求是否成功
def test_txt2img_with_complex_prompt_performed(url_txt2img, simple_txt2img_request):
    simple_txt2img_request["prompt"] = "((emphasis)), (emphasis1:1.1), [to:1], [from::2], [from:to:0.3], [alt|alt1]"
    assert requests.post(url_txt2img, json=simple_txt2img_request).status_code == 200

# 测试非正方形图像的 txt2img 请求是否成功
def test_txt2img_not_square_image_performed(url_txt2img, simple_txt2img_request):
    simple_txt2img_request["height"] = 128
    assert requests.post(url_txt2img, json=simple_txt2img_request).status_code == 200

# 测试带有 hrfix 的 txt2img 请求是否成功
def test_txt2img_with_hrfix_performed(url_txt2img, simple_txt2img_request):
    simple_txt2img_request["enable_hr"] = True
    assert requests.post(url_txt2img, json=simple_txt2img_request).status_code == 200
# 测试文本转图像功能，开启瓦片化处理
def test_txt2img_with_tiling_performed(url_txt2img, simple_txt2img_request):
    simple_txt2img_request["tiling"] = True
    # 发送 POST 请求，验证状态码是否为 200
    assert requests.post(url_txt2img, json=simple_txt2img_request).status_code == 200


# 测试文本转图像功能，开启人脸还原处理
def test_txt2img_with_restore_faces_performed(url_txt2img, simple_txt2img_request):
    simple_txt2img_request["restore_faces"] = True
    # 发送 POST 请求，验证状态码是否为 200
    assert requests.post(url_txt2img, json=simple_txt2img_request).status_code == 200


# 使用参数化测试，测试文本转图像功能，使用不同的采样器
@pytest.mark.parametrize("sampler", ["PLMS", "DDIM", "UniPC"])
def test_txt2img_with_vanilla_sampler_performed(url_txt2img, simple_txt2img_request, sampler):
    simple_txt2img_request["sampler_index"] = sampler
    # 发送 POST 请求，验证状态码是否为 200
    assert requests.post(url_txt2img, json=simple_txt2img_request).status_code == 200


# 测试文本转图像功能，处理多批次数据
def test_txt2img_multiple_batches_performed(url_txt2img, simple_txt2img_request):
    simple_txt2img_request["n_iter"] = 2
    # 发送 POST 请求，验证状态码是否为 200
    assert requests.post(url_txt2img, json=simple_txt2img_request).status_code == 200


# 测试文本转图像功能，处理单批次数据
def test_txt2img_batch_performed(url_txt2img, simple_txt2img_request):
    simple_txt2img_request["batch_size"] = 2
    # 发送 POST 请求，验证状态码是否为 200
    assert requests.post(url_txt2img, json=simple_txt2img_request).status_code == 200
```