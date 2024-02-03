# `stable-diffusion-webui\test\test_img2img.py`

```py
# 导入 pytest 和 requests 模块
import pytest
import requests

# 定义一个 pytest fixture，返回图片转换接口的 URL
@pytest.fixture()
def url_img2img(base_url):
    return f"{base_url}/sdapi/v1/img2img"

# 定义一个 pytest fixture，返回简单的图片转换请求参数
@pytest.fixture()
def simple_img2img_request(img2img_basic_image_base64):
    return {
        "batch_size": 1,
        "cfg_scale": 7,
        "denoising_strength": 0.75,
        "eta": 0,
        "height": 64,
        "include_init_images": False,
        "init_images": [img2img_basic_image_base64],
        "inpaint_full_res": False,
        "inpaint_full_res_padding": 0,
        "inpainting_fill": 0,
        "inpainting_mask_invert": False,
        "mask": None,
        "mask_blur": 4,
        "n_iter": 1,
        "negative_prompt": "",
        "override_settings": {},
        "prompt": "example prompt",
        "resize_mode": 0,
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

# 测试简单的图片转换请求是否成功
def test_img2img_simple_performed(url_img2img, simple_img2img_request):
    assert requests.post(url_img2img, json=simple_img2img_request).status_code == 200

# 测试带遮罩的图片转换请求是否成功
def test_inpainting_masked_performed(url_img2img, simple_img2img_request, mask_basic_image_base64):
    simple_img2img_request["mask"] = mask_basic_image_base64
    assert requests.post(url_img2img, json=simple_img2img_request).status_code == 200

# 测试带反转遮罩的图片转换请求是否成功
def test_inpainting_with_inverted_masked_performed(url_img2img, simple_img2img_request, mask_basic_image_base64):
    simple_img2img_request["mask"] = mask_basic_image_base64
    simple_img2img_request["inpainting_mask_invert"] = True
    assert requests.post(url_img2img, json=simple_img2img_request).status_code == 200

# 测试图片上采样请求是否成功
def test_img2img_sd_upscale_performed(url_img2img, simple_img2img_request):
    # 设置简单的图像到图像请求的脚本名称为 "sd upscale"
    simple_img2img_request["script_name"] = "sd upscale"
    # 设置简单的图像到图像请求的脚本参数为 ["", 8, "Lanczos", 2.0]
    simple_img2img_request["script_args"] = ["", 8, "Lanczos", 2.0]
    # 发送 POST 请求到指定的 URL，并确保返回状态码为 200
    assert requests.post(url_img2img, json=simple_img2img_request).status_code == 200
```