# `MetaGPT\metagpt\tools\sd_engine.py`

```

# -*- coding: utf-8 -*-
# @Date    : 2023/7/19 16:28
# @Author  : stellahong (stellahong@deepwisdom.ai)
# @Desc    :

# 引入所需的库和模块
import asyncio
import base64
import io
import json
from os.path import join
from typing import List

from aiohttp import ClientSession
from PIL import Image, PngImagePlugin

from metagpt.config import CONFIG
from metagpt.const import SD_OUTPUT_FILE_REPO
from metagpt.logs import logger

# 定义默认的 payload
payload = {
    # 提示内容
    "prompt": "",
    # 负面提示
    "negative_prompt": "(easynegative:0.8),black, dark,Low resolution",
    # 覆盖设置
    "override_settings": {"sd_model_checkpoint": "galaxytimemachinesGTM_photoV20"},
    # 种子
    "seed": -1,
    # 批处理大小
    "batch_size": 1,
    # 迭代次数
    "n_iter": 1,
    # 步数
    "steps": 20,
    # 配置规模
    "cfg_scale": 7,
    # 宽度
    "width": 512,
    # 高度
    "height": 768,
    # 恢复面孔
    "restore_faces": False,
    # 平铺
    "tiling": False,
    # 不保存样本
    "do_not_save_samples": False,
    # 不保存网格
    "do_not_save_grid": False,
    # 启用高分辨率
    "enable_hr": False,
    # 高分辨率规模
    "hr_scale": 2,
    # 高分辨率放大器
    "hr_upscaler": "Latent",
    # 高分辨率第二次通过步数
    "hr_second_pass_steps": 0,
    # 高分辨率调整 x
    "hr_resize_x": 0,
    # 高分辨率调整 y
    "hr_resize_y": 0,
    # 高分辨率放大到 x
    "hr_upscale_to_x": 0,
    # 高分辨率放大到 y
    "hr_upscale_to_y": 0,
    # 截断 x
    "truncate_x": 0,
    # 截断 y
    "truncate_y": 0,
    # 应用旧的高分辨率行为
    "applied_old_hires_behavior_to": None,
    # ETA
    "eta": None,
    # 采样器索引
    "sampler_index": "DPM++ SDE Karras",
    # 始终开启脚本
    "alwayson_scripts": {},
}

# 默认的负面提示
default_negative_prompt = "(easynegative:0.8),black, dark,Low resolution"

# 定义 SDEngine 类
class SDEngine:
    def __init__(self):
        # 初始化 SDEngine 并配置
        self.sd_url = CONFIG.get("SD_URL")
        self.sd_t2i_url = f"{self.sd_url}{CONFIG.get('SD_T2I_API')}"
        # 定义 SD API 的默认 payload 设置
        self.payload = payload
        logger.info(self.sd_t2i_url)

    def construct_payload(
        self,
        prompt,
        negtive_prompt=default_negative_prompt,
        width=512,
        height=512,
        sd_model="galaxytimemachinesGTM_photoV20",
    ):
        # 使用提供的输入配置 payload
        self.payload["prompt"] = prompt
        self.payload["negtive_prompt"] = negtive_prompt
        self.payload["width"] = width
        self.payload["height"] = height
        self.payload["override_settings"]["sd_model_checkpoint"] = sd_model
        logger.info(f"call sd payload is {self.payload}")
        return self.payload

    def _save(self, imgs, save_name=""):
        # 保存图片到指定目录
        save_dir = CONFIG.workspace_path / SD_OUTPUT_FILE_REPO
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        batch_decode_base64_to_image(imgs, str(save_dir), save_name=save_name)

    async def run_t2i(self, prompts: List):
        # 异步运行 SD API 处理多个提示
        session = ClientSession()
        for payload_idx, payload in enumerate(prompts):
            results = await self.run(url=self.sd_t2i_url, payload=payload, session=session)
            self._save(results, save_name=f"output_{payload_idx}")
        await session.close()

    async def run(self, url, payload, session):
        # 执行 HTTP POST 请求到 SD API
        async with session.post(url, json=payload, timeout=600) as rsp:
            data = await rsp.read()

        rsp_json = json.loads(data)
        imgs = rsp_json["images"]
        logger.info(f"callback rsp json is {rsp_json.keys()}")
        return imgs

    async def run_i2i(self):
        # todo: 添加图生图接口调用
        raise NotImplementedError

    async def run_sam(self):
        # todo：添加SAM接口调用
        raise NotImplementedError


def decode_base64_to_image(img, save_name):
    # 将 base64 编码的图片解码为图片并保存
    image = Image.open(io.BytesIO(base64.b64decode(img.split(",", 1)[0])))
    pnginfo = PngImagePlugin.PngInfo()
    logger.info(save_name)
    image.save(f"{save_name}.png", pnginfo=pnginfo)
    return pnginfo, image


def batch_decode_base64_to_image(imgs, save_dir="", save_name=""):
    # 批量将 base64 编码的图片解码为图片并保存
    for idx, _img in enumerate(imgs):
        save_name = join(save_dir, save_name)
        decode_base64_to_image(_img, save_name=save_name)


if __name__ == "__main__":
    # 实例化 SDEngine
    engine = SDEngine()
    # 设置提示内容
    prompt = "pixel style, game design, a game interface should be minimalistic and intuitive with the score and high score displayed at the top. The snake and its food should be easily distinguishable. The game should have a simple color scheme, with a contrasting color for the snake and its food. Complete interface boundary"

    # 构建 payload
    engine.construct_payload(prompt)

    # 获取事件循环并运行异步任务
    event_loop = asyncio.get_event_loop()
    event_loop.run_until_complete(engine.run_t2i(prompt))

```