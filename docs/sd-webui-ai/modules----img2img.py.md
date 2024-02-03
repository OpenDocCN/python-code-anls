# `stable-diffusion-webui\modules\img2img.py`

```
# 导入必要的库
import os
from contextlib import closing
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, UnidentifiedImageError
import gradio as gr
# 导入自定义模块
from modules import images as imgutil
from modules.generation_parameters_copypaste import create_override_settings_dict, parse_generation_parameters
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, state
from modules.sd_models import get_closet_checkpoint_match
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html
import modules.scripts

# 定义处理批量图像的函数
def process_batch(p, input_dir, output_dir, inpaint_mask_dir, args, to_scale=False, scale_by=1.0, use_png_info=False, png_info_props=None, png_info_dir=None):
    # 去除输出目录两端的空格
    output_dir = output_dir.strip()
    # 修复随机种子
    processing.fix_seed(p)
    
    # 获取输入目录中所有支持的图像文件
    images = list(shared.walk_files(input_dir, allowed_extensions=(".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")))
    
    # 检查是否为修复图像批处理
    is_inpaint_batch = False
    if inpaint_mask_dir:
        # 获取修复图像的掩模文件
        inpaint_masks = shared.listfiles(inpaint_mask_dir)
        is_inpaint_batch = bool(inpaint_masks)
        
        if is_inpaint_batch:
            print(f"\nInpaint batch is enabled. {len(inpaint_masks)} masks found.")
    
    # 打印将处理的图像数量
    print(f"Will process {len(images)} images, creating {p.n_iter * p.batch_size} new images for each.")
    
    # 设置作业数量
    state.job_count = len(images) * p.n_iter
    
    # 提取“默认”参数以在获取 PNG 信息失败时使用
    prompt = p.prompt
    negative_prompt = p.negative_prompt
    seed = p.seed
    cfg_scale = p.cfg_scale
    sampler_name = p.sampler_name
    steps = p.steps
    override_settings = p.override_settings
    sd_model_checkpoint_override = get_closet_checkpoint_match(override_settings.get("sd_model_checkpoint", None))
    batch_results = None
    discard_further_results = False
    return batch_results
# 定义一个函数，用于将图像转换为图像，根据给定的参数进行处理
def img2img(id_task: str, mode: int, prompt: str, negative_prompt: str, prompt_styles, init_img, sketch, init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig, init_img_inpaint, init_mask_inpaint, steps: int, sampler_name: str, mask_blur: int, mask_alpha: float, inpainting_fill: int, n_iter: int, batch_size: int, cfg_scale: float, image_cfg_scale: float, denoising_strength: float, selected_scale_tab: int, height: int, width: int, scale_by: float, resize_mode: int, inpaint_full_res: bool, inpaint_full_res_padding: int, inpainting_mask_invert: int, img2img_batch_input_dir: str, img2img_batch_output_dir: str, img2img_batch_inpaint_mask_dir: str, override_settings_texts, img2img_batch_use_png_info: bool, img2img_batch_png_info_props: list, img2img_batch_png_info_dir: str, request: gr.Request, *args):
    # 根据传入的覆盖设置文本创建覆盖设置字典
    override_settings = create_override_settings_dict(override_settings_texts)

    # 判断处理模式是否为批处理模式
    is_batch = mode == 5

    # 根据不同的处理模式进行处理
    if mode == 0:  # img2img
        # 使用初始图像作为处理的图像，不使用遮罩
        image = init_img
        mask = None
    elif mode == 1:  # img2img sketch
        # 使用草图作为处理的图像，不使用遮罩
        image = sketch
        mask = None
    elif mode == 2:  # inpaint
        # 使用带有遮罩的初始图像进行修复，创建二进制遮罩
        image, mask = init_img_with_mask["image"], init_img_with_mask["mask"]
        mask = processing.create_binary_mask(mask)
    elif mode == 3:  # inpaint sketch
        # 使用彩色草图进行修复，根据原始图像和预测图像创建遮罩
        image = inpaint_color_sketch
        orig = inpaint_color_sketch_orig or inpaint_color_sketch
        pred = np.any(np.array(image) != np.array(orig), axis=-1)
        mask = Image.fromarray(pred.astype(np.uint8) * 255, "L")
        mask = ImageEnhance.Brightness(mask).enhance(1 - mask_alpha / 100)
        blur = ImageFilter.GaussianBlur(mask_blur)
        image = Image.composite(image.filter(blur), orig, mask.filter(blur))
    elif mode == 4:  # inpaint upload mask
        # 使用上传的图像和遮罩进行修复
        image = init_img_inpaint
        mask = init_mask_inpaint
    else:
        # 处理模式不匹配时，图像和遮罩均为空
        image = None
        mask = None

    # 使用智能手机拍摄的照片的 EXIF 方向信息
    # 如果图像不为空，则对图像进行EXIF旋转
    if image is not None:
        image = ImageOps.exif_transpose(image)

    # 如果选择的缩放选项为1且不是批处理，则执行以下操作
    if selected_scale_tab == 1 and not is_batch:
        # 断言图像不为空，否则抛出异常
        assert image, "Can't scale by because no image is selected"

        # 计算缩放后的宽度和高度
        width = int(image.width * scale_by)
        height = int(image.height * scale_by)

    # 断言去噪强度在[0.0, 1.0]范围内
    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

    # 创建StableDiffusionProcessingImg2Img对象，并传入相关参数
    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=prompt_styles,
        sampler_name=sampler_name,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        init_images=[image],
        mask=mask,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        image_cfg_scale=image_cfg_scale,
        inpaint_full_res=inpaint_full_res,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=inpainting_mask_invert,
        override_settings=override_settings,
    )

    # 设置脚本模块和参数
    p.scripts = modules.scripts.scripts_img2img
    p.script_args = args

    # 设置用户
    p.user = request.username

    # 如果启用控制台提示，则打印提示信息
    if shared.opts.enable_console_prompts:
        print(f"\nimg2img: {prompt}", file=shared.progress_print_out)

    # 如果存在mask，则将mask模糊度参数添加到额外生成参数中
    if mask:
        p.extra_generation_params["Mask blur"] = mask_blur
    # 使用 closing 上下文管理器确保资源被正确关闭
    with closing(p):
        # 如果是批处理模式
        if is_batch:
            # 断言不隐藏 UI 目录配置，否则禁用批量图像转换
            assert not shared.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"
            # 处理批量图像转换，返回处理后的结果
            processed = process_batch(p, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir, args, to_scale=selected_scale_tab == 1, scale_by=scale_by, use_png_info=img2img_batch_use_png_info, png_info_props=img2img_batch_png_info_props, png_info_dir=img2img_batch_png_info_dir)

            # 如果处理结果为空
            if processed is None:
                # 创建一个空的 Processed 对象
                processed = Processed(p, [], p.seed, "")
        else:
            # 运行单个图像转换脚本，返回处理后的结果
            processed = modules.scripts.scripts_img2img.run(p, *args)
            # 如果处理结果为空
            if processed is None:
                # 处理图像
                processed = process_images(p)

    # 清空共享的 total_tqdm
    shared.total_tqdm.clear()

    # 生成处理结果的 JavaScript 信息
    generation_info_js = processed.js()
    # 如果设置了将样本日志输出到标准输出
    if opts.samples_log_stdout:
        # 打印生成信息的 JavaScript
        print(generation_info_js)

    # 如果设置了不显示图像
    if opts.do_not_show_images:
        # 将处理结果的图像列表置空
        processed.images = []

    # 返回处理后的图像列表、生成信息的 JavaScript、处理信息的 HTML 格式、评论的 HTML 格式
    return processed.images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")
```