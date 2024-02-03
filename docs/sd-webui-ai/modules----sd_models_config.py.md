# `stable-diffusion-webui\modules\sd_models_config.py`

```py
# 导入 os 模块
import os
# 导入 torch 模块
import torch
# 从 modules 模块中导入 shared、paths、sd_disable_initialization、devices
from modules import shared, paths, sd_disable_initialization, devices

# 设置 sd_configs_path 变量为 shared 模块中的 sd_configs_path
sd_configs_path = shared.sd_configs_path
# 设置 sd_repo_configs_path 变量为 Stable Diffusion 路径下的 configs/stable-diffusion
sd_repo_configs_path = os.path.join(paths.paths['Stable Diffusion'], "configs", "stable-diffusion")
# 设置 sd_xl_repo_configs_path 变量为 Stable Diffusion XL 路径下的 configs/inference
sd_xl_repo_configs_path = os.path.join(paths.paths['Stable Diffusion XL'], "configs", "inference")

# 设置 config_default 变量为 shared 模块中的 sd_default_config
config_default = shared.sd_default_config
# 设置 config_sd2 变量为 sd_repo_configs_path 路径下的 v2-inference.yaml
config_sd2 = os.path.join(sd_repo_configs_path, "v2-inference.yaml")
# 设置 config_sd2v 变量为 sd_repo_configs_path 路径下的 v2-inference-v.yaml
config_sd2v = os.path.join(sd_repo_configs_path, "v2-inference-v.yaml")
# 设置 config_sd2_inpainting 变量为 sd_repo_configs_path 路径下的 v2-inpainting-inference.yaml
config_sd2_inpainting = os.path.join(sd_repo_configs_path, "v2-inpainting-inference.yaml")
# 设置 config_sdxl 变量为 sd_xl_repo_configs_path 路径下的 sd_xl_base.yaml
config_sdxl = os.path.join(sd_xl_repo_configs_path, "sd_xl_base.yaml")
# 设置 config_sdxl_refiner 变量为 sd_xl_repo_configs_path 路径下的 sd_xl_refiner.yaml
config_sdxl_refiner = os.path.join(sd_xl_repo_configs_path, "sd_xl_refiner.yaml")
# 设置 config_depth_model 变量为 sd_repo_configs_path 路径下的 v2-midas-inference.yaml
config_depth_model = os.path.join(sd_repo_configs_path, "v2-midas-inference.yaml")
# 设置 config_unclip 变量为 sd_repo_configs_path 路径下的 v2-1-stable-unclip-l-inference.yaml
config_unclip = os.path.join(sd_repo_configs_path, "v2-1-stable-unclip-l-inference.yaml")
# 设置 config_unopenclip 变量为 sd_repo_configs_path 路径下的 v2-1-stable-unclip-h-inference.yaml
config_unopenclip = os.path.join(sd_repo_configs_path, "v2-1-stable-unclip-h-inference.yaml")
# 设置 config_inpainting 变量为 sd_configs_path 路径下的 v1-inpainting-inference.yaml
config_inpainting = os.path.join(sd_configs_path, "v1-inpainting-inference.yaml")
# 设置 config_instruct_pix2pix 变量为 sd_configs_path 路径下的 instruct-pix2pix.yaml
config_instruct_pix2pix = os.path.join(sd_configs_path, "instruct-pix2pix.yaml")
# 设置 config_alt_diffusion 变量为 sd_configs_path 路径下的 alt-diffusion-inference.yaml
config_alt_diffusion = os.path.join(sd_configs_path, "alt-diffusion-inference.yaml")
# 设置 config_alt_diffusion_m18 变量为 sd_configs_path 路径下的 alt-diffusion-m18-inference.yaml
config_alt_diffusion_m18 = os.path.join(sd_configs_path, "alt-diffusion-m18-inference.yaml")

# 定义函数 is_using_v_parameterization_for_sd2，用于检测 state_dict 中的 unet 是否使用 v-parameterization
def is_using_v_parameterization_for_sd2(state_dict):
    """
    Detects whether unet in state_dict is using v-parameterization. Returns True if it is. You're welcome.
    """
    # 导入 ldm.modules.diffusionmodules.openaimodel 模块
    import ldm.modules.diffusionmodules.openaimodel
    # 设置 device 变量为 cpu 设备
    device = devices.cpu
    # 使用 sd_disable_initialization.DisableInitialization() 上下文管理器，禁用初始化
    with sd_disable_initialization.DisableInitialization():
        # 创建 UNetModel 对象
        unet = ldm.modules.diffusionmodules.openaimodel.UNetModel(
            use_checkpoint=True,  # 使用检查点
            use_fp16=False,  # 不使用 FP16
            image_size=32,  # 图像大小为 32x32
            in_channels=4,  # 输入通道数为 4
            out_channels=4,  # 输出通道数为 4
            model_channels=320,  # 模型通道数为 320
            attention_resolutions=[4, 2, 1],  # 注意力分辨率
            num_res_blocks=2,  # 残差块数量为 2
            channel_mult=[1, 2, 4, 4],  # 通道倍增
            num_head_channels=64,  # 头通道数为 64
            use_spatial_transformer=True,  # 使用空间变换器
            use_linear_in_transformer=True,  # 在变换器中使用线性
            transformer_depth=1,  # 变换器深度为 1
            context_dim=1024,  # 上下文维度为 1024
            legacy=False  # 不使用旧版本
        )
        # 将模型设置为评估模式
        unet.eval()

    # 使用 torch.no_grad() 上下文管理器，关闭梯度计算
    with torch.no_grad():
        # 从 state_dict 中提取 UNetModel 的参数
        unet_sd = {k.replace("model.diffusion_model.", ""): v for k, v in state_dict.items() if "model.diffusion_model." in k}
        # 加载参数到 UNetModel
        unet.load_state_dict(unet_sd, strict=True)
        # 将 UNetModel 移动到指定设备上，并设置数据类型为 torch.float
        unet.to(device=device, dtype=torch.float)

        # 创建测试条件张量
        test_cond = torch.ones((1, 2, 1024), device=device) * 0.5
        # 创建测试输入张量
        x_test = torch.ones((1, 4, 8, 8), device=device) * 0.5

        # 对输入数据进行推理，计算输出与输入的差值的均值
        out = (unet(x_test, torch.asarray([999], device=device), context=test_cond) - x_test).mean().item()

    # 返回是否输出小于 -1 的布尔值
    return out < -1
# 从状态字典中猜测模型配置
def guess_model_config_from_state_dict(sd, filename):
    # 获取特定键对应的值，如果键不存在则返回 None
    sd2_cond_proj_weight = sd.get('cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight', None)
    diffusion_model_input = sd.get('model.diffusion_model.input_blocks.0.0.weight', None)
    sd2_variations_weight = sd.get('embedder.model.ln_final.weight', None)

    # 检查特定键是否存在，如果存在则返回相应的配置
    if sd.get('conditioner.embedders.1.model.ln_final.weight', None) is not None:
        return config_sdxl
    if sd.get('conditioner.embedders.0.model.ln_final.weight', None) is not None:
        return config_sdxl_refiner
    elif sd.get('depth_model.model.pretrained.act_postprocess3.0.project.0.bias', None) is not None:
        return config_depth_model
    elif sd2_variations_weight is not None and sd2_variations_weight.shape[0] == 768:
        return config_unclip
    elif sd2_variations_weight is not None and sd2_variations_weight.shape[0] == 1024:
        return config_unopenclip

    # 检查特定键对应的值是否符合条件，返回相应的配置
    if sd2_cond_proj_weight is not None and sd2_cond_proj_weight.shape[1] == 1024:
        if diffusion_model_input.shape[1] == 9:
            return config_sd2_inpainting
        elif is_using_v_parameterization_for_sd2(sd):
            return config_sd2v
        else:
            return config_sd2

    # 检查特定键对应的值是否存在，返回相应的配置
    if diffusion_model_input is not None:
        if diffusion_model_input.shape[1] == 9:
            return config_inpainting
        if diffusion_model_input.shape[1] == 8:
            return config_instruct_pix2pix

    # 检查特定键对应的值是否存在，返回相应的配置
    if sd.get('cond_stage_model.roberta.embeddings.word_embeddings.weight', None) is not None:
        if sd.get('cond_stage_model.transformation.weight').size()[0] == 1024:
            return config_alt_diffusion_m18
        return config_alt_diffusion

    # 默认返回配置
    return config_default


# 查找检查点配置
def find_checkpoint_config(state_dict, info):
    # 如果信息为空，则根据状态字典猜测模型配置
    if info is None:
        return guess_model_config_from_state_dict(state_dict, "")

    # 在文件名附近查找检查点配置
    config = find_checkpoint_config_near_filename(info)
    if config is not None:
        return config
    # 从给定的状态字典和文件名猜测模型配置，并返回结果
    return guess_model_config_from_state_dict(state_dict, info.filename)
# 根据文件名查找附近的检查点配置文件
def find_checkpoint_config_near_filename(info):
    # 如果信息为空，则返回空
    if info is None:
        return None

    # 根据文件名获取对应的配置文件名
    config = f"{os.path.splitext(info.filename)[0]}.yaml"
    # 如果配置文件存在，则返回配置文件名
    if os.path.exists(config):
        return config

    # 如果配置文件不存在，则返回空
    return None
```