# `stable-diffusion-webui\modules\extras.py`

```
# 导入必要的库
import os
import re
import shutil
import json

# 导入 torch 库
import torch
# 导入 tqdm 库
import tqdm

# 导入自定义模块
from modules import shared, images, sd_models, sd_vae, sd_models_config, errors
# 导入自定义模块中的函数
from modules.ui_common import plaintext_to_html
# 导入 gradio 库
import gradio as gr
# 导入 safetensors.torch 库
import safetensors.torch

# 定义函数 run_pnginfo，用于获取图片信息
def run_pnginfo(image):
    # 如果图片为空，则返回空字符串
    if image is None:
        return '', '', ''

    # 从图片中读取信息
    geninfo, items = images.read_info_from_image(image)
    # 将通用信息和具体信息合并
    items = {**{'parameters': geninfo}, **items}

    # 初始化信息字符串
    info = ''
    # 遍历信息字典，生成 HTML 格式的信息
    for key, text in items.items():
        info += f"""
<div>
<p><b>{plaintext_to_html(str(key))}</b></p>
<p>{plaintext_to_html(str(text))}</p>
</div>
""".strip()+"\n"

    # 如果信息为空，则返回提示信息
    if len(info) == 0:
        message = "Nothing found in the image."
        info = f"<div><p>{message}<p></div>"

    # 返回结果
    return '', geninfo, info

# 创建配置文件
def create_config(ckpt_result, config_source, a, b, c):
    # 定义内部函数 config，用于查找最接近文件名的配置文件
    def config(x):
        res = sd_models_config.find_checkpoint_config_near_filename(x) if x else None
        return res if res != shared.sd_default_config else None

    # 根据配置来源选择配置文件
    if config_source == 0:
        cfg = config(a) or config(b) or config(c)
    elif config_source == 1:
        cfg = config(b)
    elif config_source == 2:
        cfg = config(c)
    else:
        cfg = None

    # 如果配置文件为空，则返回
    if cfg is None:
        return

    # 构建检查点文件名
    filename, _ = os.path.splitext(ckpt_result)
    checkpoint_filename = filename + ".yaml"

    # 复制配置文件
    print("Copying config:")
    print("   from:", cfg)
    print("     to:", checkpoint_filename)
    shutil.copyfile(cfg, checkpoint_filename)

# 定义函数 to_half，用于将张量转换为半精度
def to_half(tensor, enable):
    # 如果启用半精度且张量为浮点型，则转换为半精度
    if enable and tensor.dtype == torch.float:
        return tensor.half()

    return tensor

# 读取元数据
def read_metadata(primary_model_name, secondary_model_name, tertiary_model_name):
    metadata = {}
    # 遍历给定的模型名称列表
    for checkpoint_name in [primary_model_name, secondary_model_name, tertiary_model_name]:
        # 获取指定模型名称对应的检查点信息，如果不存在则为 None
        checkpoint_info = sd_models.checkpoints_list.get(checkpoint_name, None)
        # 如果检查点信息为 None，则跳过当前循环
        if checkpoint_info is None:
            continue

        # 更新 metadata 字典，将当前检查点信息中的 metadata 合并到 metadata 中
        metadata.update(checkpoint_info.metadata)

    # 将 metadata 字典转换为 JSON 格式的字符串，缩进为 4 个空格，确保不转义非 ASCII 字符
    return json.dumps(metadata, indent=4, ensure_ascii=False)
# 运行模型合并任务，接受多个参数，返回合并后的模型
def run_modelmerger(id_task, primary_model_name, secondary_model_name, tertiary_model_name, interp_method, multiplier, save_as_half, custom_name, checkpoint_format, config_source, bake_in_vae, discard_weights, save_metadata, add_merge_recipe, copy_metadata_fields, metadata_json):
    # 开始模型合并任务
    shared.state.begin(job="model-merge")

    # 定义失败时的处理函数
    def fail(message):
        shared.state.textinfo = message
        shared.state.end()
        return [*[gr.update() for _ in range(4)], message]

    # 计算加权和
    def weighted_sum(theta0, theta1, alpha):
        return ((1 - alpha) * theta0) + (alpha * theta1)

    # 计算两个参数的差值
    def get_difference(theta1, theta2):
        return theta1 - theta2

    # 计算加上差值后的值
    def add_difference(theta0, theta1_2_diff, alpha):
        return theta0 + (alpha * theta1_2_diff)

    # 生成加权和的文件名
    def filename_weighted_sum():
        a = primary_model_info.model_name
        b = secondary_model_info.model_name
        Ma = round(1 - multiplier, 2)
        Mb = round(multiplier, 2)

        return f"{Ma}({a}) + {Mb}({b})"

    # 生成加上差值的文件名
    def filename_add_difference():
        a = primary_model_info.model_name
        b = secondary_model_info.model_name
        c = tertiary_model_info.model_name
        M = round(multiplier, 2)

        return f"{a} + {M}({b} - {c})"

    # 生成不进行插值的文件名
    def filename_nothing():
        return primary_model_info.model_name

    # 定义插值方法和对应的函数
    theta_funcs = {
        "Weighted sum": (filename_weighted_sum, None, weighted_sum),
        "Add difference": (filename_add_difference, get_difference, add_difference),
        "No interpolation": (filename_nothing, None, None),
    }
    # 根据插值方法选择对应的文件名生成函数和计算函数
    filename_generator, theta_func1, theta_func2 = theta_funcs[interp_method]
    # 设置任务数量
    shared.state.job_count = (1 if theta_func1 else 0) + (1 if theta_func2 else 0)

    # 如果没有指定主模型，则返回失败信息
    if not primary_model_name:
        return fail("Failed: Merging requires a primary model.")

    # 获取主模型信息
    primary_model_info = sd_models.checkpoints_list[primary_model_name]

    # 如果需要第二个参数但未指定，则返回失败信息
    if theta_func2 and not secondary_model_name:
        return fail("Failed: Merging requires a secondary model.")
    # 如果存在第二个模型的名称和对应的函数，则获取第二个模型的信息，否则设为 None
    secondary_model_info = sd_models.checkpoints_list[secondary_model_name] if theta_func2 else None

    # 如果存在第一个函数和不存在第三个模型的名称，则返回失败信息
    if theta_func1 and not tertiary_model_name:
        return fail(f"Failed: Interpolation method ({interp_method}) requires a tertiary model.")

    # 如果存在第一个函数，则获取第三个模型的信息，否则设为 None
    tertiary_model_info = sd_models.checkpoints_list[tertiary_model_name] if theta_func1 else None

    # 初始化标志变量
    result_is_inpainting_model = False
    result_is_instruct_pix2pix_model = False

    # 如果存在第二个函数，则加载第二个模型的参数
    if theta_func2:
        shared.state.textinfo = "Loading B"
        print(f"Loading {secondary_model_info.filename}...")
        theta_1 = sd_models.read_state_dict(secondary_model_info.filename, map_location='cpu')
    else:
        theta_1 = None

    # 如果存在第一个函数，则加载第三个模型的参数，并合并第二个和第三个模型的参数
    if theta_func1:
        shared.state.textinfo = "Loading C"
        print(f"Loading {tertiary_model_info.filename}...")
        theta_2 = sd_models.read_state_dict(tertiary_model_info.filename, map_location='cpu')

        shared.state.textinfo = 'Merging B and C'
        shared.state.sampling_steps = len(theta_1.keys())
        for key in tqdm.tqdm(theta_1.keys()):
            if key in checkpoint_dict_skip_on_merge:
                continue

            if 'model' in key:
                if key in theta_2:
                    t2 = theta_2.get(key, torch.zeros_like(theta_1[key]))
                    theta_1[key] = theta_func1(theta_1[key], t2)
                else:
                    theta_1[key] = torch.zeros_like(theta_1[key])

            shared.state.sampling_step += 1
        del theta_2

        shared.state.nextjob()

    # 加载主模型的参数
    shared.state.textinfo = f"Loading {primary_model_info.filename}..."
    print(f"Loading {primary_model_info.filename}...")
    theta_0 = sd_models.read_state_dict(primary_model_info.filename, map_location='cpu')

    # 输出信息并设置状态
    print("Merging...")
    shared.state.textinfo = 'Merging A and B'
    shared.state.sampling_steps = len(theta_0.keys())
    # 遍历 theta_0 字典的键
    for key in tqdm.tqdm(theta_0.keys()):
        # 检查是否存在 theta_1 字典且 key 中包含 'model'，同时 key 存在于 theta_1 中
        if theta_1 and 'model' in key and key in theta_1:

            # 如果 key 在 checkpoint_dict_skip_on_merge 中，则跳过当前循环
            if key in checkpoint_dict_skip_on_merge:
                continue

            # 获取 theta_0 和 theta_1 中对应 key 的值
            a = theta_0[key]
            b = theta_1[key]

            # 根据条件判断是否可以合并两个模型
            if a.shape != b.shape and a.shape[0:1] + a.shape[2:] == b.shape[0:1] + b.shape[2:]:
                # 如果 a 和 b 的通道数分别为 4 和 9，则抛出异常
                if a.shape[1] == 4 and b.shape[1] == 9:
                    raise RuntimeError("When merging inpainting model with a normal one, A must be the inpainting model.")
                # 如果 a 和 b 的通道数分别为 4 和 8，则抛出异常
                if a.shape[1] == 4 and b.shape[1] == 8:
                    raise RuntimeError("When merging instruct-pix2pix model with a normal one, A must be the instruct-pix2pix model.")

                # 如果 a 和 b 的通道数分别为 8 和 4，则执行以下操作
                if a.shape[1] == 8 and b.shape[1] == 4:
                    # 合并 Instruct-Pix2Pix 模型的向量
                    theta_0[key][:, 0:4, :, :] = theta_func2(a[:, 0:4, :, :], b, multiplier)
                    result_is_instruct_pix2pix_model = True
                else:
                    # 如果 a 和 b 的通道数分别为 9 和 4，则执行以下操作
                    assert a.shape[1] == 9 and b.shape[1] == 4, f"Bad dimensions for merged layer {key}: A={a.shape}, B={b.shape}"
                    # 合并 Inpainting 模型的向量
                    theta_0[key][:, 0:4, :, :] = theta_func2(a[:, 0:4, :, :], b, multiplier)
                    result_is_inpainting_model = True
            else:
                # 合并 a 和 b 的值
                theta_0[key] = theta_func2(a, b, multiplier)

            # 将合并后的值转换为半精度
            theta_0[key] = to_half(theta_0[key], save_as_half)

        # 更新采样步数
        shared.state.sampling_step += 1

    # 删除 theta_1 字典
    del theta_1

    # 获取 bake_in_vae 对应的文件名
    bake_in_vae_filename = sd_vae.vae_dict.get(bake_in_vae, None)
    # 如果指定了要嵌入的 VAE 文件名，则加载 VAE 字典并将其嵌入到模型中
    if bake_in_vae_filename is not None:
        # 打印信息，指示正在嵌入 VAE
        print(f"Baking in VAE from {bake_in_vae_filename}")
        # 设置状态信息为'Baking in VAE'
        shared.state.textinfo = 'Baking in VAE'
        # 加载 VAE 字典
        vae_dict = sd_vae.load_vae_dict(bake_in_vae_filename, map_location='cpu')

        # 遍历 VAE 字典的键
        for key in vae_dict.keys():
            # 构建第一阶段模型的键
            theta_0_key = 'first_stage_model.' + key
            # 如果键存在于 theta_0 中，则将 VAE 字典中的值转换为半精度并赋给 theta_0
            if theta_0_key in theta_0:
                theta_0[theta_0_key] = to_half(vae_dict[key], save_as_half)

        # 删除 VAE 字典
        del vae_dict

    # 如果需要将模型参数转换为半精度且没有指定第二阶段模型
    if save_as_half and not theta_func2:
        # 遍历 theta_0 的键
        for key in theta_0.keys():
            # 将 theta_0 中的值转换为半精度
            theta_0[key] = to_half(theta_0[key], save_as_half)

    # 如果需要丢弃权重
    if discard_weights:
        # 编译正则表达式
        regex = re.compile(discard_weights)
        # 遍历 theta_0 的键的副本
        for key in list(theta_0):
            # 如果键匹配正则表达式，则从 theta_0 中删除该键
            if re.search(regex, key):
                theta_0.pop(key, None)

    # 设置检查点目录为命令行参数中的检查点目录或默认模型路径
    ckpt_dir = shared.cmd_opts.ckpt_dir or sd_models.model_path

    # 生成文件名，根据是否为自定义名称和结果类型添加后缀
    filename = filename_generator() if custom_name == '' else custom_name
    filename += ".inpainting" if result_is_inpainting_model else ""
    filename += ".instruct-pix2pix" if result_is_instruct_pix2pix_model else ""
    filename += "." + checkpoint_format

    # 设置输出模型的完整路径
    output_modelname = os.path.join(ckpt_dir, filename)

    # 更新状态信息，指示正在保存模型
    shared.state.nextjob()
    shared.state.textinfo = "Saving"
    # 打印信息，指示正在保存模型到指定路径
    print(f"Saving to {output_modelname}...")

    # 初始化元数据字典
    metadata = {}

    # 如果需要保存元数据并复制元数据字段
    if save_metadata and copy_metadata_fields:
        # 如果存在主要模型信息，则更新元数据字典
        if primary_model_info:
            metadata.update(primary_model_info.metadata)
        # 如果存在次要模型信息，则更新元数据字典
        if secondary_model_info:
            metadata.update(secondary_model_info.metadata)
        # 如果存在第三个模型信息，则更新元数据字典
        if tertiary_model_info:
            metadata.update(tertiary_model_info.metadata)

    # 如果需要保存元数据
    if save_metadata:
        try:
            # 尝试从 JSON 字符串中加载元数据并更新元数据字典
            metadata.update(json.loads(metadata_json))
        except Exception as e:
            # 如果出现异常，则显示错误信息
            errors.display(e, "readin metadata from json")

        # 添加格式信息到元数据字典
        metadata["format"] = "pt"
    # 如果需要保存元数据并添加合并配方
    if save_metadata and add_merge_recipe:
        # 创建合并配方字典
        merge_recipe = {
            "type": "webui", # 指示此模型已与webui的内置合并器合并
            "primary_model_hash": primary_model_info.sha256,
            "secondary_model_hash": secondary_model_info.sha256 if secondary_model_info else None,
            "tertiary_model_hash": tertiary_model_info.sha256 if tertiary_model_info else None,
            "interp_method": interp_method,
            "multiplier": multiplier,
            "save_as_half": save_as_half,
            "custom_name": custom_name,
            "config_source": config_source,
            "bake_in_vae": bake_in_vae,
            "discard_weights": discard_weights,
            "is_inpainting": result_is_inpainting_model,
            "is_instruct_pix2pix": result_is_instruct_pix2pix_model
        }

        # 创建空字典用于存储合并模型
        sd_merge_models = {}

        # 添加模型元数据到合并模型字典中
        def add_model_metadata(checkpoint_info):
            checkpoint_info.calculate_shorthash()
            sd_merge_models[checkpoint_info.sha256] = {
                "name": checkpoint_info.name,
                "legacy_hash": checkpoint_info.hash,
                "sd_merge_recipe": checkpoint_info.metadata.get("sd_merge_recipe", None)
            }

            sd_merge_models.update(checkpoint_info.metadata.get("sd_merge_models", {}))

        # 添加主模型元数据
        add_model_metadata(primary_model_info)
        # 如果有次要模型，添加其元数据
        if secondary_model_info:
            add_model_metadata(secondary_model_info)
        # 如果有第三个模型，添加其元数据
        if tertiary_model_info:
            add_model_metadata(tertiary_model_info)

        # 将合并配方和合并模型字典转换为JSON字符串并存储在元数据中
        metadata["sd_merge_recipe"] = json.dumps(merge_recipe)
        metadata["sd_merge_models"] = json.dumps(sd_merge_models)

    # 获取输出模型文件名的扩展名
    _, extension = os.path.splitext(output_modelname)
    # 如果扩展名为".safetensors"，使用safetensors.torch保存模型文件
    if extension.lower() == ".safetensors":
        safetensors.torch.save_file(theta_0, output_modelname, metadata=metadata if len(metadata)>0 else None)
    # 否则，使用torch保存模型文件
    else:
        torch.save(theta_0, output_modelname)

    # 列出sd_models中的所有模型
    sd_models.list_models()
    # 在 sd_models.checkpoints_list.values() 中查找指定文件名的模型检查点
    created_model = next((ckpt for ckpt in sd_models.checkpoints_list.values() if ckpt.name == filename), None)
    # 如果找到了指定文件名的模型检查点
    if created_model:
        # 计算模型检查点的短哈希值
        created_model.calculate_shorthash()

    # 创建配置文件
    create_config(output_modelname, config_source, primary_model_info, secondary_model_info, tertiary_model_info)

    # 打印提示信息，显示检查点保存的路径
    print(f"Checkpoint saved to {output_modelname}.")
    # 更新共享状态的文本信息为 "Checkpoint saved"
    shared.state.textinfo = "Checkpoint saved"
    # 结束共享状态
    shared.state.end()

    # 返回一个包含下拉菜单更新操作和提示信息的列表
    return [*[gr.Dropdown.update(choices=sd_models.checkpoint_tiles()) for _ in range(4)], "Checkpoint saved to " + output_modelname]
```