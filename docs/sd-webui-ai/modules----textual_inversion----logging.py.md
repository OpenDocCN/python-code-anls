# `stable-diffusion-webui\modules\textual_inversion\logging.py`

```py
# 导入 datetime、json 和 os 模块
import datetime
import json
import os

# 定义保存的共享参数集合
saved_params_shared = {
    "batch_size",
    "clip_grad_mode",
    "clip_grad_value",
    "create_image_every",
    "data_root",
    "gradient_step",
    "initial_step",
    "latent_sampling_method",
    "learn_rate",
    "log_directory",
    "model_hash",
    "model_name",
    "num_of_dataset_images",
    "steps",
    "template_file",
    "training_height",
    "training_width",
}

# 定义保存的 Text-to-Image 参数集合
saved_params_ti = {
    "embedding_name",
    "num_vectors_per_token",
    "save_embedding_every",
    "save_image_with_stored_embedding",
}

# 定义保存的 Hypernetwork 参数集合
saved_params_hypernet = {
    "activation_func",
    "add_layer_norm",
    "hypernetwork_name",
    "layer_structure",
    "save_hypernetwork_every",
    "use_dropout",
    "weight_init",
}

# 合并所有保存的参数集合
saved_params_all = saved_params_shared | saved_params_ti | saved_params_hypernet

# 定义保存的预览参数集合
saved_params_previews = {
    "preview_cfg_scale",
    "preview_height",
    "preview_negative_prompt",
    "preview_prompt",
    "preview_sampler_index",
    "preview_seed",
    "preview_steps",
    "preview_width",
}

# 保存设置到文件
def save_settings_to_file(log_directory, all_params):
    # 获取当前时间
    now = datetime.datetime.now()
    # 创建参数字典，包含当前时间
    params = {"datetime": now.strftime("%Y-%m-%d %H:%M:%S")}

    # 根据是否有预览参数，选择要保存的参数集合
    keys = saved_params_all
    if all_params.get('preview_from_txt2img'):
        keys = keys | saved_params_previews

    # 更新参数字典，只包含指定的参数
    params.update({k: v for k, v in all_params.items() if k in keys})

    # 生成文件名
    filename = f'settings-{now.strftime("%Y-%m-%d-%H-%M-%S")}.json'
    # 打开文件并写入参数字典
    with open(os.path.join(log_directory, filename), "w") as file:
        json.dump(params, file, indent=4)
```