# `stable-diffusion-webui\modules\shared_options.py`

```
# 导入必要的库
import gradio as gr

# 导入自定义模块
from modules import localization, ui_components, shared_items, shared, interrogate, shared_gradio_themes
# 导入内部路径相关的模块
from modules.paths_internal import models_path, script_path, data_path, sd_configs_path, sd_default_config, sd_model_file, default_sd_model_file, extensions_dir, extensions_builtin_dir  # noqa: F401
# 导入命令行选项相关的模块
from modules.shared_cmd_options import cmd_opts
# 导入选项相关的模块
from modules.options import options_section, OptionInfo, OptionHTML, categories

# 初始化选项模板字典
options_templates = {}
# 隐藏的目录列表
hide_dirs = shared.hide_dirs

# 限制的选项集合
restricted_opts = {
    "samples_filename_pattern",
    "directories_filename_pattern",
    "outdir_samples",
    "outdir_txt2img_samples",
    "outdir_img2img_samples",
    "outdir_extras_samples",
    "outdir_grids",
    "outdir_txt2img_grids",
    "outdir_save",
    "outdir_init_images"
}

# 注册不同类别的选项
categories.register_category("saving", "Saving images")
categories.register_category("sd", "Stable Diffusion")
categories.register_category("ui", "User Interface")
categories.register_category("system", "System")
categories.register_category("postprocessing", "Postprocessing")
categories.register_category("training", "Training")

# 更新选项模板字典
options_templates.update(options_section(('saving-images', "Saving images/grids", "saving"), {
    "samples_save": OptionInfo(True, "Always save all generated images"),
    "samples_format": OptionInfo('png', 'File format for images'),
    "samples_filename_pattern": OptionInfo("", "Images filename pattern", component_args=hide_dirs).link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Images-Filename-Name-and-Subdirectory"),
    "save_images_add_number": OptionInfo(True, "Add number to filename when saving", component_args=hide_dirs),
    "save_images_replace_action": OptionInfo("Replace", "Saving the image to an existing file", gr.Radio, {"choices": ["Replace", "Add number suffix"], **hide_dirs}),
    "grid_save": OptionInfo(True, "Always save all generated image grids"),
    # 定义选项信息，指定文件格式为 'png'，用于网格
    "grid_format": OptionInfo('png', 'File format for grids'),
    # 定义选项信息，指定是否在保存网格时将扩展信息（种子、提示）添加到文件名中
    "grid_extended_filename": OptionInfo(False, "Add extended info (seed, prompt) to filename when saving grid"),
    # 定义选项信息，指定是否仅在网格由多个图片组成时保存
    "grid_only_if_multiple": OptionInfo(True, "Do not save grids consisting of one picture"),
    # 定义选项信息，指定是否在网格中防止空白位置（当设置为自动检测时）
    "grid_prevent_empty_spots": OptionInfo(False, "Prevent empty spots in grid (when set to autodetect)"),
    # 定义选项信息，指定存档文件名模式，链接到 wiki 页面
    "grid_zip_filename_pattern": OptionInfo("", "Archive filename pattern", component_args=hide_dirs).link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Images-Filename-Name-and-Subdirectory"),
    # 定义选项信息，指定网格行数，使用 -1 自动检测，使用 0 与批处理大小相同
    "n_rows": OptionInfo(-1, "Grid row count; use -1 for autodetect and 0 for it to be same as batch size", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}),
    # 定义选项信息，指定图像网格的字体
    "font": OptionInfo("", "Font for image grids that have text"),
    # 定义选项信息，指定图像网格中活动文本的颜色
    "grid_text_active_color": OptionInfo("#000000", "Text color for image grids", ui_components.FormColorPicker, {}),
    # 定义选项信息，指定图像网格中非活动文本的颜色
    "grid_text_inactive_color": OptionInfo("#999999", "Inactive text color for image grids", ui_components.FormColorPicker, {}),
    # 定义选项信息，指定图像网格的背景颜色
    "grid_background_color": OptionInfo("#ffffff", "Background color for image grids", ui_components.FormColorPicker, {}),

    # 定义选项信息，指定是否在进行面部恢复之前保存图像的副本
    "save_images_before_face_restoration": OptionInfo(False, "Save a copy of image before doing face restoration."),
    # 定义选项信息，指定是否在应用高分辨率修复之前保存图像的副本
    "save_images_before_highres_fix": OptionInfo(False, "Save a copy of image before applying highres fix."),
    # 定义选项信息，指定是否在将颜色校正应用于 img2img 结果之前保存图像的副本
    "save_images_before_color_correction": OptionInfo(False, "Save a copy of image before applying color correction to img2img results"),
    # 定义选项信息，指定是否保存灰度蒙版的副本用于修补
    "save_mask": OptionInfo(False, "For inpainting, save a copy of the greyscale mask"),
    # 定义选项信息，指定是否保存掩膜的复合副本用于修补
    "save_mask_composite": OptionInfo(False, "For inpainting, save a masked composite"),
    # 定义选项信息，指定保存 jpeg 图像的质量
    "jpeg_quality": OptionInfo(80, "Quality for saved jpeg images", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
    # 定义选项信息，指定是否对 webp 图像使用无损压缩
    "webp_lossless": OptionInfo(False, "Use lossless compression for webp images"),
    # 设置是否将大图像保存为 JPG 格式的选项信息
    "export_for_4chan": OptionInfo(True, "Save copy of large images as JPG").info("if the file size is above the limit, or either width or height are above the limit"),
    # 设置上述选项的文件大小限制，单位为 MB
    "img_downscale_threshold": OptionInfo(4.0, "File size limit for the above option, MB", gr.Number),
    # 设置上述选项的宽度/高度限制，单位为像素
    "target_side_length": OptionInfo(4000, "Width/height limit for the above option, in pixels", gr.Number),
    # 设置最大图像尺寸，单位为百万像素
    "img_max_size_mp": OptionInfo(200, "Maximum image size", gr.Number).info("in megapixels"),

    # 设置在批处理过程中是否使用原始名称作为输出文件名的选项信息
    "use_original_name_batch": OptionInfo(True, "Use original name for output filename during batch process in extras tab"),
    # 设置是否在额外选项卡中使用上采样器名称作为文件名后缀的选项信息
    "use_upscaler_name_as_suffix": OptionInfo(False, "Use upscaler name as filename suffix in the extras tab"),
    # 设置是否仅在使用“保存”按钮时保存单个选定的图像的选项信息
    "save_selected_only": OptionInfo(True, "When using 'Save' button, only save a single selected image"),
    # 设置是否在使用 img2img 时保存初始图像的选项信息
    "save_init_img": OptionInfo(False, "Save init images when using img2img"),

    # 设置临时图像目录的选项信息，留空表示使用默认目录
    "temp_dir":  OptionInfo("", "Directory for temporary images; leave empty for default"),
    # 设置是否在启动 webui 时清理非默认临时目录的选项信息
    "clean_temp_dir_at_start": OptionInfo(False, "Cleanup non-default temporary directory when starting webui"),

    # 设置是否保存不完整图像的选项信息
    "save_incomplete_images": OptionInfo(False, "Save incomplete images").info("save images that has been interrupted in mid-generation; even if not saved, they will still show up in webui output."),

    # 设置图像生成后是否播放通知音频的选项信息，需要在根目录中存在 notification.mp3 文件
    "notification_audio": OptionInfo(True, "Play notification sound after image generation").info("notification.mp3 should be present in the root directory").needs_reload_ui(),
    # 设置通知音频的音量大小的选项信息，范围为 0 到 100
    "notification_volume": OptionInfo(100, "Notification sound volume", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1}).info("in %"),
# 更新选项模板，添加保存路径相关选项
options_templates.update(options_section(('saving-paths', "Paths for saving", "saving"), {
    # 输出目录样本图像的选项信息
    "outdir_samples": OptionInfo("", "Output directory for images; if empty, defaults to three directories below", component_args=hide_dirs),
    # 输出目录txt2img图像的选项信息
    "outdir_txt2img_samples": OptionInfo("outputs/txt2img-images", 'Output directory for txt2img images', component_args=hide_dirs),
    # 输出目录img2img图像的选项信息
    "outdir_img2img_samples": OptionInfo("outputs/img2img-images", 'Output directory for img2img images', component_args=hide_dirs),
    # 输出目录extras图像的选项信息
    "outdir_extras_samples": OptionInfo("outputs/extras-images", 'Output directory for images from extras tab', component_args=hide_dirs),
    # 输出目录网格的选项信息
    "outdir_grids": OptionInfo("", "Output directory for grids; if empty, defaults to two directories below", component_args=hide_dirs),
    # 输出目录txt2img网格的选项信息
    "outdir_txt2img_grids": OptionInfo("outputs/txt2img-grids", 'Output directory for txt2img grids', component_args=hide_dirs),
    # 输出目录img2img网格的选项信息
    "outdir_img2img_grids": OptionInfo("outputs/img2img-grids", 'Output directory for img2img grids', component_args=hide_dirs),
    # 输出目录保存的选项信息
    "outdir_save": OptionInfo("log/images", "Directory for saving images using the Save button", component_args=hide_dirs),
    # 输出目录初始图像的选项信息
    "outdir_init_images": OptionInfo("outputs/init-images", "Directory for saving init images when using img2img", component_args=hide_dirs),
}))

# 更新选项模板，添加保存到目录相关选项
options_templates.update(options_section(('saving-to-dirs', "Saving to a directory", "saving"), {
    # 是否保存图像到子目录的选项信息
    "save_to_dirs": OptionInfo(True, "Save images to a subdirectory"),
    # 是否保存网格到子目录的选项信息
    "grid_save_to_dirs": OptionInfo(True, "Save grids to a subdirectory"),
    # 使用“保存”按钮时是否保存图像到子目录的选项信息
    "use_save_to_dirs_for_ui": OptionInfo(False, "When using \"Save\" button, save images to a subdirectory"),
    # 目录名称模式的选项信息
    "directories_filename_pattern": OptionInfo("[date]", "Directory name pattern", component_args=hide_dirs).link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Images-Filename-Name-and-Subdirectory"),
    # 定义一个名为"directories_max_prompt_words"的选项信息对象，包含最大提示词数的设置
    "directories_max_prompt_words": OptionInfo(8, "Max prompt words for [prompt_words] pattern", gr.Slider, {"minimum": 1, "maximum": 20, "step": 1, **hide_dirs}),
# 更新选项模板，添加关于图像放大的选项
options_templates.update(options_section(('upscaling', "Upscaling", "postprocessing"), {
    # ESRGAN放大器的瓦片大小选项
    "ESRGAN_tile": OptionInfo(192, "Tile size for ESRGAN upscalers.", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}).info("0 = no tiling"),
    # ESRGAN放大器的瓦片重叠选项
    "ESRGAN_tile_overlap": OptionInfo(8, "Tile overlap for ESRGAN upscalers.", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}).info("Low values = visible seam"),
    # 可用的Real-ESRGAN模型选择
    "realesrgan_enabled_models": OptionInfo(["R-ESRGAN 4x+", "R-ESRGAN 4x+ Anime6B"], "Select which Real-ESRGAN models to show in the web UI.", gr.CheckboxGroup, lambda: {"choices": shared_items.realesrgan_models_names()}),
    # img2img的放大器选项
    "upscaler_for_img2img": OptionInfo(None, "Upscaler for img2img", gr.Dropdown, lambda: {"choices": [x.name for x in shared.sd_upscalers]}),
}))

# 更新选项模板，添加关于人脸修复的选项
options_templates.update(options_section(('face-restoration', "Face restoration", "postprocessing"), {
    # 是否进行人脸修复的选项
    "face_restoration": OptionInfo(False, "Restore faces", infotext='Face restoration').info("will use a third-party model on generation result to reconstruct faces"),
    # 人脸修复模型选择
    "face_restoration_model": OptionInfo("CodeFormer", "Face restoration model", gr.Radio, lambda: {"choices": [x.name() for x in shared.face_restorers]}),
    # CodeFormer权重选项
    "code_former_weight": OptionInfo(0.5, "CodeFormer weight", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}).info("0 = maximum effect; 1 = minimum effect"),
    # 是否将人脸修复模型从VRAM移动到RAM的选项
    "face_restoration_unload": OptionInfo(False, "Move face restoration model from VRAM into RAM after processing"),
}))

# 更新选项模板，添加关于系统设置的选项
options_templates.update(options_section(('system', "System", "system"), {
    # 自动在启动时在浏览器中打开webui的选项
    "auto_launch_browser": OptionInfo("Local", "Automatically open webui in browser on startup", gr.Radio, lambda: {"choices": ["Disable", "Local", "Remote"]}),
    # 在生成时是否在控制台打印提示信息的选项
    "enable_console_prompts": OptionInfo(shared.cmd_opts.enable_console_prompts, "Print prompts to console when generating with txt2img and img2img."),
    # 是否在控制台显示警告信息的选项
    "show_warnings": OptionInfo(False, "Show warnings in console.").needs_reload_ui(),
    # 显示 Gradio 废弃警告在控制台中
    "show_gradio_deprecation_warnings": OptionInfo(True, "Show gradio deprecation warnings in console.").needs_reload_ui(),
    # 在生成期间每秒 VRAM 使用情况的轮询次数
    "memmon_poll_rate": OptionInfo(8, "VRAM usage polls per second during generation.", gr.Slider, {"minimum": 0, "maximum": 40, "step": 1}).info("0 = disable"),
    # 总是将所有生成信息打印到标准输出
    "samples_log_stdout": OptionInfo(False, "Always print all generation info to standard output"),
    # 在控制台上添加第二个进度条，显示整个作业的进度
    "multiple_tqdm": OptionInfo(True, "Add a second progress bar to the console that shows progress for an entire job."),
    # 将额外的超网络信息打印到控制台
    "print_hypernet_extra": OptionInfo(False, "Print extra hypernetwork information to console."),
    # 加载隐藏目录中的模型/文件
    "list_hidden_files": OptionInfo(True, "Load models/files in hidden directories").info("directory is hidden if its name starts with \".\""),
    # 禁用内存映射加载 .safetensors 文件
    "disable_mmap_load_safetensors": OptionInfo(False, "Disable memmapping for loading .safetensors files.").info("fixes very slow loading speed in some cases"),
    # 防止 Stability-AI 的 ldm/sgm 模块向控制台打印噪音
    "hide_ldm_prints": OptionInfo(True, "Prevent Stability-AI's ldm/sgm modules from printing noise to console."),
    # 在使用 ctrl+c 退出程序之前打印堆栈跟踪
    "dump_stacks_on_signal": OptionInfo(False, "Print stack traces before exiting the program with ctrl+c."),
# 更新选项模板，添加 API 部分的选项
options_templates.update(options_section(('API', "API", "system"), {
    # 允许在 API 中使用 http:// 和 https:// URL 作为输入图像
    "api_enable_requests": OptionInfo(True, "Allow http:// and https:// URLs for input images in API", restrict_api=True),
    # 禁止访问本地资源的 URL
    "api_forbid_local_requests": OptionInfo(True, "Forbid URLs to local resources", restrict_api=True),
    # 请求的用户代理
    "api_useragent": OptionInfo("", "User agent for requests", restrict_api=True),
}))

# 更新选项模板，添加训练部分的选项
options_templates.update(options_section(('training', "Training", "training"), {
    # 在训练时将 VAE 和 CLIP 移动到 RAM 中，节省 VRAM
    "unload_models_when_training": OptionInfo(False, "Move VAE and CLIP to RAM when training if possible. Saves VRAM."),
    # 打开 DataLoader 的 pin_memory 功能，训练速度稍快但可能增加内存使用
    "pin_memory": OptionInfo(False, "Turn on pin_memory for DataLoader. Makes training slightly faster but can increase memory usage."),
    # 将优化器状态保存为单独的 *.optim 文件，可以用于恢复嵌入或 HN 的训练
    "save_optimizer_state": OptionInfo(False, "Saves Optimizer state as separate *.optim file. Training of embedding or HN can be resumed with the matching optim file."),
    # 每次训练开始时将文本反演和超网络设置保存到文本文件中
    "save_training_settings_to_txt": OptionInfo(True, "Save textual inversion and hypernet settings to a text file whenever training starts."),
    # 文件名的单词正则表达式
    "dataset_filename_word_regex": OptionInfo("", "Filename word regex"),
    # 文件名连接字符串
    "dataset_filename_join_string": OptionInfo(" ", "Filename join string"),
    # 每个 epoch 中单个输入图像的重复次数，仅用于显示 epoch 数
    "training_image_repeats_per_epoch": OptionInfo(1, "Number of repeats for a single input image per epoch; used only for displaying epoch number", gr.Number, {"precision": 0}),
    # 每 N 步保存一个包含损失的 csv 到日志目录，设置为 0 禁用
    "training_write_csv_every": OptionInfo(500, "Save an csv containing the loss to log directory every N steps, 0 to disable"),
    # 在训练时使用交叉注意力优化
    "training_xattention_optimizations": OptionInfo(False, "Use cross attention optimizations while training"),
    # 启用 tensorboard 日志记录
    "training_enable_tensorboard": OptionInfo(False, "Enable tensorboard logging."),
    # 在 tensorboard 中保存生成的图像
    "training_tensorboard_save_images": OptionInfo(False, "Save generated images within tensorboard."),
    # 每隔多少秒刷新一次 tensorboard 事件和摘要到磁盘
    "training_tensorboard_flush_every": OptionInfo(120, "How often, in seconds, to flush the pending tensorboard events and summaries to disk."),
}))
# 更新选项模板，将稳定扩散部分的选项添加到选项模板中
options_templates.update(options_section(('sd', "Stable Diffusion", "sd"), {
    # 设置稳定扩散模型检查点选项，包括下拉框、选择项、刷新函数和信息文本
    "sd_model_checkpoint": OptionInfo(None, "Stable Diffusion checkpoint", gr.Dropdown, lambda: {"choices": shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short)}, refresh=shared_items.refresh_checkpoints, infotext='Model hash'),
    # 设置加载的最大检查点数量选项，包括滑块和取值范围
    "sd_checkpoints_limit": OptionInfo(1, "Maximum number of checkpoints loaded at the same time", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}),
    # 设置是否将模型保留在 CPU 中的选项，包括布尔值和信息文本
    "sd_checkpoints_keep_in_cpu": OptionInfo(True, "Only keep one model on device").info("will keep models other than the currently used one in RAM rather than VRAM"),
    # 设置缓存在 RAM 中的检查点数量选项，包括滑块和信息文本
    "sd_checkpoint_cache": OptionInfo(0, "Checkpoints to cache in RAM", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}).info("obsolete; set to 0 and use the two settings above instead"),
    # 设置 SD Unet 模型选项，包括下拉框、选择项、刷新函数和信息文本
    "sd_unet": OptionInfo("Automatic", "SD Unet", gr.Dropdown, lambda: {"choices": shared_items.sd_unet_items()}, refresh=shared_items.refresh_unet_list).info("choose Unet model: Automatic = use one with same filename as checkpoint; None = use Unet from checkpoint"),
    # 设置是否启用 K 个采样器中的量化以获得更清晰和更干净的结果的选项，包括需要重新加载 UI
    "enable_quantization": OptionInfo(False, "Enable quantization in K samplers for sharper and cleaner results. This may change existing seeds").needs_reload_ui(),
    # 设置是否启用强调选项，包括布尔值和信息文本
    "enable_emphasis": OptionInfo(True, "Enable emphasis").info("use (text) to make model pay more attention to text and [text] to make it pay less attention"),
    # 设置是否启用批量种子选项，使 K-扩散采样器在批处理中生成与单个图像相同的图像
    "enable_batch_seeds": OptionInfo(True, "Make K-diffusion samplers produce same images in a batch as when making a single image"),
    # 设置逗号填充回溯选项，包括滑块和信息文本
    "comma_padding_backtrack": OptionInfo(20, "Prompt word wrap length limit", gr.Slider, {"minimum": 0, "maximum": 74, "step": 1}).info("in tokens - for texts shorter than specified, if they don't fit into 75 token limit, move them to the next 75 token chunk"),
    # 定义名为"CLIP_stop_at_last_layers"的选项信息，包括默认值、显示名称、类型、滑块参数、信息文本和链接
    "CLIP_stop_at_last_layers": OptionInfo(1, "Clip skip", gr.Slider, {"minimum": 1, "maximum": 12, "step": 1}, infotext="Clip skip").link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#clip-skip").info("ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer"),
    
    # 定义名为"upcast_attn"的选项信息，包括默认值、显示名称和描述
    "upcast_attn": OptionInfo(False, "Upcast cross attention layer to float32"),
    
    # 定义名为"randn_source"的选项信息，包括默认值、显示名称、类型、单选框参数、信息文本和描述
    "randn_source": OptionInfo("GPU", "Random number generator source.", gr.Radio, {"choices": ["GPU", "CPU", "NV"]}, infotext="RNG").info("changes seeds drastically; use CPU to produce the same picture across different videocard vendors; use NV to produce same picture as on NVidia videocards"),
    
    # 定义名为"tiling"的选项信息，包括默认值、显示名称、信息文本和描述
    "tiling": OptionInfo(False, "Tiling", infotext='Tiling').info("produce a tileable picture"),
    
    # 定义名为"hires_fix_refiner_pass"的选项信息，包括默认值、显示名称、类型、单选框参数、信息文本和描述
    "hires_fix_refiner_pass": OptionInfo("second pass", "Hires fix: which pass to enable refiner for", gr.Radio, {"choices": ["first pass", "second pass", "both passes"]}, infotext="Hires refiner"),
# 更新选项模板，添加稳定扩散 XL 部分的选项
options_templates.update(options_section(('sdxl', "Stable Diffusion XL", "sd"), {
    # 设置裁剪顶部坐标选项
    "sdxl_crop_top": OptionInfo(0, "crop top coordinate"),
    # 设置裁剪左侧坐标选项
    "sdxl_crop_left": OptionInfo(0, "crop left coordinate"),
    # 设置稳定扩散 XL 低美学分数选项
    "sdxl_refiner_low_aesthetic_score": OptionInfo(2.5, "SDXL low aesthetic score", gr.Number).info("used for refiner model negative prompt"),
    # 设置稳定扩散 XL 高美学分数选项
    "sdxl_refiner_high_aesthetic_score": OptionInfo(6.0, "SDXL high aesthetic score", gr.Number).info("used for refiner model prompt"),
}))

# 更新选项模板，添加 VAE 部分的选项
options_templates.update(options_section(('vae', "VAE", "sd"), {
    # 设置 VAE 解释选项
    "sd_vae_explanation": OptionHTML("""
<abbr title='Variational autoencoder'>VAE</abbr> is a neural network that transforms a standard <abbr title='red/green/blue'>RGB</abbr>
image into latent space representation and back. Latent space representation is what stable diffusion is working on during sampling
(i.e. when the progress bar is between empty and full). For txt2img, VAE is used to create a resulting image after the sampling is finished.
For img2img, VAE is used to process user's input image before the sampling, and to create an image after sampling.
"""),
    # 设置 VAE 缓存检查点选项
    "sd_vae_checkpoint_cache": OptionInfo(0, "VAE Checkpoints to cache in RAM", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
    # 设置 SD VAE 选项
    "sd_vae": OptionInfo("Automatic", "SD VAE", gr.Dropdown, lambda: {"choices": shared_items.sd_vae_items()}, refresh=shared_items.refresh_vae_list, infotext='VAE').info("choose VAE model: Automatic = use one with same filename as checkpoint; None = use VAE from checkpoint"),
    # 设置每个模型偏好的 VAE 覆盖选项
    "sd_vae_overrides_per_model_preferences": OptionInfo(True, "Selected VAE overrides per-model preferences").info("you can set per-model VAE either by editing user metadata for checkpoints, or by making the VAE have same name as checkpoint"),
    # 创建一个名为"auto_vae_precision"的选项，值为True，表示自动将VAE还原为32位浮点数
    # 当在VAE中产生带有NaN的张量时触发；在这种情况下禁用该选项将导致黑色方块图像
    "auto_vae_precision": OptionInfo(True, "Automatically revert VAE to 32-bit floats").info("triggers when a tensor with NaNs is produced in VAE; disabling the option in this case will result in a black square image"),
    
    # 创建一个名为"sd_vae_encode_method"的选项，初始值为"Full"，表示编码的VAE类型
    # 提供单选按钮选择"Full"或"TAESD"，用于图像编码到潜在空间（在img2img、hires-fix或inpaint mask中使用）
    "sd_vae_encode_method": OptionInfo("Full", "VAE type for encode", gr.Radio, {"choices": ["Full", "TAESD"]}, infotext='VAE Encoder').info("method to encode image to latent (use in img2img, hires-fix or inpaint mask)"),
    
    # 创建一个名为"sd_vae_decode_method"的选项，初始值为"Full"，表示解码的VAE类型
    # 提供单选按钮选择"Full"或"TAESD"，用于潜在空间解码为图像
    "sd_vae_decode_method": OptionInfo("Full", "VAE type for decode", gr.Radio, {"choices": ["Full", "TAESD"]}, infotext='VAE Decoder').info("method to decode latent to image"),
# 更新选项模板，将 'img2img' 作为键，包含一系列选项的字典作为值
options_templates.update(options_section(('img2img', "img2img", "sd"), {
    # 设置修补掩模的权重选项信息
    "inpainting_mask_weight": OptionInfo(1.0, "Inpainting conditioning mask strength", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext='Conditional mask weight'),
    # 设置 img2img 的初始噪声倍增器选项信息
    "initial_noise_multiplier": OptionInfo(1.0, "Noise multiplier for img2img", gr.Slider, {"minimum": 0.0, "maximum": 1.5, "step": 0.001}, infotext='Noise multiplier'),
    # 设置 img2img 和 hires 修复的额外噪声倍增器选项信息
    "img2img_extra_noise": OptionInfo(0.0, "Extra noise multiplier for img2img and hires fix", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext='Extra noise').info("0 = disabled (default); should be lower than denoising strength"),
    # 设置是否对 img2img 结果应用颜色校正的选项信息
    "img2img_color_correction": OptionInfo(False, "Apply color correction to img2img results to match original colors."),
    # 设置是否对 img2img 进行固定步数的选项信息
    "img2img_fix_steps": OptionInfo(False, "With img2img, do exactly the amount of steps the slider specifies.").info("normally you'd do less with less denoising"),
    # 设置用于填充输入图像透明部分的背景颜色选项信息
    "img2img_background_color": OptionInfo("#ffffff", "With img2img, fill transparent parts of the input image with this color.", ui_components.FormColorPicker, {}),
    # 设置图像编辑器的高度选项信息
    "img2img_editor_height": OptionInfo(720, "Height of the image editor", gr.Slider, {"minimum": 80, "maximum": 1600, "step": 1}).info("in pixels").needs_reload_ui(),
    # 设置 img2img 初始画笔颜色选项信息
    "img2img_sketch_default_brush_color": OptionInfo("#ffffff", "Sketch initial brush color", ui_components.FormColorPicker, {}).info("default brush color of img2img sketch").needs_reload_ui(),
    # 设置修补掩模画笔颜色选项信息
    "img2img_inpaint_mask_brush_color": OptionInfo("#ffffff", "Inpaint mask brush color", ui_components.FormColorPicker,  {}).info("brush color of inpaint mask").needs_reload_ui(),
    # 设置修补草图初始画笔颜色选项信息
    "img2img_inpaint_sketch_default_brush_color": OptionInfo("#ffffff", "Inpaint sketch initial brush color", ui_components.FormColorPicker, {}).info("default brush color of img2img inpaint sketch").needs_reload_ui(),
    # 设置是否在修补时将灰度掩模包含在结果中的选项信息
    "return_mask": OptionInfo(False, "For inpainting, include the greyscale mask in results for web"),
    # 创建名为"return_mask_composite"的选项信息对象，初始值为False，用于在网页结果中包含遮罩合成图像的选项
    "return_mask_composite": OptionInfo(False, "For inpainting, include masked composite in results for web"),
    # 创建名为"img2img_batch_show_results_limit"的选项信息对象，初始值为32，用于在UI中显示前N个img2img批处理结果，滑块类型，附带参数信息
    "img2img_batch_show_results_limit": OptionInfo(32, "Show the first N batch img2img results in UI", gr.Slider, {"minimum": -1, "maximum": 1000, "step": 1}).info('0: disable, -1: show all images. Too many images can cause lag'),
# 更新选项模板，添加优化部分的选项
options_templates.update(options_section(('optimizations', "Optimizations", "sd"), {
    # 交叉注意力优化选项，使用下拉框，选项为自动生成的交叉注意力优化列表
    "cross_attention_optimization": OptionInfo("Automatic", "Cross attention optimization", gr.Dropdown, lambda: {"choices": shared_items.cross_attention_optimizations()}),
    # 负向引导最小 sigma 值选项，使用滑块，范围为 0 到 15，步长为 0.01，链接到 PR 页面，提供信息
    "s_min_uncond": OptionInfo(0.0, "Negative Guidance minimum sigma", gr.Slider, {"minimum": 0.0, "maximum": 15.0, "step": 0.01}).link("PR", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9177").info("skip negative prompt for some steps when the image is almost ready; 0=disable, higher=faster"),
    # Token 合并比例选项，使用滑块，范围为 0 到 0.9，步长为 0.1，提供信息和链接到 PR 页面
    "token_merging_ratio": OptionInfo(0.0, "Token merging ratio", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}, infotext='Token merging ratio').link("PR", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9256").info("0=disable, higher=faster"),
    # 图像到图像的 Token 合并比例选项，使用滑块，范围为 0 到 0.9，步长为 0.1，提供信息
    "token_merging_ratio_img2img": OptionInfo(0.0, "Token merging ratio for img2img", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}).info("only applies if non-zero and overrides above"),
    # 高分辨率通道的 Token 合并比例选项，使用滑块，范围为 0 到 0.9，步长为 0.1，提供信息
    "token_merging_ratio_hr": OptionInfo(0.0, "Token merging ratio for high-res pass", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}, infotext='Token merging ratio hr').info("only applies if non-zero and overrides above"),
    # 填充正向引导/负向引导使长度相同选项，提供信息
    "pad_cond_uncond": OptionInfo(False, "Pad prompt/negative prompt to be same length", infotext='Pad conds').info("improves performance when prompt and negative prompt have different lengths; changes seeds"),
    # 持久性条件缓存选项，提供信息
    "persistent_cond_cache": OptionInfo(True, "Persistent cond cache").info("do not recalculate conds from prompts if prompts have not changed since previous calculation"),
    # 批量条件/无条件选项，提供信息
    "batch_cond_uncond": OptionInfo(True, "Batch cond/uncond").info("do both conditional and unconditional denoising in one batch; uses a bit more VRAM during sampling, but improves speed; previously this was controlled by --always-batch-cond-uncond comandline argument"),
}))
# 更新选项模板，将兼容性部分的选项添加到模板中
options_templates.update(options_section(('compatibility', "Compatibility", "sd"), {
    # 使用旧的强调实现。可以用于复制旧种子。
    "use_old_emphasis_implementation": OptionInfo(False, "Use old emphasis implementation. Can be useful to reproduce old seeds."),
    # 使用旧的卡拉斯调度器 sigmas（0.1 到 10）。
    "use_old_karras_scheduler_sigmas": OptionInfo(False, "Use old karras scheduler sigmas (0.1 to 10)."),
    # 不使 DPM++ SDE 在不同批次大小之间确定性。
    "no_dpmpp_sde_batch_determinism": OptionInfo(False, "Do not make DPM++ SDE deterministic across different batch sizes."),
    # 对 hires 修复，使用宽度/高度滑块设置最终分辨率，而不是第一次通过（禁用 Upscale by, Resize width/height to）。
    "use_old_hires_fix_width_height": OptionInfo(False, "For hires fix, use width/height sliders to set final resolution rather than first pass (disables Upscale by, Resize width/height to)."),
    # 不修复第二阶采样器的时间表。
    "dont_fix_second_order_samplers_schedule": OptionInfo(False, "Do not fix prompt schedule for second order samplers."),
    # 对 hires 修复，使用第一次通过的额外网络计算第二次通过的条件。
    "hires_fix_use_firstpass_conds": OptionInfo(False, "For hires fix, calculate conds of second pass using extra networks of first pass."),
    # 使用旧的提示编辑时间表。
    "use_old_scheduling": OptionInfo(False, "Use old prompt editing timelines.", infotext="Old prompt editing timelines").info("For [red:green:N]; old: If N < 1, it's a fraction of steps (and hires fix uses range from 0 to 1), if N >= 1, it's an absolute number of steps; new: If N has a decimal point in it, it's a fraction of steps (and hires fix uses range from 1 to 2), othewrwise it's an absolute number of steps"),
}))

# 更新选项模板，将询问部分的选项添加到模板中
options_templates.update(options_section(('interrogate', "Interrogate"), {
    # 保持模型在 VRAM 中。
    "interrogate_keep_models_in_memory": OptionInfo(False, "Keep models in VRAM"),
    # 在结果中包含模型标签匹配的排名。
    "interrogate_return_ranks": OptionInfo(False, "Include ranks of model tags matches in results.").info("booru only"),
    # 限制 BLIP: num_beams 的数量。
    "interrogate_clip_num_beams": OptionInfo(1, "BLIP: num_beams", gr.Slider, {"minimum": 1, "maximum": 16, "step": 1}),
    # 限制 BLIP: 最小描述长度。
    "interrogate_clip_min_length": OptionInfo(24, "BLIP: minimum description length", gr.Slider, {"minimum": 1, "maximum": 128, "step": 1}),
    # 设置选项"interrogate_clip_max_length"，包括默认值48，描述信息"BLIP: maximum description length"，类型为滑块，限制最小值为1，最大值为256，步长为1
    "interrogate_clip_max_length": OptionInfo(48, "BLIP: maximum description length", gr.Slider, {"minimum": 1, "maximum": 256, "step": 1}),
    
    # 设置选项"interrogate_clip_dict_limit"，包括默认值1500，描述信息"CLIP: maximum number of lines in text file"，没有限制时显示"0 = No limit"
    "interrogate_clip_dict_limit": OptionInfo(1500, "CLIP: maximum number of lines in text file").info("0 = No limit"),
    
    # 设置选项"interrogate_clip_skip_categories"，包括默认值空列表，描述信息"CLIP: skip inquire categories"，类型为复选框组，选项由interrogate.category_types()生成，刷新时更新选项
    "interrogate_clip_skip_categories": OptionInfo([], "CLIP: skip inquire categories", gr.CheckboxGroup, lambda: {"choices": interrogate.category_types()}, refresh=interrogate.category_types),
    
    # 设置选项"interrogate_deepbooru_score_threshold"，包括默认值0.5，描述信息"deepbooru: score threshold"，类型为滑块，限制最小值为0，最大值为1，步长为0.01
    "interrogate_deepbooru_score_threshold": OptionInfo(0.5, "deepbooru: score threshold", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    
    # 设置选项"deepbooru_sort_alpha"，包括默认值True，描述信息"deepbooru: sort tags alphabetically"，如果为False则按分数排序
    "deepbooru_sort_alpha": OptionInfo(True, "deepbooru: sort tags alphabetically").info("if not: sort by score"),
    
    # 设置选项"deepbooru_use_spaces"，包括默认值True，描述信息"deepbooru: use spaces in tags"，如果为False则使用下划线
    "deepbooru_use_spaces": OptionInfo(True, "deepbooru: use spaces in tags").info("if not: use underscores"),
    
    # 设置选项"deepbooru_escape"，包括默认值True，描述信息"deepbooru: escape (\\) brackets"，使括号被视为文字而不是强调
    "deepbooru_escape": OptionInfo(True, "deepbooru: escape (\\) brackets").info("so they are used as literal brackets and not for emphasis"),
    
    # 设置选项"deepbooru_filter_tags"，包括默认值空字符串，描述信息"deepbooru: filter out those tags"，用逗号分隔
    "deepbooru_filter_tags": OptionInfo("", "deepbooru: filter out those tags").info("separate by comma"),
# 更新选项模板，添加额外网络部分的选项
options_templates.update(options_section(('extra_networks', "Extra Networks", "sd"), {
    # 显示隐藏目录的选项信息
    "extra_networks_show_hidden_directories": OptionInfo(True, "Show hidden directories").info("directory is hidden if its name starts with \".\"."),
    # 添加 '/' 到目录按钮开头的选项信息
    "extra_networks_dir_button_function": OptionInfo(False, "Add a '/' to the beginning of directory buttons").info("Buttons will display the contents of the selected directory without acting as a search filter."),
    # 显示隐藏目录中模型卡片的选项信息
    "extra_networks_hidden_models": OptionInfo("When searched", "Show cards for models in hidden directories", gr.Radio, {"choices": ["Always", "When searched", "Never"]}).info('"When searched" option will only show the item when the search string has 4 characters or more'),
    # 额外网络默认乘数的选项信息
    "extra_networks_default_multiplier": OptionInfo(1.0, "Default multiplier for extra networks", gr.Slider, {"minimum": 0.0, "maximum": 2.0, "step": 0.01}),
    # 额外网络卡片宽度的选项信息
    "extra_networks_card_width": OptionInfo(0, "Card width for Extra Networks").info("in pixels"),
    # 额外网络卡片高度的选项信息
    "extra_networks_card_height": OptionInfo(0, "Card height for Extra Networks").info("in pixels"),
    # 额外网络卡片文本比例的选项信息
    "extra_networks_card_text_scale": OptionInfo(1.0, "Card text scale", gr.Slider, {"minimum": 0.0, "maximum": 2.0, "step": 0.01}).info("1 = original size"),
    # 在卡片上显示描述的选项信息
    "extra_networks_card_show_desc": OptionInfo(True, "Show description on card"),
    # 额外网络卡片默认排序字段的选项信息
    "extra_networks_card_order_field": OptionInfo("Path", "Default order field for Extra Networks cards", gr.Dropdown, {"choices": ['Path', 'Name', 'Date Created', 'Date Modified']}).needs_reload_ui(),
    # 额外网络卡片默认排序方式的选项信息
    "extra_networks_card_order": OptionInfo("Ascending", "Default order for Extra Networks cards", gr.Dropdown, {"choices": ['Ascending', 'Descending']}).needs_reload_ui(),
    # 添加额外网络到提示时的分隔符选项信息
    "extra_networks_add_text_separator": OptionInfo(" ", "Extra networks separator").info("extra text to add before <...> when adding extra network to prompt"),
    # 额外网络选项卡重新排序的选项信息
    "ui_extra_networks_tab_reorder": OptionInfo("", "Extra networks tab order").needs_reload_ui(),
    # 创建一个名为"textual_inversion_print_at_load"的选项，初始值为False，用于在加载模型时打印文本反转嵌入的列表
    "textual_inversion_print_at_load": OptionInfo(False, "Print a list of Textual Inversion embeddings when loading model"),
    # 创建一个名为"textual_inversion_add_hashes_to_infotext"的选项，初始值为True，用于将文本反转哈希添加到信息文本中
    "textual_inversion_add_hashes_to_infotext": OptionInfo(True, "Add Textual Inversion hashes to infotext"),
    # 创建一个名为"sd_hypernetwork"的选项，初始值为"None"，用于将超网络添加到提示中，是一个下拉菜单，选项为["None", *shared.hypernetworks]，并在需要时刷新shared_items.reload_hypernetworks
    "sd_hypernetwork": OptionInfo("None", "Add hypernetwork to prompt", gr.Dropdown, lambda: {"choices": ["None", *shared.hypernetworks]}, refresh=shared_items.reload_hypernetworks),
# 更新选项模板，包含 UI 提示编辑部分的选项信息
options_templates.update(options_section(('ui_prompt_editing', "Prompt editing", "ui"), {
    # 编辑提示时使用 Ctrl+up/down 调整 (attention:1.1) 的精度
    "keyedit_precision_attention": OptionInfo(0.1, "Precision for (attention:1.1) when editing the prompt with Ctrl+up/down", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001}),
    # 编辑提示时使用 Ctrl+up/down 调整 <extra networks:0.9> 的精度
    "keyedit_precision_extra": OptionInfo(0.05, "Precision for <extra networks:0.9> when editing the prompt with Ctrl+up/down", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001}),
    # 编辑提示时的单词分隔符
    "keyedit_delimiters": OptionInfo(r".,\/!?%^*;:{}=`~() ", "Word delimiters when editing the prompt with Ctrl+up/down"),
    # 编辑提示时的空白符分隔符
    "keyedit_delimiters_whitespace": OptionInfo(["Tab", "Carriage Return", "Line Feed"], "Ctrl+up/down whitespace delimiters", gr.CheckboxGroup, lambda: {"choices": ["Tab", "Carriage Return", "Line Feed"]}),
    # Alt+left/right 移动提示元素
    "keyedit_move": OptionInfo(True, "Alt+left/right moves prompt elements"),
    # 禁用提示令牌计数器
    "disable_token_counters": OptionInfo(False, "Disable prompt token counters").needs_reload_ui(),
}))

# 更新选项模板，包含画廊部分的选项信息
options_templates.update(options_section(('ui_gallery', "Gallery", "ui"), {
    # 在画廊中显示网格
    "return_grid": OptionInfo(True, "Show grid in gallery"),
    # 在画廊中不显示任何图片
    "do_not_show_images": OptionInfo(False, "Do not show any images in gallery"),
    # 启用全屏图片查看器
    "js_modal_lightbox": OptionInfo(True, "Full page image viewer: enable"),
    # 默认放大显示图片
    "js_modal_lightbox_initially_zoomed": OptionInfo(True, "Full page image viewer: show images zoomed in by default"),
    # 使用游戏手柄导航全屏图片查看器
    "js_modal_lightbox_gamepad": OptionInfo(False, "Full page image viewer: navigate with gamepad"),
    # 游戏手柄导航全屏图片查看器的重复周期
    "js_modal_lightbox_gamepad_repeat": OptionInfo(250, "Full page image viewer: gamepad repeat period").info("in milliseconds"),
    # 画廊高度
    "gallery_height": OptionInfo("", "Gallery height", gr.Textbox).info("can be any valid CSS value, for example 768px or 20em").needs_reload_ui(),
}))

# 更新选项模板，包含 UI 替代方案部分的选项信息
options_templates.update(options_section(('ui_alternatives', "UI alternatives", "ui"), {
    # 创建一个名为"compact_prompt_box"的选项信息对象，初始值为False，表示不使用紧凑的提示框布局
    # 提供关于选项作用的信息，描述将提示和负面提示放在生成选项卡内，为右侧图像留更多的垂直空间
    # 需要重新加载用户界面
    "compact_prompt_box": OptionInfo(False, "Compact prompt layout").info("puts prompt and negative prompt inside the Generate tab, leaving more vertical space for the image on the right").needs_reload_ui(),
    
    # 创建一个名为"samplers_in_dropdown"的选项信息对象，初始值为True，表示使用下拉菜单来选择采样器，而不是单选按钮组
    # 需要重新加载用户界面
    "samplers_in_dropdown": OptionInfo(True, "Use dropdown for sampler selection instead of radio group").needs_reload_ui(),
    
    # 创建一个名为"dimensions_and_batch_together"的选项信息对象，初始值为True，表示在同一行显示宽度/高度和批量滑块
    # 需要重新加载用户界面
    "dimensions_and_batch_together": OptionInfo(True, "Show Width/Height and Batch sliders in same row").needs_reload_ui(),
    
    # 创建一个名为"sd_checkpoint_dropdown_use_short"的选项信息对象，初始值为False，表示在检查点下拉菜单中使用不带路径的文件名
    # 提供关于选项作用的信息，描述在子目录中的模型（如photo/sd15.ckpt）将仅列为sd15.ckpt
    "sd_checkpoint_dropdown_use_short": OptionInfo(False, "Checkpoint dropdown: use filenames without paths").info("models in subdirectories like photo/sd15.ckpt will be listed as just sd15.ckpt"),
    
    # 创建一个名为"hires_fix_show_sampler"的选项信息对象，初始值为False，表示显示高分辨率修复的检查点和采样器选择
    # 需要重新加载用户界面
    "hires_fix_show_sampler": OptionInfo(False, "Hires fix: show hires checkpoint and sampler selection").needs_reload_ui(),
    
    # 创建一个名为"hires_fix_show_prompts"的选项信息对象，初始值为False，表示显示高分辨率修复的提示和负面提示
    # 需要重新加载用户界面
    "hires_fix_show_prompts": OptionInfo(False, "Hires fix: show hires prompt and negative prompt").needs_reload_ui(),
    
    # 创建一个名为"txt2img_settings_accordion"的选项信息对象，初始值为False，表示在txt2img中将设置隐藏在手风琴下
    # 需要重新加载用户界面
    "txt2img_settings_accordion": OptionInfo(False, "Settings in txt2img hidden under Accordion").needs_reload_ui(),
    
    # 创建一个名为"img2img_settings_accordion"的选项信息对象，初始值为False，表示在img2img中将设置隐藏在手风琴下
    # 需要重新加载用户界面
    "img2img_settings_accordion": OptionInfo(False, "Settings in img2img hidden under Accordion").needs_reload_ui(),
# 更新选项模板，将 UI 部分的选项添加到选项模板中
options_templates.update(options_section(('ui', "User interface", "ui"), {
    # 本地化选项，包括下拉框和刷新函数
    "localization": OptionInfo("None", "Localization", gr.Dropdown, lambda: {"choices": ["None"] + list(localization.localizations.keys())}, refresh=lambda: localization.list_localizations(cmd_opts.localizations_dir)).needs_reload_ui(),
    # 快速设置列表选项，包括多选下拉框和相关信息
    "quicksettings_list": OptionInfo(["sd_model_checkpoint"], "Quicksettings list", ui_components.DropdownMulti, lambda: {"choices": list(shared.opts.data_labels.keys())}).js("info", "settingsHintsShowQuicksettings").info("setting entries that appear at the top of page rather than in settings tab").needs_reload_ui(),
    # UI 标签顺序选项，包括多选下拉框
    "ui_tab_order": OptionInfo([], "UI tab order", ui_components.DropdownMulti, lambda: {"choices": list(shared.tab_names)}).needs_reload_ui(),
    # 隐藏 UI 标签选项，包括多选下拉框
    "hidden_tabs": OptionInfo([], "Hidden UI tabs", ui_components.DropdownMulti, lambda: {"choices": list(shared.tab_names)}).needs_reload_ui(),
    # UI 项目顺序选项，包括多选下拉框和相关信息
    "ui_reorder_list": OptionInfo([], "UI item order for txt2img/img2img tabs", ui_components.DropdownMulti, lambda: {"choices": list(shared_items.ui_reorder_categories())}).info("selected items appear first").needs_reload_ui(),
    # Gradio 主题选项，包括可编辑下拉框和相关信息
    "gradio_theme": OptionInfo("Default", "Gradio theme", ui_components.DropdownEditable, lambda: {"choices": ["Default"] + shared_gradio_themes.gradio_hf_hub_themes}).info("you can also manually enter any of themes from the <a href='https://huggingface.co/spaces/gradio/theme-gallery'>gallery</a>.").needs_reload_ui(),
    # 缓存 Gradio 主题选项，包括信息
    "gradio_themes_cache": OptionInfo(True, "Cache gradio themes locally").info("disable to update the selected Gradio theme"),
    # 在窗口标题中显示生成进度选项
    "show_progress_in_title": OptionInfo(True, "Show generation progress in window title."),
    # 发送种子选项
    "send_seed": OptionInfo(True, "Send seed when sending prompt or image to other interface"),
    # 发送大小选项
    "send_size": OptionInfo(True, "Send size when sending prompt or image to another interface"),
}))
# 更新选项模板，将 Infotext 部分的选项添加到选项模板中
options_templates.update(options_section(('infotext', "Infotext", "ui"), {
    # 定义一个包含 HTML 选项的字符串，用于显示信息文本的解释
# Infotext 是此软件称为包含生成参数的文本，可以用于再次生成相同的图片。
# 它显示在图像下方的用户界面中。要使用 infotext，请将其粘贴到提示中，然后单击 ↙️ 粘贴按钮。
# "enable_pnginfo": 将 infotext 写入生成图像的元数据
# "save_txt": 创建一个文本文件，其中包含每个生成图像旁边的 infotext

# "add_model_name_to_info": 将模型名称添加到 infotext
# "add_model_hash_to_info": 将模型哈希添加到 infotext
# "add_vae_name_to_info": 将 VAE 名称添加到 infotext
# "add_vae_hash_to_info": 将 VAE 哈希添加到 infotext
# "add_user_name_to_info": 在身份验证时将用户名添加到 infotext
# "add_version_to_infotext": 将程序版本添加到 infotext
# "disable_weights_auto_swap": 忽略从粘贴的 infotext 中的过去检查点信息（在从文本读取生成参数到用户界面时）
# "infotext_skip_pasting": 忽略从粘贴的 infotext 中的字段
# "infotext_styles": 从粘贴的 infotext 的提示中推断样式，根据选择应用样式到用户界面中的生成参数（当从文本读取生成参数到用户界面时）
# 更新选项模板，将 UI 部分的 Live previews 选项添加到选项模板中
options_templates.update(options_section(('ui', "Live previews", "ui"), {
    # 是否显示进度条
    "show_progressbar": OptionInfo(True, "Show progressbar"),
    # 是否显示创建图像的实时预览
    "live_previews_enable": OptionInfo(True, "Show live previews of the created image"),
    # 实时预览文件格式
    "live_previews_image_format": OptionInfo("png", "Live preview file format", gr.Radio, {"choices": ["jpeg", "png", "webp"]}),
    # 是否以网格形式显示批处理生成的所有图像的预览
    "show_progress_grid": OptionInfo(True, "Show previews of all images generated in a batch as a grid"),
    # 实时预览显示周期
    "show_progress_every_n_steps": OptionInfo(10, "Live preview display period", gr.Slider, {"minimum": -1, "maximum": 32, "step": 1}).info("in sampling steps - show new live preview image every N sampling steps; -1 = only show after completion of batch"),
    # 实时预览方法
    "show_progress_type": OptionInfo("Approx NN", "Live preview method", gr.Radio, {"choices": ["Full", "Approx NN", "Approx cheap", "TAESD"]}).info("Full = slow but pretty; Approx NN and TAESD = fast but low quality; Approx cheap = super fast but terrible otherwise"),
    # 是否允许在低 VRAM/中 VRAM 环境下使用 Full 实时预览方法
    "live_preview_allow_lowvram_full": OptionInfo(False, "Allow Full live preview method with lowvram/medvram").info("If not, Approx NN will be used instead; Full live preview method is very detrimental to speed if lowvram/medvram optimizations are enabled"),
    # 实时预览主题
    "live_preview_content": OptionInfo("Prompt", "Live preview subject", gr.Radio, {"choices": ["Combined", "Prompt", "Negative prompt"]}),
    # 进度条和预览更新周期
    "live_preview_refresh_period": OptionInfo(1000, "Progressbar and preview update period").info("in milliseconds"),
    # 在中断时返回选择的实时预览方法图像
    "live_preview_fast_interrupt": OptionInfo(False, "Return image with chosen live preview method on interrupt").info("makes interrupts faster"),
    # 在全屏图像查看器中显示实时预览
    "js_live_preview_in_modal_lightbox": OptionInfo(False, "Show Live preview in full page image viewer"),
}))
# 更新选项模板，将 Sampler parameters 部分的选项添加到选项模板中
options_templates.update(options_section(('sampler-params', "Sampler parameters", "sd"), {
    # 创建一个选项信息对象，用于隐藏用户界面中的采样器
    "hide_samplers": OptionInfo([], "Hide samplers in user interface", gr.CheckboxGroup, lambda: {"choices": [x.name for x in shared_items.list_samplers()]}).needs_reload_ui(),
    # 创建一个选项信息对象，用于设置 DDIM 的 Eta 值
    "eta_ddim": OptionInfo(0.0, "Eta for DDIM", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext='Eta DDIM').info("noise multiplier; higher = more unpredictable results"),
    # 创建一个选项信息对象，用于设置祖先采样器的 Eta 值
    "eta_ancestral": OptionInfo(1.0, "Eta for k-diffusion samplers", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext='Eta').info("noise multiplier; currently only applies to ancestral samplers (i.e. Euler a) and SDE samplers"),
    # 创建一个选项信息对象，用于设置 img2img DDIM 的离散化方式
    "ddim_discretize": OptionInfo('uniform', "img2img DDIM discretize", gr.Radio, {"choices": ['uniform', 'quad']}),
    # 创建一个选项信息对象，用于设置 sigma churn 的值
    's_churn': OptionInfo(0.0, "sigma churn", gr.Slider, {"minimum": 0.0, "maximum": 100.0, "step": 0.01}, infotext='Sigma churn').info('amount of stochasticity; only applies to Euler, Heun, and DPM2'),
    # 创建一个选项信息对象，用于设置 sigma tmin 的值
    's_tmin':  OptionInfo(0.0, "sigma tmin",  gr.Slider, {"minimum": 0.0, "maximum": 10.0, "step": 0.01}, infotext='Sigma tmin').info('enable stochasticity; start value of the sigma range; only applies to Euler, Heun, and DPM2'),
    # 创建一个选项信息对象，用于设置 sigma tmax 的值
    's_tmax':  OptionInfo(0.0, "sigma tmax",  gr.Slider, {"minimum": 0.0, "maximum": 999.0, "step": 0.01}, infotext='Sigma tmax').info("0 = inf; end value of the sigma range; only applies to Euler, Heun, and DPM2"),
    # 创建一个选项信息对象，用于设置 sigma noise 的值
    's_noise': OptionInfo(1.0, "sigma noise", gr.Slider, {"minimum": 0.0, "maximum": 1.1, "step": 0.001}, infotext='Sigma noise').info('amount of additional noise to counteract loss of detail during sampling'),
    # 创建一个选项信息对象，用于设置调度器类型
    'k_sched_type':  OptionInfo("Automatic", "Scheduler type", gr.Dropdown, {"choices": ["Automatic", "karras", "exponential", "polyexponential"]}, infotext='Schedule type').info("lets you override the noise schedule for k-diffusion samplers; choosing Automatic disables the three parameters below"),
    # 定义'sigma_min'参数，初始值为0.0，描述为"sigma min"，类型为gr.Number，infotext为'Schedule min sigma'，info为"0 = default (~0.03); minimum noise strength for k-diffusion noise scheduler"
    'sigma_min': OptionInfo(0.0, "sigma min", gr.Number, infotext='Schedule min sigma').info("0 = default (~0.03); minimum noise strength for k-diffusion noise scheduler"),
    
    # 定义'sigma_max'参数，初始值为0.0，描述为"sigma max"，类型为gr.Number，infotext为'Schedule max sigma'，info为"0 = default (~14.6); maximum noise strength for k-diffusion noise scheduler"
    'sigma_max': OptionInfo(0.0, "sigma max", gr.Number, infotext='Schedule max sigma').info("0 = default (~14.6); maximum noise strength for k-diffusion noise scheduler"),
    
    # 定义'rho'参数，初始值为0.0，描述为"rho"，类型为gr.Number，infotext为'Schedule rho'，info为"0 = default (7 for karras, 1 for polyexponential); higher values result in a steeper noise schedule (decreases faster)"
    'rho':  OptionInfo(0.0, "rho", gr.Number, infotext='Schedule rho').info("0 = default (7 for karras, 1 for polyexponential); higher values result in a steeper noise schedule (decreases faster)"),
    
    # 定义'eta_noise_seed_delta'参数，初始值为0，描述为"Eta noise seed delta"，类型为gr.Number，包含precision为0，infotext为'ENSD'，info为"ENSD; does not improve anything, just produces different results for ancestral samplers - only useful for reproducing images"
    'eta_noise_seed_delta': OptionInfo(0, "Eta noise seed delta", gr.Number, {"precision": 0}, infotext='ENSD').info("ENSD; does not improve anything, just produces different results for ancestral samplers - only useful for reproducing images"),
    
    # 定义'always_discard_next_to_last_sigma'参数，初始值为False，描述为"Always discard next-to-last sigma"，infotext为'Discard penultimate sigma'，链接到"https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/6044"
    'always_discard_next_to_last_sigma': OptionInfo(False, "Always discard next-to-last sigma", infotext='Discard penultimate sigma').link("PR", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/6044"),
    
    # 定义'sgm_noise_multiplier'参数，初始值为False，描述为"SGM noise multiplier"，infotext为'SGM noise multplier'，链接到"https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12818"，info为"Match initial noise to official SDXL implementation - only useful for reproducing images"
    'sgm_noise_multiplier': OptionInfo(False, "SGM noise multiplier", infotext='SGM noise multplier').link("PR", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12818").info("Match initial noise to official SDXL implementation - only useful for reproducing images"),
    
    # 定义'uni_pc_variant'参数，初始值为"bh1"，描述为"UniPC variant"，类型为gr.Radio，包含choices为["bh1", "bh2", "vary_coeff"]，infotext为'UniPC variant'
    'uni_pc_variant': OptionInfo("bh1", "UniPC variant", gr.Radio, {"choices": ["bh1", "bh2", "vary_coeff"]}, infotext='UniPC variant'),
    
    # 定义'uni_pc_skip_type'参数，初始值为"time_uniform"，描述为"UniPC skip type"，类型为gr.Radio，包含choices为["time_uniform", "time_quadratic", "logSNR"]，infotext为'UniPC skip type'
    'uni_pc_skip_type': OptionInfo("time_uniform", "UniPC skip type", gr.Radio, {"choices": ["time_uniform", "time_quadratic", "logSNR"]}, infotext='UniPC skip type'),
    
    # 定义'uni_pc_order'参数，初始值为3，描述为"UniPC order"，类型为gr.Slider，包含minimum为1，maximum为50，step为1，infotext为'UniPC order'，info为"must be < sampling steps"
    'uni_pc_order': OptionInfo(3, "UniPC order", gr.Slider, {"minimum": 1, "maximum": 50, "step": 1}, infotext='UniPC order').info("must be < sampling steps"),
    
    # 定义'uni_pc_lower_order_final'参数，初始值为True，描述为"UniPC lower order final"，infotext为'UniPC lower order final'
    'uni_pc_lower_order_final': OptionInfo(True, "UniPC lower order final", infotext='UniPC lower order final'),
# 更新选项模板，添加 postprocessing 部分的选项
options_templates.update(options_section(('postprocessing', "Postprocessing", "postprocessing"), {
    # 启用在 txt2img 和 img2img 标签中进行后处理操作的选项
    'postprocessing_enable_in_main_ui': OptionInfo([], "Enable postprocessing operations in txt2img and img2img tabs", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.postprocessing_scripts()]}),
    # 后处理操作顺序的选项
    'postprocessing_operation_order': OptionInfo([], "Postprocessing operation order", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.postprocessing_scripts()]}),
    # 图像放大缓存中的最大图像数量选项
    'upscaling_max_images_in_cache': OptionInfo(5, "Maximum number of images in upscaling cache", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
    # 存在标题时的操作选项
    'postprocessing_existing_caption_action': OptionInfo("Ignore", "Action for existing captions", gr.Radio, {"choices": ["Ignore", "Keep", "Prepend", "Append"]}).info("when generating captions using postprocessing; Ignore = use generated; Keep = use original; Prepend/Append = combine both"),
}))

# 更新选项模板，添加隐藏选项部分的选项
options_templates.update(options_section((None, "Hidden options"), {
    # 禁用这些扩展名的选项
    "disabled_extensions": OptionInfo([], "Disable these extensions"),
    # 禁用所有扩展名的选项
    "disable_all_extensions": OptionInfo("none", "Disable all extensions (preserves the list of disabled extensions)", gr.Radio, {"choices": ["none", "extra", "all"]}),
    # 恢复配置状态文件的选项
    "restore_config_state_file": OptionInfo("", "Config state file to restore from, under 'config-states/' folder"),
    # 当前检查点的 SHA256 哈希值选项
    "sd_checkpoint_hash": OptionInfo("", "SHA256 hash of the current checkpoint"),
}))
```