# `stable-diffusion-webui\modules\shared.py`

```
# 导入 sys 模块
import sys

# 导入 gradio 模块并重命名为 gr
import gradio as gr

# 从 modules 模块中导入 shared_cmd_options、shared_gradio_themes、options、shared_items、sd_models_types
from modules import shared_cmd_options, shared_gradio_themes, options, shared_items, sd_models_types

# 从 modules.paths_internal 模块中导入 models_path、script_path、data_path、sd_configs_path、sd_default_config、sd_model_file、default_sd_model_file、extensions_dir、extensions_builtin_dir
from modules.paths_internal import models_path, script_path, data_path, sd_configs_path, sd_default_config, sd_model_file, default_sd_model_file, extensions_dir, extensions_builtin_dir  # noqa: F401

# 从 modules 模块中导入 util
from modules import util

# 定义变量 cmd_opts 为 shared_cmd_options 模块中的 cmd_opts
cmd_opts = shared_cmd_options.cmd_opts

# 定义变量 parser 为 shared_cmd_options 模块中的 parser
parser = shared_cmd_options.parser

# 定义变量 batch_cond_uncond 为 True，旧字段，现在不再使用，而是使用 shared.opts.batch_cond_uncond
batch_cond_uncond = True

# 定义变量 parallel_processing_allowed 为 True
parallel_processing_allowed = True

# 定义变量 styles_filename 为 cmd_opts 中的 styles_file
styles_filename = cmd_opts.styles_file

# 定义变量 config_filename 为 cmd_opts 中的 ui_settings_file
config_filename = cmd_opts.ui_settings_file

# 定义变量 hide_dirs 为一个字典，根据 cmd_opts 中的 hide_ui_dir_config 决定是否可见
hide_dirs = {"visible": not cmd_opts.hide_ui_dir_config}

# 定义变量 demo 为 None
demo = None

# 定义变量 device 为 None
device = None

# 定义变量 weight_load_location 为 None
weight_load_location = None

# 定义变量 xformers_available 为 False
xformers_available = False

# 定义空字典 hypernetworks
hypernetworks = {}

# 定义空列表 loaded_hypernetworks
loaded_hypernetworks = []

# 定义变量 state 为 None
state = None

# 定义变量 prompt_styles 为 None
prompt_styles = None

# 定义变量 interrogator 为 None
interrogator = None

# 定义空列表 face_restorers
face_restorers = []

# 定义变量 options_templates 为 None
options_templates = None

# 定义变量 opts 为 None
opts = None

# 定义变量 restricted_opts 为 None
restricted_opts = None

# 定义变量 sd_model 为 None，类型为 sd_models_types.WebuiSdModel
sd_model: sd_models_types.WebuiSdModel = None

# 定义变量 settings_components 为 None，从 ui.py 分配，一个将设置名称映射到负责该设置的 gradio 组件的映射
settings_components = None

# 定义空列表 tab_names
tab_names = []

# 定义变量 latent_upscale_default_mode 为 "Latent"
latent_upscale_default_mode = "Latent"

# 定义字典 latent_upscale_modes 包含不同的缩放模式和参数
latent_upscale_modes = {
    "Latent": {"mode": "bilinear", "antialias": False},
    "Latent (antialiased)": {"mode": "bilinear", "antialias": True},
    "Latent (bicubic)": {"mode": "bicubic", "antialias": False},
    "Latent (bicubic antialiased)": {"mode": "bicubic", "antialias": True},
    "Latent (nearest)": {"mode": "nearest", "antialias": False},
    "Latent (nearest-exact)": {"mode": "nearest-exact", "antialias": False},
}

# 定义空列表 sd_upscalers
sd_upscalers = []

# 定义变量 clip_model 为 None
clip_model = None

# 定义变量 progress_print_out 为 sys.stdout
progress_print_out = sys.stdout

# 定义变量 gradio_theme 为 gr.themes.Base()
gradio_theme = gr.themes.Base()

# 定义变量 total_tqdm 为 None
total_tqdm = None

# 定义变量 mem_mon 为 None
mem_mon = None

# 从 options 模块中导入 options_section、OptionInfo、OptionHTML
options_section = options.options_section
OptionInfo = options.OptionInfo
OptionHTML = options.OptionHTML

# 定义变量 natural_sort_key 为 util 模块中的 natural_sort_key
natural_sort_key = util.natural_sort_key

# 定义变量 listfiles 为 util 模块中的 listfiles
listfiles = util.listfiles

# 定义变量 html_path 为 util 模块中的 html_path
html_path = util.html_path
# 导入 util 模块中的 html、walk_files 和 ldm_print 函数
html = util.html
walk_files = util.walk_files
ldm_print = util.ldm_print

# 导入 shared_gradio_themes 模块中的 reload_gradio_theme 函数
reload_gradio_theme = shared_gradio_themes.reload_gradio_theme

# 导入 shared_items 模块中的 list_checkpoint_tiles、refresh_checkpoints、list_samplers 和 reload_hypernetworks 函数
list_checkpoint_tiles = shared_items.list_checkpoint_tiles
refresh_checkpoints = shared_items.refresh_checkpoints
list_samplers = shared_items.list_samplers
reload_hypernetworks = shared_items.reload_hypernetworks
```