# `stable-diffusion-webui\modules\shared_cmd_options.py`

```
# 导入 os 模块
import os

# 导入 launch 模块
import launch
# 从 modules 模块中导入 cmd_args 和 script_loading
from modules import cmd_args, script_loading
# 从 modules.paths_internal 模块中导入指定的路径变量，忽略 F401 错误
from modules.paths_internal import models_path, script_path, data_path, sd_configs_path, sd_default_config, sd_model_file, default_sd_model_file, extensions_dir, extensions_builtin_dir

# 获取命令行参数解析器
parser = cmd_args.parser

# 预加载扩展
script_loading.preload_extensions(extensions_dir, parser, extension_list=launch.list_extensions(launch.args.ui_settings_file))
script_loading.preload_extensions(extensions_builtin_dir, parser)

# 如果环境变量中没有设置 IGNORE_CMD_ARGS_ERRORS，则解析命令行参数
if os.environ.get('IGNORE_CMD_ARGS_ERRORS', None) is None:
    cmd_opts = parser.parse_args()
# 否则，解析命令行参数但忽略错误
else:
    cmd_opts, _ = parser.parse_known_args()

# 设置 webui_is_non_local 标志，表示 webui 是否为非本地的
cmd_opts.webui_is_non_local = any([cmd_opts.share, cmd_opts.listen, cmd_opts.ngrok, cmd_opts.server_name])
# 设置 disable_extension_access 标志，表示是否禁用扩展访问
cmd_opts.disable_extension_access = cmd_opts.webui_is_non_local and not cmd_opts.enable_insecure_extension_access
```