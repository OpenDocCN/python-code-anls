# `stable-diffusion-webui\modules\shared_gradio_themes.py`

```py
# 导入操作系统模块
import os

# 导入 Gradio 库
import gradio as gr

# 导入自定义模块 errors 和 shared
from modules import errors, shared

# 导入自定义模块 paths_internal 中的 script_path 函数
from modules.paths_internal import script_path

# 预定义的 Gradio 主题列表，包含多个主题名称
gradio_hf_hub_themes = [
    "gradio/base",
    "gradio/glass",
    "gradio/monochrome",
    "gradio/seafoam",
    "gradio/soft",
    "gradio/dracula_test",
    "abidlabs/dracula_test",
    "abidlabs/Lime",
    "abidlabs/pakistan",
    "Ama434/neutral-barlow",
    "dawood/microsoft_windows",
    "finlaymacklon/smooth_slate",
    "Franklisi/darkmode",
    "freddyaboulton/dracula_revamped",
    "freddyaboulton/test-blue",
    "gstaff/xkcd",
    "Insuz/Mocha",
    "Insuz/SimpleIndigo",
    "JohnSmith9982/small_and_pretty",
    "nota-ai/theme",
    "nuttea/Softblue",
    "ParityError/Anime",
    "reilnuud/polite",
    "remilia/Ghostly",
    "rottenlittlecreature/Moon_Goblin",
    "step-3-profit/Midnight-Deep",
    "Taithrah/Minimal",
    "ysharma/huggingface",
    "ysharma/steampunk",
    "NoCrypt/miku"
]

# 重新加载 Gradio 主题
def reload_gradio_theme(theme_name=None):
    # 如果未指定主题名称，则使用共享选项中的 Gradio 主题
    if not theme_name:
        theme_name = shared.opts.gradio_theme

    # 默认主题参数
    default_theme_args = dict(
        font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
        font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
    )

    # 如果主题名称为 "Default"，则设置共享 Gradio 主题为默认主题
    if theme_name == "Default":
        shared.gradio_theme = gr.themes.Default(**default_theme_args)
    # 如果不是默认主题，则尝试加载或创建新的主题
    else:
        # 定义主题缓存目录和主题缓存路径
        theme_cache_dir = os.path.join(script_path, 'tmp', 'gradio_themes')
        theme_cache_path = os.path.join(theme_cache_dir, f'{theme_name.replace("/", "_")}.json')
        # 如果启用了主题缓存且主题缓存文件存在，则加载缓存中的主题
        if shared.opts.gradio_themes_cache and os.path.exists(theme_cache_path):
            shared.gradio_theme = gr.themes.ThemeClass.load(theme_cache_path)
        else:
            # 否则创建主题缓存目录，加载主题并保存到缓存文件中
            os.makedirs(theme_cache_dir, exist_ok=True)
            shared.gradio_theme = gr.themes.ThemeClass.from_hub(theme_name)
            shared.gradio_theme.dump(theme_cache_path)
        # 捕获异常，显示错误信息并设置默认主题
        except Exception as e:
            errors.display(e, "changing gradio theme")
            shared.gradio_theme = gr.themes.Default(**default_theme_args)
```