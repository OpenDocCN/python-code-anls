# `stable-diffusion-webui\modules\ui_gradio_extensions.py`

```py
# 导入所需的模块
import os
import gradio as gr

# 导入自定义模块
from modules import localization, shared, scripts
from modules.paths import script_path, data_path, cwd

# 根据文件路径生成 web 路径
def webpath(fn):
    # 如果文件路径以当前工作目录开头，则生成相对路径
    if fn.startswith(cwd):
        web_path = os.path.relpath(fn, cwd)
    # 否则生成绝对路径
    else:
        web_path = os.path.abspath(fn)

    return f'file={web_path}?{os.path.getmtime(fn)}'

# 生成包含 JavaScript 的 HTML 头部
def javascript_html():
    # 确保在脚本之前将本地化信息添加到 `window` 对象中
    head = f'<script type="text/javascript">{localization.localization_js(shared.opts.localization)}</script>\n'

    # 添加主 JavaScript 脚本
    script_js = os.path.join(script_path, "script.js")
    head += f'<script type="text/javascript" src="{webpath(script_js)}"></script>\n'

    # 添加所有 JavaScript 脚本文件
    for script in scripts.list_scripts("javascript", ".js"):
        head += f'<script type="text/javascript" src="{webpath(script.path)}"></script>\n'

    # 添加所有模块化 JavaScript 脚本文件
    for script in scripts.list_scripts("javascript", ".mjs"):
        head += f'<script type="module" src="{webpath(script.path)}"></script>\n'

    # 如果有主题选项，则设置主题
    if shared.cmd_opts.theme:
        head += f'<script type="text/javascript">set_theme(\"{shared.cmd_opts.theme}\");</script>\n'

    return head

# 生成包含 CSS 的 HTML 头部
def css_html():
    head = ""

    # 生成样式表链接
    def stylesheet(fn):
        return f'<link rel="stylesheet" property="stylesheet" href="{webpath(fn)}">'

    # 添加所有名为 "style.css" 的 CSS 文件
    for cssfile in scripts.list_files_with_name("style.css"):
        if not os.path.isfile(cssfile):
            continue
        head += stylesheet(cssfile)

    # 如果存在用户自定义 CSS 文件，则添加
    if os.path.exists(os.path.join(data_path, "user.css")):
        head += stylesheet(os.path.join(data_path, "user.css"))

    return head

# 重新加载 JavaScript 和 CSS
def reload_javascript():
    # 生成包含 JavaScript 的 HTML 头部和包含 CSS 的 HTML 头部
    js = javascript_html()
    css = css_html()

    # 替换响应中的头部信息
    def template_response(*args, **kwargs):
        res = shared.GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{js}</head>'.encode("utf8"))
        res.body = res.body.replace(b'</body>', f'{css}</body>'.encode("utf8"))
        res.init_headers()
        return res
    # 将template_response赋值给gr.routes.templates.TemplateResponse，可能是为了在后续代码中使用template_response
    gr.routes.templates.TemplateResponse = template_response
# 如果 shared 模块中没有属性 'GradioTemplateResponseOriginal'，则执行以下代码
if not hasattr(shared, 'GradioTemplateResponseOriginal'):
    # 将 gr.routes.templates.TemplateResponse 赋值给 shared 模块的 'GradioTemplateResponseOriginal' 属性
    shared.GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse
```