# `stable-diffusion-webui\modules\ui_extra_networks.py`

```
# 导入必要的模块
import functools
import os.path
import urllib.parse
from pathlib import Path

from modules import shared, ui_extra_networks_user_metadata, errors, extra_networks
from modules.images import read_info_from_image, save_image_with_geninfo
import gradio as gr
import json
import html
from fastapi.exceptions import HTTPException

# 定义额外页面列表和允许的目录集合
extra_pages = []
allowed_dirs = set()

# 默认允许预览的文件扩展名
default_allowed_preview_extensions = ["png", "jpg", "jpeg", "webp", "gif"]

# 使用缓存装饰器定义允许的预览文件扩展名（包括额外的扩展名）
@functools.cache
def allowed_preview_extensions_with_extra(extra_extensions=None):
    return set(default_allowed_preview_extensions) | set(extra_extensions or [])

# 获取允许的预览文件扩展名
def allowed_preview_extensions():
    return allowed_preview_extensions_with_extra((shared.opts.samples_format, ))

# 注册页面到额外网络页面列表
def register_page(page):
    """registers extra networks page for the UI; recommend doing it in on_before_ui() callback for extensions"""
    extra_pages.append(page)
    allowed_dirs.clear()
    allowed_dirs.update(set(sum([x.allowed_directories_for_previews() for x in extra_pages], [])))

# 获取文件并返回文件响应
def fetch_file(filename: str = ""):
    from starlette.responses import FileResponse

    # 如果文件不存在，则抛出404错误
    if not os.path.isfile(filename):
        raise HTTPException(status_code=404, detail="File not found")

    # 如果文件不在允许的目录中，则抛出值错误
    if not any(Path(x).absolute() in Path(filename).absolute().parents for x in allowed_dirs):
        raise ValueError(f"File cannot be fetched: {filename}. Must be in one of directories registered by extra pages.")

    # 获取文件扩展名并检查是否在允许的预览文件扩展名中
    ext = os.path.splitext(filename)[1].lower()[1:]
    if ext not in allowed_preview_extensions():
        raise ValueError(f"File cannot be fetched: {filename}. Extensions allowed: {allowed_preview_extensions()}.")

    # 返回文件响应
    # 可以从返回头部中返回304状态码
    return FileResponse(filename, headers={"Accept-Ranges": "bytes"})

# 获取元数据并返回JSON响应
def get_metadata(page: str = "", item: str = ""):
    from starlette.responses import JSONResponse
    # 从额外页面列表中找到指定页面对象
    page = next(iter([x for x in extra_pages if x.name == page]), None)
    # 如果找不到指定页面对象，则返回空 JSON 响应
    if page is None:
        return JSONResponse({})

    # 获取指定页面对象的元数据
    metadata = page.metadata.get(item)
    # 如果元数据为空，则返回空 JSON 响应
    if metadata is None:
        return JSONResponse({})

    # 返回包含元数据的 JSON 响应，使用 json.dumps 方法将元数据转换为 JSON 格式
    return JSONResponse({"metadata": json.dumps(metadata, indent=4, ensure_ascii=False)})
# 定义一个函数，用于获取单个卡片的信息
def get_single_card(page: str = "", tabname: str = "", name: str = ""):
    # 导入 JSONResponse 类
    from starlette.responses import JSONResponse

    # 从额外页面列表中找到指定名称的页面
    page = next(iter([x for x in extra_pages if x.name == page]), None)

    try:
        # 创建指定名称的项目，禁用过滤器
        item = page.create_item(name, enable_filter=False)
        # 将项目添加到页面的项目字典中
        page.items[name] = item
    except Exception as e:
        # 显示创建额外网络项目时的错误信息
        errors.display(e, "creating item for extra network")
        # 获取已存在的项目
        item = page.items.get(name)

    # 读取用户元数据
    page.read_user_metadata(item)
    # 为项目创建 HTML 内容
    item_html = page.create_html_for_item(item, tabname)

    # 返回 JSONResponse 对象，包含 HTML 内容
    return JSONResponse({"html": item_html})


# 将页面添加到演示应用中
def add_pages_to_demo(app):
    # 添加获取文件的 API 路由
    app.add_api_route("/sd_extra_networks/thumb", fetch_file, methods=["GET"])
    # 添加获取元数据的 API 路由
    app.add_api_route("/sd_extra_networks/metadata", get_metadata, methods=["GET"])
    # 添加获取单个卡片信息的 API 路由
    app.add_api_route("/sd_extra_networks/get-single-card", get_single_card, methods=["GET"])


# 对 JavaScript 字符串进行转义处理
def quote_js(s):
    # 替换反斜杠为双反斜杠
    s = s.replace('\\', '\\\\')
    # 替换单引号为双引号
    s = s.replace('"', '\\"')
    return f'"{s}"'


# 定义额外网络页面类
class ExtraNetworksPage:
    def __init__(self, title):
        # 初始化页面标题、名称、ID、卡片页面、提示设置等属性
        self.title = title
        self.name = title.lower()
        self.id_page = self.name.replace(" ", "_")
        self.card_page = shared.html("extra-networks-card.html")
        self.allow_prompt = True
        self.allow_negative_prompt = False
        self.metadata = {}
        self.items = {}

    # 刷新页面
    def refresh(self):
        pass

    # 读取用户元数据
    def read_user_metadata(self, item):
        # 获取项目的文件名
        filename = item.get("filename", None)
        # 获取用户元数据
        metadata = extra_networks.get_user_metadata(filename)

        # 获取描述信息并添加到项目中
        desc = metadata.get("description", None)
        if desc is not None:
            item["description"] = desc

        # 将用户元数据添加到项目中
        item["user_metadata"] = metadata

    # 生成文件链接预览
    def link_preview(self, filename):
        # 对文件名进行 URL 编码
        quoted_filename = urllib.parse.quote(filename.replace('\\', '/'))
        # 获取文件的修改时间
        mtime = os.path.getmtime(filename)
        return f"./sd_extra_networks/thumb?filename={quoted_filename}&mtime={mtime}"
    # 从给定的文件路径中提取搜索关键词
    def search_terms_from_path(self, filename, possible_directories=None):
        # 获取文件的绝对路径
        abspath = os.path.abspath(filename)

        # 遍历可能的目录，查找文件所在的目录
        for parentdir in (possible_directories if possible_directories is not None else self.allowed_directories_for_previews()):
            parentdir = os.path.abspath(parentdir)
            # 如果文件路径以目录路径开头，则返回文件相对于目录的路径
            if abspath.startswith(parentdir):
                return abspath[len(parentdir):].replace('\\', '/')

        # 如果没有找到匹配的目录，则返回空字符串
        return ""

    # 创建 HTML 内容
    def create_html(self, tabname):
        # 初始化 HTML 内容
        items_html = ''

        # 初始化元数据字典
        self.metadata = {}

        # 初始化子目录字典
        subdirs = {}
        # 遍历允许预览的目录
        for parentdir in [os.path.abspath(x) for x in self.allowed_directories_for_previews()]:
            # 遍历目录及其子目录
            for root, dirs, _ in sorted(os.walk(parentdir, followlinks=True), key=lambda x: shared.natural_sort_key(x[0])):
                # 遍历子目录
                for dirname in sorted(dirs, key=shared.natural_sort_key):
                    x = os.path.join(root, dirname)

                    # 如果不是目录，则继续下一个循环
                    if not os.path.isdir(x):
                        continue

                    # 获取子目录相对于父目录的路径
                    subdir = os.path.abspath(x)[len(parentdir):].replace("\\", "/")

                    # 处理额外的网络目录按钮功能
                    if shared.opts.extra_networks_dir_button_function:
                        if not subdir.startswith("/"):
                            subdir = "/" + subdir
                    else:
                        while subdir.startswith("/"):
                            subdir = subdir[1:]

                    # 检查目录是否为空
                    is_empty = len(os.listdir(x)) == 0
                    # 如果目录不为空且不以斜杠结尾，则添加斜杠
                    if not is_empty and not subdir.endswith("/"):
                        subdir = subdir + "/"

                    # 如果目录包含隐藏文件或以点开头，并且不显示隐藏目录，则继续下一个循环
                    if ("/." in subdir or subdir.startswith(".")) and not shared.opts.extra_networks_show_hidden_directories:
                        continue

                    # 将子目录添加到子目录字典中
                    subdirs[subdir] = 1

        # 如果子目录字典不为空，则添加空字符串键值对
        if subdirs:
            subdirs = {"": 1, **subdirs}

        # 将子目录 HTML 内容拼接成字符串
        subdirs_html = "".join([f"""
# 创建一个按钮元素，根据子目录是否为空添加不同的样式类，点击按钮时调用extraNetworksSearchButton函数
<button class='lg secondary gradio-button custom-button{" search-all" if subdir=="" else ""}' onclick='extraNetworksSearchButton("{tabname}_extra_search", event)'>
# 将子目录转义并插入到按钮元素中
{html.escape(subdir if subdir!="" else "all")}
</button>

# 遍历子目录列表，为每个子目录创建按钮元素
""" for subdir in subdirs])

# 创建一个字典，键为item的name属性，值为item对象本身
self.items = {x["name"]: x for x in self.list_items()}

# 遍历items字典中的每个item对象
for item in self.items.values():
    # 获取item对象的metadata属性
    metadata = item.get("metadata")
    if metadata:
        # 将metadata添加到self.metadata字典中
        self.metadata[item["name"]] = metadata

    # 如果item对象中没有user_metadata属性，则调用read_user_metadata方法
    if "user_metadata" not in item:
        self.read_user_metadata(item)

    # 为item对象创建HTML元素，并添加到items_html中
    items_html += self.create_html_for_item(item, tabname)

# 如果items_html为空，根据allowed_directories_for_previews方法返回的目录列表创建HTML元素
if items_html == '':
    dirs = "".join([f"<li>{x}</li>" for x in self.allowed_directories_for_previews()])
    items_html = shared.html("extra-networks-no-cards.html").format(dirs=dirs)

# 将self.name中的空格替换为下划线
self_name_id = self.name.replace(" ", "_")

# 构建包含子目录和卡片的HTML元素
res = f"""
<div id='{tabname}_{self_name_id}_subdirs' class='extra-network-subdirs extra-network-subdirs-cards'>
{subdirs_html}
</div>
<div id='{tabname}_{self_name_id}_cards' class='extra-network-cards'>
{items_html}
</div>
"""

# 返回HTML元素字符串
return res

# 创建一个名为name的item对象，可选参数index
def create_item(self, name, index=None):
    raise NotImplementedError()

# 列出所有item对象，需要在子类中实现
def list_items(self):
    raise NotImplementedError()

# 返回用于预览的目录列表，默认为空列表
def allowed_directories_for_previews(self):
    return []

# 获取用于UI排序的默认键列表
def get_sort_keys(self, path):
    """
    List of default keys used for sorting in the UI.
    """
    # 获取路径的Path对象
    pth = Path(path)
    # 获取路径的stat信息
    stat = pth.stat()
    # 返回包含日期创建、日期修改、名称和路径的字典
    return {
        "date_created": int(stat.st_ctime or 0),
        "date_modified": int(stat.st_mtime or 0),
        "name": pth.name.lower(),
        "path": str(pth.parent).lower(),
    }
    # 为给定路径（不包含扩展名）查找预览 PNG 文件，并调用 link_preview 方法
    def find_preview(self, path):
        """
        Find a preview PNG for a given path (without extension) and call link_preview on it.
        """

        # 生成可能的预览文件列表，包括带有不同扩展名的文件
        potential_files = sum([[path + "." + ext, path + ".preview." + ext] for ext in allowed_preview_extensions()], [])

        # 遍历可能的文件列表，如果找到文件则调用 link_preview 方法
        for file in potential_files:
            if os.path.isfile(file):
                return self.link_preview(file)

        # 如果未找到文件，则返回 None
        return None

    # 为给定路径（不包含扩展名）查找并读取描述文件
    def find_description(self, path):
        """
        Find and read a description file for a given path (without extension).
        """
        # 遍历可能的描述文件列表，尝试打开文件并读取内容
        for file in [f"{path}.txt", f"{path}.description.txt"]:
            try:
                with open(file, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
            except OSError:
                pass
        # 如果未找到文件或读取失败，则返回 None
        return None

    # 创建用户元数据编辑器
    def create_user_metadata_editor(self, ui, tabname):
        return ui_extra_networks_user_metadata.UserMetadataEditor(ui, tabname, self)
# 清空额外页面列表
def initialize():
    extra_pages.clear()


# 注册默认页面
def register_default_pages():
    # 导入额外网络文本反转页面类
    from modules.ui_extra_networks_textual_inversion import ExtraNetworksPageTextualInversion
    # 导入额外网络超网络页面类
    from modules.ui_extra_networks_hypernets import ExtraNetworksPageHypernetworks
    # 导入额外网络检查点页面类
    from modules.ui_extra_networks_checkpoints import ExtraNetworksPageCheckpoints
    # 注册额外网络页面
    register_page(ExtraNetworksPageTextualInversion())
    register_page(ExtraNetworksPageHypernetworks())
    register_page(ExtraNetworksPageCheckpoints())


# 额外网络用户界面类
class ExtraNetworksUi:
    def __init__(self):
        self.pages = None
        """gradio HTML components related to extra networks' pages"""

        self.page_contents = None
        """HTML content of the above; empty initially, filled when extra pages have to be shown"""

        self.stored_extra_pages = None

        self.button_save_preview = None
        self.preview_target_filename = None

        self.tabname = None


# 根据首选顺序对页面进行排序
def pages_in_preferred_order(pages):
    # 获取首选标签顺序列表
    tab_order = [x.lower().strip() for x in shared.opts.ui_extra_networks_tab_reorder.split(",")]

    # 计算标签名称得分
    def tab_name_score(name):
        name = name.lower()
        for i, possible_match in enumerate(tab_order):
            if possible_match in name:
                return i

        return len(pages)

    # 计算每个页面的得分
    tab_scores = {page.name: (tab_name_score(page.name), original_index) for original_index, page in enumerate(pages)}

    # 根据得分对页面进行排序
    return sorted(pages, key=lambda x: tab_scores[x.name])


# 创建用户界面
def create_ui(interface: gr.Blocks, unrelated_tabs, tabname):
    # 导入切换值符号模块
    from modules.ui import switch_values_symbol

    # 创建额外网络用户界面对象
    ui = ExtraNetworksUi()
    ui.pages = []
    ui.pages_contents = []
    ui.user_metadata_editors = []
    # 复制并按首选顺序排序额外页面
    ui.stored_extra_pages = pages_in_preferred_order(extra_pages.copy())
    ui.tabname = tabname

    related_tabs = []
    # 遍历存储的额外页面列表
    for page in ui.stored_extra_pages:
        # 创建一个选项卡，标题为页面的标题，ID为tabname_page.id_page，类名为extra-page
        with gr.Tab(page.title, elem_id=f"{tabname}_{page.id_page}", elem_classes=["extra-page"]) as tab:
            # 在选项卡中创建一个列，ID为tabname_page.id_page_prompts，类名为extra-page-prompts
            with gr.Column(elem_id=f"{tabname}_{page.id_page}_prompts", elem_classes=["extra-page-prompts"]):
                # 空操作，暂时没有内容

            # 创建一个HTML元素，显示'Loading...'，ID为tabname_page.id_page_cards_html
            elem_id = f"{tabname}_{page.id_page}_cards_html"
            page_elem = gr.HTML('Loading...', elem_id=elem_id)
            # 将HTML元素添加到UI的页面列表中
            ui.pages.append(page_elem)

            # 为页面元素添加change事件，调用applyExtraNetworkFilter函数，传入tabname参数
            page_elem.change(fn=lambda: None, _js='function(){applyExtraNetworkFilter(' + quote_js(tabname) + '); return []}', inputs=[], outputs=[])

            # 创建用户元数据编辑器
            editor = page.create_user_metadata_editor(ui, tabname)
            # 创建用户界面
            editor.create_ui()
            # 将编辑器添加到UI的用户元数据编辑器列表中
            ui.user_metadata_editors.append(editor)

            # 将选项卡添加到相关选项卡列表中
            related_tabs.append(tab)

    # 创建一个文本框，初始值为空，不显示标签，ID为tabname_extra_search，类名为search，占位符为'Search...'，初始不可见，可交互
    edit_search = gr.Textbox('', show_label=False, elem_id=tabname+"_extra_search", elem_classes="search", placeholder="Search...", visible=False, interactive=True)
    # 创建一个下拉框，选项为['Path', 'Name', 'Date Created', 'Date Modified']，初始值为shared.opts.extra_networks_card_order_field，ID为tabname_extra_sort，类名为sort，不可见，不显示标签，可交互，标签为tabname_extra_sort_order
    dropdown_sort = gr.Dropdown(choices=['Path', 'Name', 'Date Created', 'Date Modified', ], value=shared.opts.extra_networks_card_order_field, elem_id=tabname+"_extra_sort", elem_classes="sort", multiselect=False, visible=False, show_label=False, interactive=True, label=tabname+"_extra_sort_order")
    # 创建一个工具按钮，初始不可见，提示为'Invert sort order'
    button_sortorder = ToolButton(switch_values_symbol, elem_id=tabname+"_extra_sortorder", elem_classes=["sortorder"] + ([] if shared.opts.extra_networks_card_order == "Ascending" else ["sortReverse"]), visible=False, tooltip="Invert sort order")
    # 创建一个按钮，显示'Refresh'，初始不可见
    button_refresh = gr.Button('Refresh', elem_id=tabname+"_extra_refresh", visible=False)
    # 创建一个复选框，初始勾选，标签为'Show dirs'，ID为tabname_extra_show_dirs，类名为show-dirs，初始不可见
    checkbox_show_dirs = gr.Checkbox(True, label='Show dirs', elem_id=tabname+"_extra_show_dirs", elem_classes="show-dirs", visible=False)

    # 将UI的按钮保存预览设置为一个按钮，显示'Save preview'，ID为tabname_save_preview，初始不可见
    ui.button_save_preview = gr.Button('Save preview', elem_id=tabname+"_save_preview", visible=False)
    # 将UI的预览目标文件名设置为一个文本框，初始值为'Preview save filename'，ID为tabname_preview_filename，初始不可见
    ui.preview_target_filename = gr.Textbox('Preview save filename', elem_id=tabname+"_preview_filename", visible=False)
    # 定义一个包含多个控件的列表
    tab_controls = [edit_search, dropdown_sort, button_sortorder, button_refresh, checkbox_show_dirs]

    # 遍历未关联的选项卡
    for tab in unrelated_tabs:
        # 选择选项卡，并执行指定的函数和 JavaScript 代码
        tab.select(fn=lambda: [gr.update(visible=False) for _ in tab_controls], _js='function(){ extraNetworksUrelatedTabSelected("' + tabname + '"); }', inputs=[], outputs=tab_controls, show_progress=False)

    # 遍历存储的额外页面和相关的选项卡
    for page, tab in zip(ui.stored_extra_pages, related_tabs):
        # 根据页面的属性设置允许提示和允许负面提示的值
        allow_prompt = "true" if page.allow_prompt else "false"
        allow_negative_prompt = "true" if page.allow_negative_prompt else "false"

        # 构建 JavaScript 代码
        jscode = 'extraNetworksTabSelected("' + tabname + '", "' + f"{tabname}_{page.id_page}_prompts" + '", ' + allow_prompt + ', ' + allow_negative_prompt + ');'

        # 选择选项卡，并执行指定的函数和 JavaScript 代码
        tab.select(fn=lambda: [gr.update(visible=True) for _ in tab_controls],  _js='function(){ ' + jscode + ' }', inputs=[], outputs=tab_controls, show_progress=False)

    # 更改下拉排序控件时执行的函数和 JavaScript 代码
    dropdown_sort.change(fn=lambda: None, _js="function(){ applyExtraNetworkSort('" + tabname + "'); }")

    # 定义一个返回页面 HTML 内容的函数
    def pages_html():
        # 如果页面内容为空，则刷新页面
        if not ui.pages_contents:
            return refresh()

        return ui.pages_contents

    # 刷新页面内容的函数
    def refresh():
        # 刷新存储的额外页面
        for pg in ui.stored_extra_pages:
            pg.refresh()

        # 更新页面内容列表
        ui.pages_contents = [pg.create_html(ui.tabname) for pg in ui.stored_extra_pages]

        return ui.pages_contents

    # 加载页面内容，并指定输入和输出
    interface.load(fn=pages_html, inputs=[], outputs=[*ui.pages])
    # 点击刷新按钮时执行刷新函数
    button_refresh.click(fn=refresh, inputs=[], outputs=ui.pages)

    # 返回用户界面对象
    return ui
# 检查给定的 parent_path 是否是 child_path 的父路径
def path_is_parent(parent_path, child_path):
    # 获取 parent_path 的绝对路径
    parent_path = os.path.abspath(parent_path)
    # 获取 child_path 的绝对路径
    child_path = os.path.abspath(child_path)

    # 判断 child_path 是否以 parent_path 开头
    return child_path.startswith(parent_path)


# 设置用户界面
def setup_ui(ui, gallery):
    # 保存预览图片的函数，用于向后兼容，可能会被删除
    def save_preview(index, images, filename):
        if len(images) == 0:
            print("There is no image in gallery to save as a preview.")
            return [page.create_html(ui.tabname) for page in ui.stored_extra_pages]

        # 确保 index 是整数
        index = int(index)
        # 如果 index 小于 0，则设置为 0
        index = 0 if index < 0 else index
        # 如果 index 大于等于 images 的长度，则设置为最后一个索引
        index = len(images) - 1 if index >= len(images) else index

        # 获取指定索引的图片信息
        img_info = images[index if index >= 0 else 0]
        # 从图片信息中获取图片对象
        image = image_from_url_text(img_info)
        # 从图片中读取信息
        geninfo, items = read_info_from_image(image)

        # 检查是否允许保存到指定的文件名
        is_allowed = False
        for extra_page in ui.stored_extra_pages:
            if any(path_is_parent(x, filename) for x in extra_page.allowed_directories_for_previews()):
                is_allowed = True
                break

        # 断言是否允许保存到指定文件名
        assert is_allowed, f'writing to {filename} is not allowed'

        # 保存图片和信息到指定文件名
        save_image_with_geninfo(image, geninfo, filename)

        # 返回所有额外页面的 HTML
        return [page.create_html(ui.tabname) for page in ui.stored_extra_pages]

    # 点击按钮时触发保存预览函数
    ui.button_save_preview.click(
        fn=save_preview,
        _js="function(x, y, z){return [selected_gallery_index(), y, z]}",
        inputs=[ui.preview_target_filename, gallery, ui.preview_target_filename],
        outputs=[*ui.pages]
    )

    # 针对每个用户元数据编辑器设置用户界面
    for editor in ui.user_metadata_editors:
        editor.setup_ui(gallery)
```