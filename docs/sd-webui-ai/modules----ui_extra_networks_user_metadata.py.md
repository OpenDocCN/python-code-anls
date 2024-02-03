# `stable-diffusion-webui\modules\ui_extra_networks_user_metadata.py`

```py
# 导入必要的模块
import datetime
import html
import json
import os.path

# 导入 gradio 模块并重命名为 gr
import gradio as gr

# 从 modules 模块中导入指定的子模块
from modules import generation_parameters_copypaste, images, sysinfo, errors, ui_extra_networks

# 定义用户元数据编辑器类
class UserMetadataEditor:

    # 初始化方法，接受 UI、标签名和页面作为参数
    def __init__(self, ui, tabname, page):
        # 初始化 UI、标签名和页面属性
        self.ui = ui
        self.tabname = tabname
        self.page = page
        self.id_part = f"{self.tabname}_{self.page.id_page}_edit_user_metadata"

        # 初始化各个编辑器元素为 None
        self.box = None
        self.edit_name_input = None
        self.button_edit = None
        self.edit_name = None
        self.edit_description = None
        self.edit_notes = None
        self.html_filedata = None
        self.html_preview = None
        self.html_status = None
        self.button_cancel = None
        self.button_replace_preview = None
        self.button_save = None

    # 获取指定名称的用户元数据
    def get_user_metadata(self, name):
        # 获取页面中指定名称的项目
        item = self.page.items.get(name, {})

        # 获取项目中的用户元数据，如果不存在则创建一个空的用户元数据
        user_metadata = item.get('user_metadata', None)
        if not user_metadata:
            user_metadata = {'description': item.get('description', '')}
            item['user_metadata'] = user_metadata

        return user_metadata

    # 创建左列中额外默认项目
    def create_extra_default_items_in_left_column(self):
        pass

    # 创建默认的编辑器元素
    def create_default_editor_elems(self):
        # 创建一个行元素
        with gr.Row():
            # 创建一个列元素，比例为 2
            with gr.Column(scale=2):
                # 创建 HTML 元素用于显示额外网络名称
                self.edit_name = gr.HTML(elem_classes="extra-network-name")
                # 创建文本框元素用于编辑描述，行数为 4
                self.edit_description = gr.Textbox(label="Description", lines=4)
                # 创建 HTML 元素用于显示文件数据

                self.html_filedata = gr.HTML()

                # 调用创建左列中额外默认项目的方法
                self.create_extra_default_items_in_left_column()

            # 创建一个列元素，比例为 1，最小宽度为 0
            with gr.Column(scale=1, min_width=0):
                # 创建 HTML 元素用于显示预览
                self.html_preview = gr.HTML()
    # 创建默认的按钮组
    def create_default_buttons(self):
        # 在编辑用户元数据按钮组中创建一行
        with gr.Row(elem_classes="edit-user-metadata-buttons"):
            # 创建取消按钮
            self.button_cancel = gr.Button('Cancel')
            # 创建替换预览按钮
            self.button_replace_preview = gr.Button('Replace preview', variant='primary')
            # 创建保存按钮
            self.button_save = gr.Button('Save', variant='primary')

        # 创建用于显示状态信息的 HTML 元素
        self.html_status = gr.HTML(elem_classes="edit-user-metadata-status")

        # 设置取消按钮的点击事件为关闭弹出窗口
        self.button_cancel.click(fn=None, _js="closePopup")

    # 获取卡片的 HTML 内容
    def get_card_html(self, name):
        # 获取指定名称的项目信息
        item = self.page.items.get(name, {})

        # 获取项目的预览 URL
        preview_url = item.get("preview", None)

        # 如果没有预览 URL，则根据文件名查找预览 URL
        if not preview_url:
            filename, _ = os.path.splitext(item["filename"])
            preview_url = self.page.find_preview(filename)
            item["preview"] = preview_url

        # 如果存在预览 URL，则创建包含预览图片的 HTML 内容
        if preview_url:
            preview = f'''
            <div class='card standalone-card-preview'>
                <img src="{html.escape(preview_url)}" class="preview">
            </div>
            '''
        # 如果不存在预览 URL，则创建空的预览卡片
        else:
            preview = "<div class='card standalone-card-preview'></div>"

        return preview

    # 获取相对路径
    def relative_path(self, path):
        # 遍历允许预览的父目录列表
        for parent_path in self.page.allowed_directories_for_previews():
            # 如果路径是父目录的子目录，则返回相对路径
            if ui_extra_networks.path_is_parent(parent_path, path):
                return os.path.relpath(path, parent_path)

        # 如果路径不在允许的父目录中，则返回路径的基本名称
        return os.path.basename(path)
    # 获取指定名称的元数据表格
    def get_metadata_table(self, name):
        # 获取页面中指定名称的项目，如果不存在则返回空字典
        item = self.page.items.get(name, {})
        try:
            # 获取项目中的文件名和短哈希值
            filename = item["filename"]
            shorthash = item.get("shorthash", None)

            # 获取文件的统计信息
            stats = os.stat(filename)
            # 构建参数列表，包括文件名、文件大小、哈希值和修改时间
            params = [
                ('Filename: ', self.relative_path(filename)),
                ('File size: ', sysinfo.pretty_bytes(stats.st_size)),
                ('Hash: ', shorthash),
                ('Modified: ', datetime.datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M')),
            ]

            # 返回参数列表
            return params
        except Exception as e:
            # 如果出现异常，显示错误信息并返回空列表
            errors.display(e, f"reading info for {name}")
            return []

    # 将数值放入组件中
    def put_values_into_components(self, name):
        # 获取用户元数据
        user_metadata = self.get_user_metadata(name)

        try:
            # 获取元数据表格
            params = self.get_metadata_table(name)
        except Exception as e:
            # 如果出现异常，显示错误信息并将参数列表设为空列表
            errors.display(e, f"reading metadata info for {name}")
            params = []

        # 构建表格的 HTML 代码
        table = '<table class="file-metadata">' + "".join(f"<tr><th>{name}</th><td>{value}</td></tr>" for name, value in params if value is not None) + '</table>'

        # 返回转义后的名称、用户描述、表格、卡片 HTML 和用户注释
        return html.escape(name), user_metadata.get('description', ''), table, self.get_card_html(name), user_metadata.get('notes', '')

    # 写入用户元数据
    def write_user_metadata(self, name, metadata):
        # 获取页面中指定名称的项目，如果不存在则返回空字典
        item = self.page.items.get(name, {})
        # 获取文件名，如果不存在则设为 None
        filename = item.get("filename", None)
        # 获取文件名的基本名称和扩展名
        basename, ext = os.path.splitext(filename)

        # 打开 JSON 文件并写入元数据
        with open(basename + '.json', "w", encoding="utf8") as file:
            json.dump(metadata, file, indent=4, ensure_ascii=False)

    # 保存用户元数据
    def save_user_metadata(self, name, desc, notes):
        # 获取用户元数据
        user_metadata = self.get_user_metadata(name)
        # 更新描述和注释信息
        user_metadata["description"] = desc
        user_metadata["notes"] = notes

        # 写入用户元数据
        self.write_user_metadata(name, user_metadata)
    # 设置保存处理程序，当按钮被点击时执行指定函数，传入参数和组件，不返回任何输出
    def setup_save_handler(self, button, func, components):
        # 点击按钮时执行函数，传入参数和组件，不返回任何输出
        button\
            .click(fn=func, inputs=[self.edit_name_input, *components], outputs=[])\
            # 然后执行指定的 JavaScript 函数，传入参数，不返回任何输出
            .then(fn=None, _js="function(name){closePopup(); extraNetworksRefreshSingleCard(" + json.dumps(self.page.name) + "," + json.dumps(self.tabname) + ", name);}", inputs=[self.edit_name_input], outputs=[])

    # 创建编辑器
    def create_editor(self):
        # 创建默认的编辑器元素
        self.create_default_editor_elems()

        # 创建文本区域用于编辑笔记
        self.edit_notes = gr.TextArea(label='Notes', lines=4)

        # 创建默认按钮
        self.create_default_buttons()

        # 点击编辑按钮时执行函数，传入参数和组件，返回输出
        self.button_edit\
            .click(fn=self.put_values_into_components, inputs=[self.edit_name_input], outputs=[self.edit_name, self.edit_description, self.html_filedata, self.html_preview, self.edit_notes])\
            # 然后执行匿名函数，更新可见性，不传入参数，返回输出
            .then(fn=lambda: gr.update(visible=True), inputs=[], outputs=[self.box])

        # 设置保存处理程序，当保存按钮被点击时执行保存用户元数据函数，传入描述和笔记组件
        self.setup_save_handler(self.button_save, self.save_user_metadata, [self.edit_description, self.edit_notes])

    # 创建用户界面
    def create_ui(self):
        # 创建一个不可见的 Box 元素，用于编辑用户元数据
        with gr.Box(visible=False, elem_id=self.id_part, elem_classes="edit-user-metadata") as box:
            self.box = box

            # 创建文本框用于编辑用户元数据卡片 ID，不可见
            self.edit_name_input = gr.Textbox("Edit user metadata card id", visible=False, elem_id=f"{self.id_part}_name")
            # 创建按钮用于编辑用户元数据，不可见
            self.button_edit = gr.Button("Edit user metadata", visible=False, elem_id=f"{self.id_part}_button")

            # 创建编辑器
            self.create_editor()
    # 保存预览图片的方法，根据索引和相册名称
    def save_preview(self, index, gallery, name):
        # 如果相册中没有图片，则返回卡片的 HTML 和提示信息
        if len(gallery) == 0:
            return self.get_card_html(name), "There is no image in gallery to save as a preview."

        # 获取页面中指定名称的项目信息，如果不存在则为空字典
        item = self.page.items.get(name, {})

        # 将索引转换为整数类型
        index = int(index)
        # 如果索引小于0，则将索引设为0
        index = 0 if index < 0 else index
        # 如果索引大于等于相册长度，则将索引设为相册长度减1
        index = len(gallery) - 1 if index >= len(gallery) else index

        # 获取指定索引处的图片信息
        img_info = gallery[index if index >= 0 else 0]
        # 从图片信息中获取图片对象
        image = generation_parameters_copypaste.image_from_url_text(img_info)
        # 从图片中读取生成信息和项目信息
        geninfo, items = images.read_info_from_image(image)

        # 保存带有生成信息的图片到本地预览
        images.save_image_with_geninfo(image, geninfo, item["local_preview"])

        # 返回卡片的 HTML 和空字符串
        return self.get_card_html(name), ''

    # 设置用户界面的方法，传入相册参数
    def setup_ui(self, gallery):
        # 点击替换预览按钮，调用保存预览方法
        self.button_replace_preview.click(
            fn=self.save_preview,
            # 定义 JavaScript 函数，返回选定的相册索引、y 和 z
            _js="function(x, y, z){return [selected_gallery_index(), y, z]}",
            # 输入参数为编辑名称输入框、相册和编辑名称输入框
            inputs=[self.edit_name_input, gallery, self.edit_name_input],
            # 输出为预览的 HTML 和状态信息
            outputs=[self.html_preview, self.html_status]
        ).then(
            fn=None,
            # 定义 JavaScript 函数，刷新单个卡片的额外网络
            _js="function(name){extraNetworksRefreshSingleCard(" + json.dumps(self.page.name) + "," + json.dumps(self.tabname) + ", name);}",
            # 输入参数为编辑名称输入框
            inputs=[self.edit_name_input],
            # 输出为空
            outputs=[]
        )
```