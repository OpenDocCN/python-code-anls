# `stable-diffusion-webui\modules\ui_settings.py`

```
# 导入必要的库
import gradio as gr

# 从自定义模块中导入所需的函数和类
from modules import ui_common, shared, script_callbacks, scripts, sd_models, sysinfo, timer
from modules.call_queue import wrap_gradio_call
from modules.shared import opts
from modules.ui_components import FormRow
from modules.ui_gradio_extensions import reload_javascript
from concurrent.futures import ThreadPoolExecutor, as_completed

# 获取设置项的值
def get_value_for_setting(key):
    # 获取设置项的值
    value = getattr(opts, key)

    # 获取设置项的信息
    info = opts.data_labels[key]
    # 获取设置项的组件参数
    args = info.component_args() if callable(info.component_args) else info.component_args or {}
    # 过滤掉 precision 参数
    args = {k: v for k, v in args.items() if k not in {'precision'}}

    # 返回更新后的设置项值和参数
    return gr.update(value=value, **args)

# 创建设置项组件
def create_setting_component(key, is_quicksettings=False):
    # 获取设置项的默认值
    def fun():
        return opts.data[key] if key in opts.data else opts.data_labels[key].default

    # 获取设置项的信息
    info = opts.data_labels[key]
    # 获取设置项默认值的类型
    t = type(info.default)

    # 获取设置项的组件参数
    args = info.component_args() if callable(info.component_args) else info.component_args

    # 根据设置项的类型选择对应的组件
    if info.component is not None:
        comp = info.component
    elif t == str:
        comp = gr.Textbox
    elif t == int:
        comp = gr.Number
    elif t == bool:
        comp = gr.Checkbox
    else:
        raise Exception(f'bad options item type: {t} for key {key}')

    # 设置组件的 ID
    elem_id = f"setting_{key}"

    # 如果设置项需要刷新
    if info.refresh is not None:
        # 如果是快速设置
        if is_quicksettings:
            # 创建组件并添加刷新按钮
            res = comp(label=info.label, value=fun(), elem_id=elem_id, **(args or {}))
            ui_common.create_refresh_button(res, info.refresh, info.component_args, f"refresh_{key}")
        else:
            # 在表单行中创建组件并添加刷新按钮
            with FormRow():
                res = comp(label=info.label, value=fun(), elem_id=elem_id, **(args or {}))
                ui_common.create_refresh_button(res, info.refresh, info.component_args, f"refresh_{key}")
    else:
        # 创建组件
        res = comp(label=info.label, value=fun(), elem_id=elem_id, **(args or {}))

    return res

# 定义一个 UiSettings 类
class UiSettings:
    submit = None
    result = None
    interface = None
    components = None
    # 初始化变量，设置为 None
    component_dict = None
    dummy_component = None
    quicksettings_list = None
    quicksettings_names = None
    text_settings = None
    show_all_pages = None
    show_one_page = None
    search_input = None

    # 运行设置函数，接受一系列参数
    def run_settings(self, *args):
        # 存储已更改的设置
        changed = []

        # 遍历参数和组件，检查设置值是否符合预期类型
        for key, value, comp in zip(opts.data_labels.keys(), args, self.components):
            assert comp == self.dummy_component or opts.same_type(value, opts.data_labels[key].default), f"Bad value for setting {key}: {value}; expecting {type(opts.data_labels[key].default).__name__}"

        # 再次遍历参数和组件，设置新值并记录更改的设置
        for key, value, comp in zip(opts.data_labels.keys(), args, self.components):
            if comp == self.dummy_component:
                continue

            if opts.set(key, value):
                changed.append(key)

        # 尝试保存设置到配置文件，如果失败则返回更改的设置和未保存的消息
        try:
            opts.save(shared.config_filename)
        except RuntimeError:
            return opts.dumpjson(), f'{len(changed)} settings changed without save: {", ".join(changed)}.'
        # 返回更改的设置和保存成功的消息
        return opts.dumpjson(), f'{len(changed)} settings changed{": " if changed else ""}{", ".join(changed)}.'

    # 运行单个设置函数，接受值和键
    def run_settings_single(self, value, key):
        # 如果值类型不符合预期，则返回更新可见性和设置的 JSON 数据
        if not opts.same_type(value, opts.data_labels[key].default):
            return gr.update(visible=True), opts.dumpjson()

        # 如果值为空或设置失败，则返回更新值和设置的 JSON 数据
        if value is None or not opts.set(key, value):
            return gr.update(value=getattr(opts, key)), opts.dumpjson()

        # 保存设置到配置文件
        opts.save(shared.config_filename)

        # 返回设置键对应的值和设置的 JSON 数据
        return get_value_for_setting(key), opts.dumpjson()

    # 添加快速设置函数
    def add_quicksettings(self):
        # 创建一个行元素，用于显示快速设置
        with gr.Row(elem_id="quicksettings", variant="compact"):
            # 遍历快速设置列表，按名称排序，创建设置组件并添加到组件字典中
            for _i, k, _item in sorted(self.quicksettings_list, key=lambda x: self.quicksettings_names.get(x[1], x[0])):
                component = create_setting_component(k, is_quicksettings=True)
                self.component_dict[k] = component
    # 为界面添加功能性，包括点击事件、输入输出设置等
    def add_functionality(self, demo):
        # 点击事件，调用 run_settings 方法，并更新输出
        self.submit.click(
            fn=wrap_gradio_call(lambda *args: self.run_settings(*args), extra_outputs=[gr.update()]),
            inputs=self.components,
            outputs=[self.text_settings, self.result],
        )

        # 遍历快速设置列表
        for _i, k, _item in self.quicksettings_list:
            # 获取组件和信息
            component = self.component_dict[k]
            info = opts.data_labels[k]

            # 根据组件类型选择不同的方法
            if isinstance(component, gr.Textbox):
                methods = [component.submit, component.blur]
            elif hasattr(component, 'release'):
                methods = [component.release]
            else:
                methods = [component.change]

            # 遍历方法，设置事件处理函数
            for method in methods:
                method(
                    fn=lambda value, k=k: self.run_settings_single(value, key=k),
                    inputs=[component],
                    outputs=[component, self.text_settings],
                    show_progress=info.refresh is not None,
                )

        # 创建按钮用于更改检查点
        button_set_checkpoint = gr.Button('Change checkpoint', elem_id='change_checkpoint', visible=False)
        button_set_checkpoint.click(
            fn=lambda value, _: self.run_settings_single(value, key='sd_model_checkpoint'),
            _js="function(v){ var res = desiredCheckpointName; desiredCheckpointName = ''; return [res || v, null]; }",
            inputs=[self.component_dict['sd_model_checkpoint'], self.dummy_component],
            outputs=[self.component_dict['sd_model_checkpoint'], self.text_settings],
        )

        # 获取组件键列表
        component_keys = [k for k in opts.data_labels.keys() if k in self.component_dict]

        # 获取设置值的函数
        def get_settings_values():
            return [get_value_for_setting(key) for key in component_keys]

        # 加载设置值到界面
        demo.load(
            fn=get_settings_values,
            inputs=[],
            outputs=[self.component_dict[k] for k in component_keys],
            queue=False,
        )
    # 定义一个搜索方法，接收一个文本参数
    def search(self, text):
        # 打印搜索的文本内容
        print(text)

        # 遍历组件列表，更新每个组件的可见性属性，根据搜索文本是否在组件标签中出现来确定
        return [gr.update(visible=text in (comp.label or "")) for comp in self.components]
```