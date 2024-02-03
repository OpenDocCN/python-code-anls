# `stable-diffusion-webui\modules\ui_postprocessing.py`

```py
# 导入所需的库和模块
import gradio as gr
from modules import scripts, shared, ui_common, postprocessing, call_queue, ui_toprow
import modules.generation_parameters_copypaste as parameters_copypaste

# 创建用户界面
def create_ui():
    # 创建一个不可见的标签组件
    dummy_component = gr.Label(visible=False)
    # 创建一个状态组件，用于跟踪选项卡的索引
    tab_index = gr.State(value=0)

    # 创建一个行布局，设置高度不相等，样式为紧凑
    with gr.Row(equal_height=False, variant='compact'):
        # 创建一个列布局，样式为紧凑
        with gr.Column(variant='compact'):
            # 创建选项卡组件
            with gr.Tabs(elem_id="mode_extras"):
                # 创建选项卡项目，名称为'Single Image'，设置 ID 和元素 ID
                with gr.TabItem('Single Image', id="single_image", elem_id="extras_single_tab") as tab_single:
                    # 创建图像组件，用于上传单个图像，设置交互性和类型
                    extras_image = gr.Image(label="Source", source="upload", interactive=True, type="pil", elem_id="extras_image")

                # 创建选项卡项目，名称为'Batch Process'，设置 ID 和元素 ID
                with gr.TabItem('Batch Process', id="batch_process", elem_id="extras_batch_process_tab") as tab_batch:
                    # 创建文件组件，用于批量处理图像，设置交互性
                    image_batch = gr.Files(label="Batch Process", interactive=True, elem_id="extras_image_batch")

                # 创建选项卡项目，名称为'Batch from Directory'，设置 ID 和元素 ID
                with gr.TabItem('Batch from Directory', id="batch_from_directory", elem_id="extras_batch_directory_tab") as tab_batch_dir:
                    # 创建文本框组件，用于输入目录路径，设置占位符和元素 ID
                    extras_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, placeholder="A directory on the same machine where the server is running.", elem_id="extras_batch_input_dir")
                    # 创建文本框组件，用于输入输出目录路径，设置占位符和元素 ID
                    extras_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, placeholder="Leave blank to save images to the default path.", elem_id="extras_batch_output_dir")
                    # 创建复选框组件，用于控制是否显示结果图像，设置默认值和元素 ID
                    show_extras_results = gr.Checkbox(label='Show result images', value=True, elem_id="extras_show_extras_results")

            # 设置脚本输入参数
            script_inputs = scripts.scripts_postproc.setup_ui()

        # 创建顶部行组件
        with gr.Column():
            # 创建顶部行组件，设置为紧凑模式，不是图像到图像的转换，设置 ID 部分为'extras'
            toprow = ui_toprow.Toprow(is_compact=True, is_img2img=False, id_part="extras")
            # 创建内联顶部行图像
            toprow.create_inline_toprow_image()
            # 获取提交按钮
            submit = toprow.submit

            # 创建结果图像、信息和日志的输出面板
            result_images, html_info_x, html_info, html_log = ui_common.create_output_panel("extras", shared.opts.outdir_extras_samples)
    # 选择单图模式，将 tab_index 设置为 0
    tab_single.select(fn=lambda: 0, inputs=[], outputs=[tab_index])
    # 选择批处理模式，将 tab_index 设置为 1
    tab_batch.select(fn=lambda: 1, inputs=[], outputs=[tab_index])
    # 选择批处理目录模式，将 tab_index 设置为 2
    tab_batch_dir.select(fn=lambda: 2, inputs=[], outputs=[tab_index])

    # 点击提交按钮，调用 postprocessing.run_postprocessing_webui 函数，并将结果输出到指定的输出组件中
    submit.click(
        fn=call_queue.wrap_gradio_gpu_call(postprocessing.run_postprocessing_webui, extra_outputs=[None, '']),
        _js="submit_extras",
        inputs=[
            dummy_component,
            tab_index,
            extras_image,
            image_batch,
            extras_batch_input_dir,
            extras_batch_output_dir,
            show_extras_results,
            *script_inputs
        ],
        outputs=[
            result_images,
            html_info_x,
            html_log,
        ],
        show_progress=False,
    )

    # 将 extras_image 添加到参数复制粘贴字段中，字段名称为 "extras"
    parameters_copypaste.add_paste_fields("extras", extras_image, None)

    # 当 extras_image 发生改变时，调用 scripts.scripts_postproc.image_changed 函数
    extras_image.change(
        fn=scripts.scripts_postproc.image_changed,
        inputs=[], outputs=[]
    )
```