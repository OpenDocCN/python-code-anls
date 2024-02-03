# `stable-diffusion-webui\modules\postprocessing.py`

```
# 导入所需的模块
import os
from PIL import Image
from modules import shared, images, devices, scripts, scripts_postprocessing, ui_common, generation_parameters_copypaste
from modules.shared import opts

# 运行后处理程序，根据 extras_mode 参数的不同执行不同的操作
def run_postprocessing(extras_mode, image, image_folder, input_dir, output_dir, show_extras_results, *args, save_output: bool = True):
    # 执行内存回收
    devices.torch_gc()
    
    # 开始处理任务
    shared.state.begin(job="extras")
    
    # 初始化输出列表
    outputs = []
    
    # 定义获取图片的函数
    def get_images(extras_mode, image, image_folder, input_dir):
        # 如果 extras_mode 为 1
        if extras_mode == 1:
            # 遍历图片文件夹中的图片
            for img in image_folder:
                # 如果图片是 PIL.Image.Image 类型
                if isinstance(img, Image.Image):
                    image = img
                    fn = ''
                else:
                    # 打开图片文件
                    image = Image.open(os.path.abspath(img.name))
                    fn = os.path.splitext(img.orig_name)[0]
                yield image, fn
        # 如果 extras_mode 为 2
        elif extras_mode == 2:
            # 断言 --hide-ui-dir-config 选项未禁用
            assert not shared.cmd_opts.hide_ui_dir_config, '--hide-ui-dir-config option must be disabled'
            # 断言已选择输入目录
            assert input_dir, 'input directory not selected'
            
            # 获取输入目录中的图片列表
            image_list = shared.listfiles(input_dir)
            for filename in image_list:
                yield filename, filename
        else:
            # 断言已选择图片
            assert image, 'image not selected'
            yield image, None
    
    # 如果 extras_mode 为 2 且输出目录不为空，则将输出路径设置为输出目录，否则设置为 opts.outdir_samples 或 opts.outdir_extras_samples
    if extras_mode == 2 and output_dir != '':
        outpath = output_dir
    else:
        outpath = opts.outdir_samples or opts.outdir_extras_samples
    
    # 初始化信息文本
    infotext = ''
    
    # 获取待处理数据列表
    data_to_process = list(get_images(extras_mode, image, image_folder, input_dir))
    shared.state.job_count = len(data_to_process)
    
    # 执行内存回收
    devices.torch_gc()
    # 结束任务处理
    shared.state.end()
    # 返回输出列表、信息文本的 HTML 格式和空字符串
    return outputs, ui_common.plaintext_to_html(infotext), ''

# 在 Web UI 中运行后处理程序
def run_postprocessing_webui(id_task, *args, **kwargs):
    return run_postprocessing(*args, **kwargs)
# 运行额外功能的处理程序，接受多个参数
def run_extras(extras_mode, resize_mode, image, image_folder, input_dir, output_dir, show_extras_results, gfpgan_visibility, codeformer_visibility, codeformer_weight, upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility, upscale_first: bool, save_output: bool = True):
    """old handler for API"""

    # 创建运行参数字典
    args = scripts.scripts_postproc.create_args_for_run({
        "Upscale": {
            "upscale_mode": resize_mode,
            "upscale_by": upscaling_resize,
            "upscale_to_width": upscaling_resize_w,
            "upscale_to_height": upscaling_resize_h,
            "upscale_crop": upscaling_crop,
            "upscaler_1_name": extras_upscaler_1,
            "upscaler_2_name": extras_upscaler_2,
            "upscaler_2_visibility": extras_upscaler_2_visibility,
        },
        "GFPGAN": {
            "enable": True,
            "gfpgan_visibility": gfpgan_visibility,
        },
        "CodeFormer": {
            "enable": True,
            "codeformer_visibility": codeformer_visibility,
            "codeformer_weight": codeformer_weight,
        },
    })

    # 运行后处理函数，传入额外功能模式、图像、图像文件夹、输入目录、输出目录、是否显示额外功能结果、参数列表，保存输出结果
    return run_postprocessing(extras_mode, image, image_folder, input_dir, output_dir, show_extras_results, *args, save_output=save_output)
```