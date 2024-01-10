# `so-vits-svc\webUI.py`

```
# 导入所需的库
import glob  # 用于查找匹配的文件路径名
import json  # 用于处理 JSON 数据
import logging  # 用于记录日志
import os  # 用于与操作系统交互
import re  # 用于处理正则表达式
import subprocess  # 用于在子进程中执行命令
import sys  # 提供对 Python 解释器的访问
import time  # 用于处理时间
import traceback  # 用于追踪异常
from itertools import chain  # 用于将多个迭代器连接在一起
from pathlib import Path  # 用于处理文件路径

# 下面是一些特定库的导入和设置
import gradio as gr  # 用于构建交互式界面
import librosa  # 用于音频处理
import numpy as np  # 用于处理数组
import soundfile  # 用于读写音频文件
import torch  # 用于构建神经网络模型

# 导入自定义的模块和函数
from compress_model import removeOptimizer  # 从 compress_model 模块中导入 removeOptimizer 函数
from edgetts.tts_voices import SUPPORTED_LANGUAGES  # 从 edgetts.tts_voices 模块中导入 SUPPORTED_LANGUAGES 变量
from inference.infer_tool import Svc  # 从 inference.infer_tool 模块中导入 Svc 类
from utils import mix_model  # 从 utils 模块中导入 mix_model 函数

# 设置日志级别
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)

# 初始化变量
model = None  # 模型变量
spk = None  # 说话人变量
debug = False  # 调试模式开关

local_model_root = './trained'  # 本地模型根目录

cuda = {}  # CUDA 设备信息字典
if torch.cuda.is_available():  # 如果有可用的 CUDA 设备
    for i in range(torch.cuda.device_count()):  # 遍历 CUDA 设备
        device_name = torch.cuda.get_device_properties(i).name  # 获取 CUDA 设备名称
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"  # 将 CUDA 设备信息添加到字典中

# 定义函数 upload_mix_append_file
def upload_mix_append_file(files, sfiles):
    try:
        if sfiles is None:  # 如果 sfiles 为空
            file_paths = [file.name for file in files]  # 获取 files 中文件的路径名列表
        else:
            file_paths = [file.name for file in chain(files, sfiles)]  # 获取 files 和 sfiles 中文件的路径名列表
        p = {file: 100 for file in file_paths}  # 创建文件路径名到数值的字典
        return file_paths, mix_model_output1.update(value=json.dumps(p, indent=2))  # 返回文件路径名列表和更新 mix_model_output1 的值
    except Exception as e:  # 捕获异常
        if debug:  # 如果调试模式开启
            traceback.print_exc()  # 打印异常信息
        raise gr.Error(e)  # 抛出 gr.Error 异常

# 定义函数 mix_submit_click
def mix_submit_click(js, mode):
    try:
        assert js.lstrip() != ""  # 断言 js 去除左侧空格后不为空
        modes = {"凸组合": 0, "线性组合": 1}  # 定义模式字典
        mode = modes[mode]  # 获取指定模式
        data = json.loads(js)  # 解析 JSON 数据
        data = list(data.items())  # 将 JSON 数据转换为列表
        model_path, mix_rate = zip(*data)  # 解压缩数据
        path = mix_model(model_path, mix_rate, mode)  # 调用 mix_model 函数进行混合模型
        return f"成功，文件被保存在了{path}"  # 返回成功信息
    except Exception as e:  # 捕获异常
        if debug:  # 如果调试模式开启
            traceback.print_exc()  # 打印异常信息
        raise gr.Error(e)  # 抛出 gr.Error 异常

# 定义函数 updata_mix_info
def updata_mix_info(files):
    # 尝试执行以下代码块，捕获可能出现的异常
    try:
        # 如果 files 为空，则更新 mix_model_output1 的值为空字符串
        if files is None :
            return mix_model_output1.update(value="")
        # 使用文件列表中每个文件的名称作为键，值为 100，创建字典 p
        p = {file.name:100 for file in files}
        # 更新 mix_model_output1 的值为 p 的 JSON 格式字符串，缩进为 2
        return mix_model_output1.update(value=json.dumps(p,indent=2))
    # 捕获任何异常，并将其作为变量 e 存储
    except Exception as e:
        # 如果 debug 为真，则打印异常的堆栈信息
        if debug:
            traceback.print_exc()
        # 抛出自定义的错误，内容为捕获到的异常 e
        raise gr.Error(e)
# 加载模型并进行分析
def modelAnalysis(model_path,config_path,cluster_model_path,device,enhance,diff_model_path,diff_config_path,only_diffusion,use_spk_mix,local_model_enabled,local_model_selection):
    global model
    # 异常处理
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)

# 卸载模型
def modelUnload():
    global model
    if model is None:
        return sid.update(choices = [],value=""),"没有模型需要卸载!"
    else:
        model.unload_model()
        model = None
        torch.cuda.empty_cache()
        return sid.update(choices = [],value=""),"模型卸载完毕!"

# VC 推断
def vc_infer(output_format, sid, audio_path, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment):
    global model
    # 使用模型进行切片推断
    _audio = model.slice_inference(
        audio_path,
        sid,
        vc_transform,
        slice_db,
        cluster_ratio,
        auto_f0,
        noise_scale,
        pad_seconds,
        cl_num,
        lg_num,
        lgr_num,
        f0_predictor,
        enhancer_adaptive_key,
        cr_threshold,
        k_step,
        use_spk_mix,
        second_encoding,
        loudness_envelope_adjustment
    )  
    # 清除空白
    model.clear_empty()
    # 构建保存文件的路径，并保存到results文件夹内
    str(int(time.time()))
    if not os.path.exists("results"):
        os.makedirs("results")
    key = "auto" if auto_f0 else f"{int(vc_transform)}key"
    cluster = "_" if cluster_ratio == 0 else f"_{cluster_ratio}_"
    isdiffusion = "sovits"
    if model.shallow_diffusion:
        isdiffusion = "sovdiff"
    if model.only_diffusion:
        isdiffusion = "diff"
    output_file_name = 'result_'+truncated_basename+f'_{sid}_{key}{cluster}{isdiffusion}.{output_format}'
    output_file = os.path.join("results", output_file_name)
    soundfile.write(output_file, _audio, model.target_sample, format=output_format)
    # 返回 output_file 变量的值作为函数的输出结果
    return output_file
# 定义一个声音转换函数，接受多个参数
def vc_fn(sid, input_audio, output_format, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold,k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment):
    # 声明全局变量 model
    global model
    try:
        # 如果输入音频为空，则返回提示信息和空值
        if input_audio is None:
            return "You need to upload an audio", None
        # 如果模型为空，则返回提示信息和空值
        if model is None:
            return "You need to upload an model", None
        # 如果模型的 cluster_model 属性为空且 feature_retrieval 属性为 False，则根据 cluster_ratio 的值返回相应提示信息和空值
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "You need to upload an cluster model or feature retrieval model before assigning cluster ratio!", None
        # 读取输入音频文件，并返回音频数据和采样率
        audio, sampling_rate = soundfile.read(input_audio)
        # 如果音频数据的类型为整数类型，则将其转换为浮点数类型
        if np.issubdtype(audio.dtype, np.integer):
            audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        # 如果音频数据的维度大于1，则转换为单声道
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        # 获取输入音频文件的文件名，并去掉奇怪的固定后缀
        truncated_basename = Path(input_audio).stem[:-6]
        # 将处理后的音频数据写入文件
        processed_audio = os.path.join("raw", f"{truncated_basename}.wav")
        soundfile.write(processed_audio, audio, sampling_rate, format="wav")
        # 调用 vc_infer 函数进行声音转换，返回处理结果文件
        output_file = vc_infer(output_format, sid, processed_audio, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment)
        # 返回成功提示和处理结果文件
        return "Success", output_file
    # 捕获异常并处理
    except Exception as e:
        # 如果开启了调试模式，则打印异常信息
        if debug:
            traceback.print_exc()
        # 抛出异常
        raise gr.Error(e)

# 定义一个文本清理函数，用于去除文本中的换行符、逗号、括号和空格
def text_clear(text):
    return re.sub(r"[\n\,\(\) ]", "", text)
# 定义一个函数，用于处理文本转语音的相关参数，并返回处理结果
def vc_fn2(_text, _lang, _gender, _rate, _volume, sid, output_format, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold, k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment):
    global model
    try:
        # 如果模型为空，则返回提示信息和空值
        if model is None:
            return "You need to upload an model", None
        # 如果模型的 cluster_model 属性为空且 feature_retrieval 属性为 False，则返回提示信息和空值
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "You need to upload an cluster model or feature retrieval model before assigning cluster ratio!", None
        # 根据速率参数设置语速
        _rate = f"+{int(_rate*100)}%" if _rate >= 0 else f"{int(_rate*100)}%"
        # 根据音量参数设置音量
        _volume = f"+{int(_volume*100)}%" if _volume >= 0 else f"{int(_volume*100)}%"
        # 如果语言为自动识别，则根据性别参数设置性别，并调用子进程执行文本转语音
        if _lang == "Auto":
            _gender = "Male" if _gender == "男" else "Female"
            subprocess.run([sys.executable, "edgetts/tts.py", _text, _lang, _rate, _volume, _gender])
        # 否则，调用子进程执行文本转语音
        else:
            subprocess.run([sys.executable, "edgetts/tts.py", _text, _lang, _rate, _volume])
        # 设置目标采样率
        target_sr = 44100
        # 加载音频文件，并重采样
        y, sr = librosa.load("tts.wav")
        resampled_y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        # 将重采样后的音频写入文件
        soundfile.write("tts.wav", resampled_y, target_sr, subtype = "PCM_16")
        input_audio = "tts.wav"
        # 调用 vc_infer 函数进行声音转换
        output_file_path = vc_infer(output_format, sid, input_audio, "tts", vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment)
        # 删除临时生成的音频文件
        os.remove("tts.wav")
        # 返回成功提示和输出文件路径
        return "Success", output_file_path
    # 捕获异常并处理
    except Exception as e:
        # 如果 debug 为真，则打印异常信息
        if debug: traceback.print_exc()  # noqa: E701
        # 抛出自定义异常
        raise gr.Error(e)

# 定义一个函数，用于模型压缩
def model_compression(_model):
    # 如果模型为空，则返回提示信息
    if _model == "":
        return "请先选择要压缩的模型"
    else:
        # 获取模型路径的目录和文件名
        model_path = os.path.split(_model.name)
        # 分离文件名和扩展名
        filename, extension = os.path.splitext(model_path[1])
        # 生成压缩后的模型文件名
        output_model_name = f"{filename}_compressed{extension}"
        # 生成压缩后的模型文件路径
        output_path = os.path.join(os.getcwd(), output_model_name)
        # 移除优化器并保存压缩后的模型
        removeOptimizer(_model.name, output_path)
        # 返回保存成功的消息
        return f"模型已成功被保存在了{output_path}"
# 扫描本地模型文件夹，返回符合条件的文件夹列表
def scan_local_models():
    # 初始化结果列表
    res = []
    # 获取所有符合条件的 JSON 文件路径
    candidates = glob.glob(os.path.join(local_model_root, '**', '*.json'), recursive=True)
    # 将文件路径的目录部分提取出来，去重
    candidates = set([os.path.dirname(c) for c in candidates])
    # 遍历每个候选文件夹
    for candidate in candidates:
        # 获取当前文件夹下的所有 JSON 文件
        jsons = glob.glob(os.path.join(candidate, '*.json'))
        # 获取当前文件夹下的所有 PTH 文件
        pths = glob.glob(os.path.join(candidate, '*.pth'))
        # 如果当前文件夹下恰好有一个 JSON 文件和一个 PTH 文件
        if (len(jsons) == 1 and len(pths) == 1):
            # 将当前文件夹路径加入结果列表
            res.append(candidate)
    # 返回结果列表
    return res

# 刷新本地模型下拉框的选项
def local_model_refresh_fn():
    # 获取符合条件的本地模型文件夹列表
    choices = scan_local_models()
    # 更新下拉框的选项
    return gr.Dropdown.update(choices=choices)

# 调试模式切换函数
def debug_change():
    # 声明使用全局变量 debug
    global debug
    # 将 debug_button 的值赋给 debug
    debug = debug_button.value

# 创建应用程序界面
with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue = gr.themes.colors.green,
        font=["Source Sans Pro", "Arial", "sans-serif"],
        font_mono=['JetBrains mono', "Consolas", 'Courier New']
    ),
) as app:
    # 创建一个包含多个选项卡的组件
    with gr.Tabs():
        # 在选项卡中创建一个面板
        with gr.Row(variant="panel"):
            # 在面板中创建一个列
            with gr.Column():
                # 在列中添加一个Markdown文本，用于显示WebUI设置
                gr.Markdown(value="""
                    <font size=2> WebUI设置</font>
                    """)
                # 创建一个复选框，用于控制是否开启Debug模式
                debug_button = gr.Checkbox(label="Debug模式，如果向社区反馈BUG需要打开，打开后控制台可以显示具体错误提示", value=debug)
        # 点击按钮，刷新本地模型列表
        local_model_refresh_btn.click(local_model_refresh_fn, outputs=local_model_selection)
        # 在选项卡切换时，设置本地模型的启用/禁用状态
        local_model_tab_upload.select(lambda: False, outputs=local_model_enabled)
        local_model_tab_local.select(lambda: True, outputs=local_model_enabled)
        
        # 点击按钮，执行vc_fn函数，并传入指定的参数，将结果输出到指定的位置
        vc_submit.click(vc_fn, [sid, vc_input3, output_format, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold,k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment], [vc_output1, vc_output2])
        # 点击按钮，执行vc_fn2函数，并传入指定的参数，将结果输出到指定的位置
        vc_submit2.click(vc_fn2, [text2tts, tts_lang, tts_gender, tts_rate, tts_volume, sid, output_format, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold,k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment], [vc_output1, vc_output2])

        # 当debug_button的状态改变时，执行debug_change函数
        debug_button.change(debug_change,[],[])
        # 点击按钮，执行modelAnalysis函数，并传入指定的参数，将结果输出到指定的位置
        model_load_button.click(modelAnalysis,[model_path,config_path,cluster_model_path,device,enhance,diff_model_path,diff_config_path,only_diffusion,use_spk_mix,local_model_enabled,local_model_selection],[sid,sid_output])
        # 点击按钮，执行modelUnload函数，并传入指定的参数，将结果输出到指定的位置
        model_unload_button.click(modelUnload,[],[sid,sid_output])
    # 打开浏览器，访问指定的URL
    os.system("start http://127.0.0.1:7860")
    # 启动应用程序
    app.launch()
```