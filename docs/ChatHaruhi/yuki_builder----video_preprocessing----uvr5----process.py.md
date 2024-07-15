# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\uvr5\process.py`

```py
# 导入必要的模块和库
from .MDXNet import MDXNetDereverb  # 从当前包中导入 MDXNetDereverb 类
from .infer_uvr5 import _audio_pre_, _audio_pre_new  # 从当前包中导入 _audio_pre_ 和 _audio_pre_new 函数
from tqdm import tqdm  # 导入 tqdm 库，用于显示进度条
import os  # 导入操作系统模块
import traceback, pdb  # 导入用于调试的 traceback 和 pdb 模块
import ffmpeg  # 导入 ffmpeg 模块
import torch  # 导入 PyTorch 模块
import shutil  # 导入 shutil 模块，用于文件和目录操作
import sys  # 导入 sys 模块，用于系统相关操作

now_dir = os.getcwd()  # 获取当前工作目录
sys.path.append(now_dir)  # 将当前工作目录添加到系统路径中
torch.manual_seed(114514)  # 设置 PyTorch 随机种子
tmp = os.path.join(now_dir, "TEMP")  # 设置临时目录路径

# 删除临时目录，忽略不存在的错误
shutil.rmtree(tmp, ignore_errors=True)
# 创建临时目录，如果不存在则创建
os.makedirs(tmp, exist_ok=True)

config = dict({
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # 根据CUDA是否可用选择设备
    # 16系/10系显卡和P40强制单精度 需要修改为False
    'is_half': True  # 设置是否使用半精度
})

weight_uvr5_root = os.path.dirname(os.path.realpath(__file__))+"/uvr5_weights"  # 获取uvr5权重文件的根目录路径
uvr5_names = []
# 遍历权重文件根目录下的文件名，将.pth结尾或包含'onnx'的文件名添加到uvr5_names列表中
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))

os.environ['OPENBLAS_NUM_THREADS'] = '1'  # 设置环境变量 OPENBLAS_NUM_THREADS 为1，控制线程数

# 获取目录下的子目录列表
def get_subdir(folder_path):
    subdirectories = [os.path.abspath(os.path.join(folder_path, name)) for name in os.listdir(folder_path) if
                      os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories

# 获取目录下的文件名和文件路径列表
def get_filename(directory, format=None):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.startswith('.') and os.path.isfile(file_path):
                if format:
                    if file.endswith(format):
                        file_list.append([file, file_path])
                else:
                    file_list.append([file, file_path])
    file_list.sort()  # 对文件列表进行排序
    return file_list

# 定义 uvr 函数，接受模型名称、输入根目录、保存人声根目录、保存乐器根目录、聚合方式和格式参数作为输入
def uvr(model_name, inp_root, save_root_vocal, save_root_ins, agg, format0):
    infos = []  # 初始化信息列表
    try:
        # 去除输入路径的多余空白字符和引号，并规范化换行符和引号
        inp_root = inp_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        # 去除保存人声路径的多余空白字符和引号，并规范化换行符和引号
        save_root_vocal = (
            save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        # 去除保存乐器路径的多余空白字符和引号，并规范化换行符和引号
        save_root_ins = (
            save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        # 如果模型名称是特定值，则使用特定的音频处理函数
        if model_name == "onnx_dereverb_By_FoxJoy":
            # 使用指定的 ONNX 模型路径创建 MDXNetDereverb 对象
            pre_fun = MDXNetDereverb(onnx=f"{weight_uvr5_root}/onnx_dereverb_By_FoxJoy", chunks=15)
        else:
            # 根据模型名称选择音频预处理函数
            func = _audio_pre_ if "DeEcho" not in model_name else _audio_pre_new
            # 使用模型路径、设备、精度等参数创建预处理函数对象
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                device=config['device'],
                is_half=config['is_half'],
            )
        # 获取输入根目录下的所有子目录列表
        sub_dirs = get_subdir(f'{inp_root}')
        # 遍历每个子目录
        for dir in sub_dirs[:]:
            # 获取当前子目录下的所有音频文件列表
            voice_files = get_filename(dir)
            # 获取当前子目录的基本名称
            name = os.path.basename(os.path.normpath(dir))
            # 构建保存乐器路径和保存人声路径
            save_ins_path = f'{save_root_ins}/instrument/{name}'
            save_vocal_path = f'{save_root_vocal}/voice/{name}'
            # 遍历当前子目录下的每个音频文件
            for file, inp_path in tqdm(voice_files, f'extract {name} uvr ,convert .wav to .wav'):
                # 初始化变量
                need_reformat = 1
                done = 0
                try:
                    # 使用ffmpeg获取音频文件信息
                    info = ffmpeg.probe(inp_path, cmd="ffprobe")
                    # 如果音频文件是双声道且采样率为44100Hz，则无需重新格式化
                    if (
                            info["streams"][0]["channels"] == 2
                            and info["streams"][0]["sample_rate"] == "44100"
                    ):
                        need_reformat = 0
                        # 使用预处理函数处理音频文件
                        pre_fun._path_audio_(
                            inp_path, save_ins_path, save_vocal_path, format0
                        )
                        done = 1
                except:
                    # 发生异常时标记需要重新格式化
                    need_reformat = 1
                    traceback.print_exc()
                # 如果需要重新格式化音频文件
                if need_reformat == 1:
                    tmp_path = "%s/%s.reformatted.wav" % (tmp, os.path.basename(inp_path))
                    # 使用ffmpeg重新格式化音频文件
                    os.system(
                        "ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y -loglevel error"
                        % (inp_path, tmp_path)
                    )
                    inp_path = tmp_path
                try:
                    # 如果处理成功，则记录信息并生成结果
                    if done == 0:
                        pre_fun._path_audio_(
                            inp_path, save_ins_path, save_vocal_path, format0
                        )
                    infos.append("%s->Success" % (os.path.basename(inp_path)))
                    yield "\n".join(infos)
                except:
                    # 如果处理失败，则记录错误信息并生成结果
                    infos.append(
                        "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
                    )
                    yield "\n".join(infos)

    except:
        # 捕获所有异常并记录错误信息
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
    # 最终执行块，无论如何都会执行的部分，通常用于清理资源或完成必要的操作
    finally:
        try:
            # 检查模型名称，根据不同情况释放相关资源
            if model_name == "onnx_dereverb_By_FoxJoy":
                # 如果模型名称为特定值，删除预测函数中的模型对象
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                # 否则，删除预测函数中的模型对象
                del pre_fun.model
                del pre_fun
        except:
            # 捕获任何异常并打印堆栈信息
            traceback.print_exc()
        # 输出调试信息，指示清理空缓存的操作即将进行
        print("clean_empty_cache")
        # 如果CUDA可用，则清空当前CUDA设备的缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # 使用生成器返回包含信息的字符串，以换行符连接所有信息
    yield "\n".join(infos)
def uvr_prediction(model_name, inp_path, save_root_vocal, save_root_ins, agg, format0):
    """
    分离音频
    :param model_name: 模型名称
    :param inp_path: 输入文件
    :param save_root_vocal: 说话人保存位置
    :param save_root_ins: 伴奏保存位置
    :param agg: 人声提取激进程度
    :param format0: 输出音频格式列表
    :return: vocal_path: 分离后的说话人音频文件路径
             others_path: 分离后的伴奏音频文件路径
    """
    try:
        # 路径格式化
        save_root_vocal = (
            save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        save_root_ins = (
            save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )

        # 模型加载
        if model_name == "onnx_dereverb_By_FoxJoy":
            # 使用指定的 ONNX 模型进行音频去混响处理
            pre_fun = MDXNetDereverb(onnx=f"{weight_uvr5_root}/onnx_dereverb_By_FoxJoy", chunks=15)
        else:
            # 根据模型名称选择不同的预处理函数
            func = _audio_pre_ if "DeEcho" not in model_name else _audio_pre_new
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                device=config['device'],
                is_half=config['is_half'],
            )
        
        # 判断音频文件是否符合要求
        need_reformat = 1
        info = ffmpeg.probe(inp_path, cmd="ffprobe")

        if (
                'mov' in info["format"]['format_name'] or (
                    info["streams"][0]["channels"] == 2
                    and info["streams"][0]["sample_rate"] == "44100")
        ):
            need_reformat = 0

        if need_reformat == 1:
            # 对音频文件进行重新格式化
            tmp_path = "%s/%s.reformatted.wav" % (tmp, os.path.basename(inp_path))
            os.system(
                "ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y -loglevel error"
                % (inp_path, tmp_path)
            )
            inp_path = tmp_path

        # 处理音频文件，分离说话人音频和伴奏音频
        vocal_path, others_path = pre_fun._path_audio_(
            inp_path, save_root_vocal, save_root_ins, format0
        )
        return vocal_path, others_path
    except:
        traceback.print_exc()
        return None, None
    finally:
        try:
            # 清理模型相关资源
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()
        
        # 清空 CUDA 缓存
        print("clean_empty_cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    # 音频说话人的输出文件夹路径
    opt_vocal_root = '/media/checkpoint/speech_data/video/audio/test/output/'
    # 音频伴奏的输出文件夹路径
    opt_ins_root = '/media/checkpoint/speech_data/video/audio/test/output/'

    # 删除已存在的输出文件夹并创建新的空文件夹
    shutil.rmtree(opt_vocal_root, ignore_errors=True)
    os.makedirs(opt_vocal_root, exist_ok=True)

    shutil.rmtree(opt_ins_root, ignore_errors=True)
    os.makedirs(opt_ins_root, exist_ok=True)
    
    # 设置人声提取激进程度
    agg = 10
    # 设置输出音频格式列表
    format0 = ["wav", "flac", "mp3", "m4a"]

    # 在使用的文件夹中，存储来自 uvr5_weights 文件夹下发的模型名称列表
    uvr5_names
    # 定义一个包含字符串元素的列表
    ['onnx_dereverb_By_FoxJoy',
     'HP2_all_vocals',
     'HP2-人声vocals+非人声instrumentals',
     'HP3_all_vocals',
     'HP5_only_main_vocal',
     'HP5-主旋律人声vocals+其他instrumentals',
     'VR-DeEchoAggressive',
     'VR-DeEchoDeReverb',
     'VR-DeEchoNormal']
    """
    # 指定音频输入路径
    wav_input = '/media/checkpoint/speech_data/video/PleasantGoatandBigBigMovie_23mi.mp4'
    # 调用 uvr_prediction 函数，传入参数并返回人声路径和其他声音路径
    vocal_path, others_path = uvr_prediction(uvr5_names[5], wav_input,
                                             opt_vocal_root,
                                             opt_ins_root,
                                             agg,
                                             format0[0]
                                             )
    # 打印人声路径
    print(vocal_path)
    # 打印其他声音路径
    print(others_path)
```