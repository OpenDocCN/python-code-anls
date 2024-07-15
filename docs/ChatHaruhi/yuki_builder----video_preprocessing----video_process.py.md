# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\video_process.py`

```py
from .uvr5 import uvr_prediction, uvr5_names  # 导入自定义模块中的函数和变量
from argparse import Namespace  # 导入Namespace类，用于处理命令行参数
import os  # 导入操作系统相关的功能
import shutil  # 导入文件和目录操作相关的功能
import argparse  # 导入命令行参数解析相关的功能

def run_bgm_remover(args: Namespace):
    # 检查输入文件是否存在并且是一个文件
    if not os.path.isfile(args.input_file):
        print('input_file is not exist')
        return

    # 检查 opt_vocal_root 是否存在并且是一个目录
    if not os.path.isdir(args.opt_vocal_root):
        print('warning opt_vocal_root is not exist')
        # 如果不存在，先删除 opt_vocal_root，然后创建它
        shutil.rmtree(args.opt_vocal_root, ignore_errors=True)
        os.makedirs(args.opt_vocal_root, exist_ok=True)
        print('create folder', args.opt_vocal_root)

    # 检查 opt_ins_root 是否存在并且是一个目录
    if not os.path.isdir(args.opt_ins_root):
        print('warning opt_ins_root is not exist')
        # 如果不存在，先删除 opt_ins_root，然后创建它
        shutil.rmtree(args.opt_ins_root, ignore_errors=True)
        os.makedirs(args.opt_ins_root, exist_ok=True)
        print('create folder', args.opt_ins_root)

    # 设置人声提取的激进程度
    agg = 10
    # 设置文件格式为 "wav"
    format0 = ["wav", "flac", "mp3", "m4a"]
    
    # 调用 uvr_prediction 函数进行人声和其它声音的分离
    vocal_path, others_path = uvr_prediction(uvr5_names[5], args.input_file,
                                             args.opt_vocal_root,
                                             args.opt_ins_root,
                                             agg,
                                             format0[0]
                                             )

    return vocal_path, others_path

if __name__ == '__main__':
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(
        description='uvr5 processing of vocal accompaniment separation'
    )
    
    # 添加命令行参数：输入文件路径
    parser.add_argument('--input_file', default='input_file', type=str, required=True, help="source path")
    # 添加命令行参数：输出人声的根目录路径
    parser.add_argument('--opt_vocal_root', default='out_folder', type=str,  help="vocal path")
    # 添加命令行参数：输出乐器的根目录路径
    parser.add_argument('--opt_ins_root', default='out_folder', type=str,  help="instrument path")
    
    # 解析命令行参数
    args = parser.parse_args()
    # 打印命令行参数帮助信息
    parser.print_help()
    
    # 调用 run_bgm_remover 函数，执行人声和乐器分离操作
    run_bgm_remover(args)
```