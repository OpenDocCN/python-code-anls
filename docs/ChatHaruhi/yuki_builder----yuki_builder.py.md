# `.\Chat-Haruhi-Suzumiya\yuki_builder\yuki_builder.py`

```py
import argparse  # 导入命令行参数解析模块

from run_whisper import run_whisper  # 导入自定义模块 run_whisper 中的 run_whisper 函数
from srt2csv import srt2csv  # 导入自定义模块 srt2csv 中的 srt2csv 函数
from crop import crop  # 导入自定义模块 crop 中的 crop 函数
from recognize import recognize  # 导入自定义模块 recognize 中的 recognize 函数
from video_preprocessing.video_process import run_bgm_remover  # 导入自定义模块 video_process 中的 run_bgm_remover 函数

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YukiBuilder')  # 创建命令行参数解析对象，设置描述信息为 'YukiBuilder'

    parser.add_argument('-verbose', action='store_true')  # 添加命令行参数选项 '-verbose'，若指定则设为 True

    subparsers = parser.add_subparsers(dest='subcommand')  # 添加子命令解析器，并指定目标属性为 'subcommand'

    whisper_parser = subparsers.add_parser('whisper')  # 添加子命令 'whisper' 的解析器
    whisper_parser.add_argument('-input_video', required=True)  # 为 'whisper' 子命令添加必选参数 '-input_video'
    whisper_parser.add_argument('-srt_folder', required=True)  # 为 'whisper' 子命令添加必选参数 '-srt_folder'

    srt2csv_parser = subparsers.add_parser('srt2csv')  # 添加子命令 'srt2csv' 的解析器
    srt2csv_parser.add_argument('-input_srt', required=True)  # 为 'srt2csv' 子命令添加必选参数 '-input_srt'
    srt2csv_parser.add_argument('-srt_folder', required=True)  # 为 'srt2csv' 子命令添加必选参数 '-srt_folder'

    crop_parser = subparsers.add_parser('crop')  # 添加子命令 'crop' 的解析器
    crop_parser.add_argument('-annotate_map', required=True)  # 为 'crop' 子命令添加必选参数 '-annotate_map'
    crop_parser.add_argument('-role_audios', required=True)  # 为 'crop' 子命令添加必选参数 '-role_audios'

    recognize_parser = subparsers.add_parser('recognize')  # 添加子命令 'recognize' 的解析器
    recognize_parser.add_argument('-input_video', required=True)  # 为 'recognize' 子命令添加必选参数 '-input_video'
    recognize_parser.add_argument('-input_srt', required=True)  # 为 'recognize' 子命令添加必选参数 '-input_srt'
    recognize_parser.add_argument('-role_audios', required=True)  # 为 'recognize' 子命令添加必选参数 '-role_audios'
    recognize_parser.add_argument('-output_folder', required=True)  # 为 'recognize' 子命令添加必选参数 '-output_folder'

    bgm_remover_parser = subparsers.add_parser('bgm_remover')  # 添加子命令 'bgm_remover' 的解析器
    bgm_remover_parser.add_argument('--input_file', default='input_file', type=str, required=True, help="vocal path")  # 为 'bgm_remover' 子命令添加可选参数 '--input_file'，默认值为 'input_file'，类型为字符串，必选
    bgm_remover_parser.add_argument('--opt_vocal_root', default='out_folder', type=str, help="vocal path")  # 为 'bgm_remover' 子命令添加可选参数 '--opt_vocal_root'，默认值为 'out_folder'，类型为字符串，帮助信息为 "vocal path"
    bgm_remover_parser.add_argument('--opt_ins_root', default='out_folder', type=str, help="instrument path")  # 为 'bgm_remover' 子命令添加可选参数 '--opt_ins_root'，默认值为 'out_folder'，类型为字符串，帮助信息为 "instrument path"

    args = parser.parse_args()  # 解析命令行参数并将结果存储在 args 中

    if args.subcommand == 'bgm_remover':  # 如果命令行参数的子命令是 'bgm_remover'
        run_bgm_remover(args)  # 调用 run_bgm_remover 函数处理参数 args
    if args.subcommand == 'whisper':  # 如果命令行参数的子命令是 'whisper'
        run_whisper(args)  # 调用 run_whisper 函数处理参数 args
    elif args.subcommand == 'srt2csv':  # 如果命令行参数的子命令是 'srt2csv'
        srt2csv(args)  # 调用 srt2csv 函数处理参数 args
    elif args.subcommand == 'crop':  # 如果命令行参数的子命令是 'crop'
        crop(args)  # 调用 crop 函数处理参数 args
    elif args.subcommand == 'recognize':  # 如果命令行参数的子命令是 'recognize'
        recognize(args)  # 调用 recognize 函数处理参数 args
```