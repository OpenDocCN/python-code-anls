# `.\Chat-Haruhi-Suzumiya\yuki_builder\srt2csv.py`

```py
"""
Covert .ass or .srt subtitles to a 4 columns .csv file
"""
# 导入必要的库
import argparse  # 用于解析命令行参数
import os  # 提供与操作系统相关的功能
import pathlib  # 提供操作路径的类和方法
import csv  # 用于读写CSV文件
import ass  # 处理ASS字幕格式的库
import re  # 提供正则表达式操作的功能

HAS_CHINESE = False  # 全局变量，标记是否含有中文字符，默认为False

# 将srt或ass字幕文件转换为CSV文件的函数
def srt2csv(args):
    if args.verbose:
        print('runing srt2csv')  # 如果设置了verbose参数，则打印提示信息

    # 检查输入的srt_folder路径是否是一个文件夹
    if not os.path.isdir(args.srt_folder):
        print('warning: the folder{} is not exist'.format(args.srt_folder))
        # 如果不存在，则创建该文件夹
        os.makedirs(args.srt_folder)
        print('create folder', args.srt_folder)

    # 检查输入的input_srt文件是否存在
    input_srt_file = args.input_srt
    output_folder = args.srt_folder
    if not os.path.isfile(input_srt_file):
        print('Error: The input file {} is not exist'.format(input_srt_file))
        return
    
    # 检查输入的input_srt文件是否是srt或ass格式
    if not (pathlib.Path(input_srt_file).suffix == '.srt' or pathlib.Path(input_srt_file).suffix == '.ass'):
        print('Error: The input file {} must be a .srt or .ass file'.format(input_srt_file))
        return
    # 调用convert函数进行转换
    convert(input_srt_file, output_folder, True)

# 创建CSV文件的函数
def render_csv(final_result, csv_file):
    if os.path.exists(csv_file):
        os.remove(csv_file)  # 如果CSV文件已经存在，则删除之
    # 打开CSV文件，写入标题行，并遍历final_result写入数据行
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["空白","内容","开始时间","结束时间"])  # 写入CSV的标题行
        for i in final_result:
            if not (i["Text"] and i["TimecodeIn"] and i["TimecodeOut"]):
                continue  # 如果Text、TimecodeIn、TimecodeOut有为空的情况，则跳过该行数据
            writer.writerow(['', i["Text"], i["TimecodeIn"], i["TimecodeOut"]])  # 写入CSV的数据行
    return

# 判断一行文本是否含有日文字符的函数
def is_japanese(line):
    # 使用正则表达式判断是否包含日文片假名（Katakana）或平假名（Hiragana）
    re_words_1 = re.compile(u"[\u30a0-\u30ff]+")  # 日文片假名（Katakana）
    re_words_2 = re.compile(u"[\u3040-\u309f]+")  # 日文平假名（Hiragana）
    m_1 = re_words_1.search(line, 0)
    m_2 = re_words_2.search(line, 0)
    if m_1 or m_2:
        return True  # 如果匹配到日文字符，则返回True
    return False  # 否则返回False

# 解析srt字幕文件的函数
def internalise(lines, keep_japanese):
    result = []
    GET_TEXT = 1
    WAITING = 2
    cue = 0    
    current_state = WAITING
    start_time = ""
    prev_start_time = ""
    prev_end_time = ""
    end_time = ""
    text = ""
    text_line = 0
    current_cue = {}
    # 遍历输入的文本行列表
    for line in lines:
        # 去除行两端的空白字符
        line = line.strip()
        # 如果行包含 "-->" 字符串，表示这是一个时间码行
        if "-->" in line:
            # 递增字幕序号计数器
            cue += 1
            # 提取起始时间和结束时间
            start_time = line.split('-->')[0].strip()
            end_time = line.split('-->')[1].strip()
            
            # 设置当前状态为获取文本内容
            current_state = GET_TEXT
            # 初始化文本行计数器
            text_line = 0
            
            # 将起始时间和结束时间存入当前字幕对象
            current_cue["TimecodeIn"] = start_time
            current_cue["TimecodeOut"] = end_time
            
            # 继续处理下一行
            continue
        
        # 如果当前行为空白行或者是日文且不保留日文内容
        if line == "" or (is_japenese(line) and not keep_japanese):
            # 将已收集的文本存入当前字幕对象
            current_cue["Text"] = text
            # 将当前字幕对象添加到结果列表中
            result.append(current_cue)
            # 重置当前字幕对象和文本缓存
            current_cue = {}
            text = ""
            # 将当前状态设置为等待下一个字幕的开始
            current_state = WAITING
            # 继续处理下一行
            continue
        
        # 如果当前状态为获取文本内容
        if current_state == GET_TEXT:
            # 如果是第一行文本
            if text_line == 0:
                # 直接赋值给文本缓存
                text += line
                text_line += 1
            else:
                # 对于非第一行文本，在文本末尾添加空格后再拼接
                text += " " + line
    
    # 最后如果当前状态仍为获取文本内容，则将最后一次收集的文本存入当前字幕对象
    if current_state == GET_TEXT:
        current_cue["Text"] = text
        # 将当前字幕对象添加到结果列表中
        result.append(current_cue)
    
    # 返回最终的字幕列表
    return result
#read srt file
def read_srt(input_file):
    try:
        # 尝试以 UTF-8 编码打开输入文件
        file1 = open(input_file, 'r', encoding='utf-8')
        # 读取文件所有行到列表中
        lines = file1.readlines()
        # 关闭文件
        file1.close()
    except Exception as error:
        # 捕获任何异常并打印错误信息
        print(error)
        # 退出程序
        exit()    
    # 返回读取的文件内容列表
    return lines


#parse ass
def parse_ass(input_file):     
    # 使用 UTF-8-SIG 编码打开 ASS 文件
    with open(input_file, encoding='utf-8-sig') as f:
        # 解析 ASS 文件内容
        s = ass.parse(f)
    # 初始化结果列表
    result = []
    # 遍历解析后的事件
    for line in s.events:
        # 将每个事件的时间码入、出和文本添加到结果列表中
        result.append({
        'TimecodeIn': line.start,
        'TimecodeOut': line.end,
        'Text': line.fields['Text']
        })
    # 返回解析后的结果列表
    return result


def convert(input_srt_file, output_folder, keep_japanese):
    # 确保输出文件夹存在，如果不存在则创建
    os.makedirs(output_folder, exist_ok=True)
    # 构建输出的 CSV 文件路径，文件名基于输入 SRT 文件名
    output_csv_file = output_folder + "/" + pathlib.Path(input_srt_file).stem + "." + 'csv'
    result = None
    # 根据输入 SRT 文件的后缀名选择处理方法
    if pathlib.Path(input_srt_file).suffix == '.srt':
        # 如果是 SRT 文件，读取其内容并进行内部处理
        lines = read_srt(input_srt_file)
        result = internalise(lines, keep_japanese)     
    elif pathlib.Path(input_srt_file).suffix == '.ass':
        # 如果是 ASS 文件，解析其内容
        result = parse_ass(input_srt_file)
    # 将处理后的结果渲染为 CSV 文件
    render_csv(result, output_csv_file)
    # 返回输出的 CSV 文件路径
    return output_csv_file


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='convert srt/ass to CSV',
        epilog='author:LengYue(https://github.com/zealot52099)'
    )
    # 添加 verbose 参数，类型为布尔值
    parser.add_argument("verbose", type=bool, action="store")
    # 添加 --srt_folder 参数，默认为 'srt_folder'，指定输出的 CSV 文件夹路径
    parser.add_argument('--srt_folder', default='srt_folder', type=str, required=True, help="folder to output .csv files")
    # 添加 --input_srt 参数，默认为 'input_srt'，指定输入的 .srt/.ass 文件路径
    parser.add_argument('--input_srt', default='input_srt', type=str, required=True, help="path of input .srt/.ass file")
    # 解析命令行参数
    args = parser.parse_args()
    # 打印命令行参数帮助信息
    parser.print_help()
    # 调用 srt2csv 函数处理输入参数
    srt2csv(args)
```