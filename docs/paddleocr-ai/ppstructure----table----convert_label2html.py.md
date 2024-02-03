# `.\PaddleOCR\ppstructure\table\convert_label2html.py`

```
# 版权声明
# 2022年PaddlePaddle作者保留所有权利。
#
# 根据Apache许可证2.0版（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按“原样”分发，不附带任何明示或暗示的担保或条件。
# 请查看许可证以获取特定语言的权限和
# 许可证下的限制。
"""
将表标签转换为HTML
"""

import json
import argparse
from tqdm import tqdm

# 保存预测文本到临时文件
def save_pred_txt(key, val, tmp_file_path):
    with open(tmp_file_path, 'a+', encoding='utf-8') as f:
        f.write('{}\t{}\n'.format(key, val))

# 跳过特殊字符
def skip_char(text, sp_char_list):
    """
    跳过空单元格
    @param text: 单元格中的文本
    @param sp_char_list: 样式字符和特殊代码
    @return:
    """
    for sp_char in sp_char_list:
        text = text.replace(sp_char, '')
    return text

# 生成HTML代码
def gen_html(img):
    ''' 
    从图像的标记注释中格式化HTML代码
    '''
    html_code = img['html']['structure']['tokens'].copy()
    to_insert = [i for i, tag in enumerate(html_code) if tag in ('<td>', '>')]
    for i, cell in zip(to_insert[::-1], img['html']['cells'][::-1]):
        if cell['tokens']:
            text = ''.join(cell['tokens'])
            # 跳过空文本
            sp_char_list = ['<b>', '</b>', '\u2028', ' ', '<i>', '</i>']
            text_remove_style = skip_char(text, sp_char_list)
            if len(text_remove_style) == 0:
                continue
            html_code.insert(i + 1, text)
    html_code = ''.join(html_code)
    html_code = '<html><body><table>{}</table></body></html>'.format(html_code)
    return html_code

# 加载gt数据
def load_gt_data(gt_path):
    """
    加载gt
    @param gt_path:
    @return:
    """
    data_list = {}
    # 使用只读模式打开文件 gt_path，并将文件对象赋值给变量 f
    with open(gt_path, 'rb') as f:
        # 读取文件的所有行，并将其存储在列表 lines 中
        lines = f.readlines()
        # 遍历文件的每一行
        for line in tqdm(lines):
            # 将每行数据解码为 UTF-8 格式，并去除末尾的换行符
            data_line = line.decode('utf-8').strip("\n")
            # 将解码后的数据转换为 JSON 格式
            info = json.loads(data_line)
            # 将文件名作为键，解析后的信息作为值，存储在字典 data_list 中
            data_list[info['filename']] = info
    # 返回存储了文件名和信息的字典 data_list
    return data_list
# 将原始标签文件转换为 HTML 文件
def convert(origin_gt_path, save_path):
    """
    gen html from label file
    @param origin_gt_path: 原始标签文件路径
    @param save_path: 保存路径
    @return: None
    """
    # 加载原始标签数据
    data_dict = load_gt_data(origin_gt_path)
    # 遍历数据字典中的每个图片名和标签
    for img_name, gt in tqdm(data_dict.items()):
        # 生成 HTML 文件
        html = gen_html(gt)
        # 保存预测文本文件
        save_pred_txt(img_name, html, save_path)
    # 打印转换完成信息
    print('conver finish')


# 解析命令行参数
def parse_args():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="args for paddleserving")
    # 添加参数：原始标签路径
    parser.add_argument(
        "--ori_gt_path", type=str, required=True, help="label gt path")
    # 添加参数：保存路径
    parser.add_argument(
        "--save_path", type=str, required=True, help="path to save file")
    # 解析参数
    args = parser.parse_args()
    return args


# 主函数入口
if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    # 转换原始标签文件为 HTML 文件
    convert(args.ori_gt_path, args.save_path)
```