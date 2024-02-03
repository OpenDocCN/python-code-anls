# `.\PaddleOCR\ppocr\utils\gen_label.py`

```py
# 版权声明
#
# 版权所有 (c) 2020 PaddlePaddle 作者。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于“按原样”基础分发的，没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言下的权限和限制。
#
# 导入所需的库
import os
import argparse
import json

# 定义生成分类任务标签的函数
def gen_rec_label(input_path, out_label):
    # 打开输出标签文件，准备写入
    with open(out_label, 'w') as out_file:
        # 打开输入文件，准备读取
        with open(input_path, 'r') as f:
            # 逐行读取输入文件内容
            for line in f.readlines():
                # 去除换行符，去除空格，按逗号分割
                tmp = line.strip('\n').replace(" ", "").split(',')
                # 获取图片路径和标签
                img_path, label = tmp[0], tmp[1]
                # 去除标签中的双引号
                label = label.replace("\"", "")
                # 写入图片路径和标签到输出文件
                out_file.write(img_path + '\t' + label + '\n')

# 定义生成检测任务标签的函数
def gen_det_label(root_path, input_dir, out_label):
    # 打开输出标签文件，以写入模式打开
    with open(out_label, 'w') as out_file:
        # 遍历输入目录中的所有标签文件
        for label_file in os.listdir(input_dir):
            # 构建图像路径，根据标签文件名获取图像文件名
            img_path = os.path.join(root_path, label_file[3:-4] + ".jpg")
            # 初始化标签列表
            label = []
            # 打开标签文件，以只读模式打开，指定编码为utf-8-sig
            with open(
                    os.path.join(input_dir, label_file), 'r',
                    encoding='utf-8-sig') as f:
                # 逐行读取标签文件内容
                for line in f.readlines():
                    # 处理每行数据，去除换行符和特殊字符，按逗号分割
                    tmp = line.strip("\n\r").replace("\xef\xbb\xbf",
                                                     "").split(',')
                    # 提取坐标点信息
                    points = tmp[:8]
                    s = []
                    # 将坐标点转换为整数列表
                    for i in range(0, len(points), 2):
                        b = points[i:i + 2]
                        b = [int(t) for t in b]
                        s.append(b)
                    # 构建包含转录和坐标点的字典
                    result = {"transcription": tmp[8], "points": s}
                    # 将字典添加到标签列表中
                    label.append(result)

            # 将图像路径和标签列表写入输出文件
            out_file.write(img_path + '\t' + json.dumps(
                label, ensure_ascii=False) + '\n')
# 如果当前脚本被作为主程序执行
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加一个参数--mode，类型为字符串，默认为"rec"，帮助信息为'Generate rec_label or det_label, can be set rec or det'
    parser.add_argument(
        '--mode',
        type=str,
        default="rec",
        help='Generate rec_label or det_label, can be set rec or det')
    # 添加一个参数--root_path，类型为字符串，默认为"."，帮助信息为'The root directory of images.Only takes effect when mode=det '
    parser.add_argument(
        '--root_path',
        type=str,
        default=".",
        help='The root directory of images.Only takes effect when mode=det ')
    # 添加一个参数--input_path，类型为字符串，默认为"."，帮助信息为'Input_label or input path to be converted'
    parser.add_argument(
        '--input_path',
        type=str,
        default=".",
        help='Input_label or input path to be converted')
    # 添加一个参数--output_label，类型为字符串，默认为"out_label.txt"，帮助信息为'Output file name'
    parser.add_argument(
        '--output_label',
        type=str,
        default="out_label.txt",
        help='Output file name')

    # 解析命令行参数
    args = parser.parse_args()
    # 如果参数中的mode为"rec"
    if args.mode == "rec":
        # 打印信息"Generate rec label"
        print("Generate rec label")
        # 调用gen_rec_label函数，传入参数input_path和output_label
        gen_rec_label(args.input_path, args.output_label)
    # 如果参数中的mode为"det"
    elif args.mode == "det":
        # 调用gen_det_label函数，传入参数root_path、input_path和output_label
        gen_det_label(args.root_path, args.input_path, args.output_label)
```