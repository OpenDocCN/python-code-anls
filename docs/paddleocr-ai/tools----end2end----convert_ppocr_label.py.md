# `.\PaddleOCR\tools\end2end\convert_ppocr_label.py`

```
# 版权声明和许可证信息
# 本代码版权归 PaddlePaddle 作者所有，保留所有权利
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 导入所需的库
import numpy as np
import json
import os

# 将多边形数据转换为字符串格式
def poly_to_string(poly):
    # 如果多边形数据的维度大于1，则将其展平为一维数组
    if len(poly.shape) > 1:
        poly = np.array(poly).flatten()

    # 将多边形数据转换为以制表符分隔的字符串
    string = "\t".join(str(i) for i in poly)
    return string

# 将标签数据转换为指定格式
def convert_label(label_dir, mode="gt", save_dir="./save_results/"):
    # 检查标签文件夹是否存在
    if not os.path.exists(label_dir):
        raise ValueError(f"The file {label_dir} does not exist!")

    # 断言标签文件夹路径不等于保存结果文件夹路径
    assert label_dir != save_dir, "hahahhaha"

    # 打开标签文件，读取数据
    label_file = open(label_dir, 'r')
    data = label_file.readlines()

    # 创建存储转换后标签数据的字典
    gt_dict = {}
    # 遍历数据列表中的每一行
    for line in data:
        try:
            # 尝试使用制表符分割行数据
            tmp = line.split('\t')
            # 断言分割后的数据长度为2
            assert len(tmp) == 2, ""
        except:
            # 如果出现异常，则使用空格分割行数据
            tmp = line.strip().split('    ')

        # 初始化 gt_lists 列表
        gt_lists = []

        # 如果分割后的第一个元素不为空
        if tmp[0].split('/')[0] is not None:
            # 提取图片路径和注释信息
            img_path = tmp[0]
            anno = json.loads(tmp[1])
            gt_collect = []
            # 遍历注释信息中的每个字典
            for dic in anno:
                # 提取文本信息
                txt = dic['transcription']
                # 如果字典中包含 'score' 键且其值小于0.5，则跳过当前字典
                if 'score' in dic and float(dic['score']) < 0.5:
                    continue
                # 替换特殊字符
                if u'\u3000' in txt: txt = txt.replace(u'\u3000', u' ')
                # 将多余的空格替换为空
                poly = np.array(dic['points']).flatten()
                # 根据文本内容判断标签类型
                if txt == "###":
                    txt_tag = 1  ## ignore 1
                else:
                    txt_tag = 0
                # 根据模式生成标签
                if mode == "gt":
                    gt_label = poly_to_string(poly) + "\t" + str(
                        txt_tag) + "\t" + txt + "\n"
                else:
                    gt_label = poly_to_string(poly) + "\t" + txt + "\n"

                # 将标签添加到 gt_lists 列表中
                gt_lists.append(gt_label)

            # 将图片路径和对应的标签列表添加到 gt_dict 字典中
            gt_dict[img_path] = gt_lists
        else:
            # 如果分割后的第一个元素为空，则跳过当前行
            continue

    # 如果保存目录不存在，则创建目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 遍历 gt_dict 字典中的每个图片名
    for img_name in gt_dict.keys():
        # 提取图片名
        save_name = img_name.split("/")[-1]
        # 构建保存文件路径
        save_file = os.path.join(save_dir, save_name + ".txt")
        # 将标签写入文件
        with open(save_file, "w") as f:
            f.writelines(gt_dict[img_name])

    # 打印保存标签的信息
    print("The convert label saved in {}".format(save_dir))
# 解析命令行参数
def parse_args():
    # 导入 argparse 模块
    import argparse
    # 创建 ArgumentParser 对象，设置描述信息
    parser = argparse.ArgumentParser(description="args")
    # 添加命令行参数 --label_path，类型为字符串，必须提供
    parser.add_argument("--label_path", type=str, required=True)
    # 添加命令行参数 --save_folder，类型为字符串，必须提供
    parser.add_argument("--save_folder", type=str, required=True)
    # 添加命令行参数 --mode，类型为字符串，默认值为 False
    parser.add_argument("--mode", type=str, default=False)
    # 解析命令行参数
    args = parser.parse_args()
    # 返回解析后的参数对象
    return args

# 主程序入口
if __name__ == "__main__":
    # 解析命令行参数，获取参数对象
    args = parse_args()
    # 调用 convert_label 函数，传入参数对象中的 label_path、mode 和 save_folder
    convert_label(args.label_path, args.mode, args.save_folder)
```