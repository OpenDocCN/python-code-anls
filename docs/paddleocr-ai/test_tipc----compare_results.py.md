# `.\PaddleOCR\test_tipc\compare_results.py`

```
import numpy as np
import os
import subprocess
import json
import argparse
import glob

# 初始化命令行参数解析器
def init_args():
    parser = argparse.ArgumentParser()
    # 参数用于测试 assert allclose
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--gt_file", type=str, default="")
    parser.add_argument("--log_file", type=str, default="")
    parser.add_argument("--precision", type=str, default="fp32")
    return parser

# 解析命令行参数
def parse_args():
    parser = init_args()
    return parser.parse_args()

# 运行 shell 命令
def run_shell_command(cmd):
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate()

    if p.returncode == 0:
        return out.decode('utf-8')
    else:
        return None

# 通过名称从日志中解析结果
def parser_results_from_log_by_name(log_path, names_list):
    if not os.path.exists(log_path):
        raise ValueError("The log file {} does not exists!".format(log_path))

    if names_list is None or len(names_list) < 1:
        return []

    parser_results = {}
    for name in names_list:
        cmd = "grep {} {}".format(name, log_path)
        outs = run_shell_command(cmd)
        outs = outs.split("\n")[0]
        result = outs.split("{}".format(name))[-1]
        try:
            result = json.loads(result)
        except:
            result = np.array([int(r) for r in result.split()]).reshape(-1, 4)
        parser_results[name] = result
    return parser_results

# 从文件中加载 ground truth
def load_gt_from_file(gt_file):
    if not os.path.exists(gt_file):
        raise ValueError("The log file {} does not exists!".format(gt_file))
    with open(gt_file, 'r') as f:
        data = f.readlines()
        f.close()
    parser_gt = {}
    # 遍历数据中的每一行
    for line in data:
        # 从每一行中提取图像名称和结果，去除换行符并按制表符分割
        image_name, result = line.strip("\n").split("\t")
        # 从图像名称中提取文件名部分
        image_name = image_name.split('/')[-1]
        # 尝试将结果解析为 JSON 格式，如果失败则将结果解析为包含整数的数组并重新组织形状
        try:
            result = json.loads(result)
        except:
            result = np.array([int(r) for r in result.split()]).reshape(-1, 4)
        # 将图像名称和结果存储到名为 parser_gt 的字典中
        parser_gt[image_name] = result
    # 返回存储了图像名称和结果的字典
    return parser_gt
# 从给定的 ground truth 文件中加载数据，返回一个包含不同数据集的字典
def load_gt_from_txts(gt_file):
    # 使用 glob 模块获取所有符合条件的文件路径
    gt_list = glob.glob(gt_file)
    # 创建一个空字典用于存储不同数据集
    gt_collection = {}
    # 遍历每个 ground truth 文件
    for gt_f in gt_list:
        # 从文件中加载 ground truth 数据，返回一个字典
        gt_dict = load_gt_from_file(gt_f)
        # 获取文件名
        basename = os.path.basename(gt_f)
        # 根据文件名中的关键词将数据存储到不同的数据集中
        if "fp32" in basename:
            gt_collection["fp32"] = [gt_dict, gt_f]
        elif "fp16" in basename:
            gt_collection["fp16"] = [gt_dict, gt_f]
        elif "int8" in basename:
            gt_collection["int8"] = [gt_dict, gt_f]
        else:
            continue
    # 返回包含不同数据集的字典
    return gt_collection


# 从给定的日志文件中收集预测数据，返回一个包含不同预测结果的字典
def collect_predict_from_logs(log_path, key_list):
    # 使用 glob 模块获取所有符合条件的文件路径
    log_list = glob.glob(log_path)
    # 创建一个空字典用于存储不同预测结果
    pred_collection = {}
    # 遍历每个日志文件
    for log_f in log_list:
        # 从日志文件中解析出预测结果，返回一个字典
        pred_dict = parser_results_from_log_by_name(log_f, key_list)
        # 获取文件名作为字典的键
        key = os.path.basename(log_f)
        # 将预测结果存储到字典中
        pred_collection[key] = pred_dict

    # 返回包含不同预测结果的字典
    return pred_collection


# 对比两个字典中的值是否在指定的误差范围内
def testing_assert_allclose(dict_x, dict_y, atol=1e-7, rtol=1e-7):
    # 遍历第一个字典的键
    for k in dict_x:
        # 使用 numpy.testing 模块对比两个数组是否在指定的误差范围内
        np.testing.assert_allclose(
            np.array(dict_x[k]), np.array(dict_y[k]), atol=atol, rtol=rtol)


if __name__ == "__main__":
    # 用法提示
    # python3.7 tests/compare_results.py --gt_file=./tests/results/*.txt  --log_file=./tests/output/infer_*.log

    # 解析命令行参数
    args = parse_args()

    # 加载 ground truth 数据集
    gt_collection = load_gt_from_txts(args.gt_file)
    # 获取 fp32 数据集的键列表
    key_list = gt_collection["fp32"][0].keys()

    # 收集预测结果数据集
    pred_collection = collect_predict_from_logs(args.log_file, key_list)
    # 遍历预测结果集合中的文件名
    for filename in pred_collection.keys():
        # 如果文件名中包含"fp32"，则使用"fp32"对应的ground truth数据和文件名
        if "fp32" in filename:
            gt_dict, gt_filename = gt_collection["fp32"]
        # 如果文件名中包含"fp16"，则使用"fp16"对应的ground truth数据和文件名
        elif "fp16" in filename:
            gt_dict, gt_filename = gt_collection["fp16"]
        # 如果文件名中包含"int8"，则使用"int8"对应的ground truth数据和文件名
        elif "int8" in filename:
            gt_dict, gt_filename = gt_collection["int8"]
        # 如果文件名不符合以上条件，则跳过当前循环
        else:
            continue
        # 获取当前文件名对应的预测结果数据
        pred_dict = pred_collection[filename]

        try:
            # 使用自定义的测试函数测试预测结果和ground truth数据是否一致
            testing_assert_allclose(
                gt_dict, pred_dict, atol=args.atol, rtol=args.rtol)
            # 打印一致性信息
            print(
                "Assert allclose passed! The results of {} and {} are consistent!".
                format(filename, gt_filename))
        # 捕获异常
        except Exception as E:
            # 打印异常信息
            print(E)
            # 抛出数值错误，说明结果不一致
            raise ValueError(
                "The results of {} and the results of {} are inconsistent!".
                format(filename, gt_filename))
```