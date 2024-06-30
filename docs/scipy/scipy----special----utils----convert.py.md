# `D:\src\scipysrc\scipy\scipy\special\utils\convert.py`

```
# 这个脚本用于将 BOOST 特殊函数测试数据解析为能够轻松导入 numpy 的格式。

# 导入 re 和 os 模块
import re
import os

# 数据存放位置（目录将会被创建）
DATA_DIR = 'scipy/special/tests/data/boost'
# BOOST 源数据位置
BOOST_SRC = "boostmath/test"

# C++ 注释的正则表达式
CXX_COMMENT = re.compile(r'^\s+//')
# 匹配 BOOST 数据的正则表达式
DATA_REGEX = re.compile(r'^\s*/*\{*\s*SC_')
# 匹配数据项的正则表达式
ITEM_REGEX = re.compile(r'[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?')
# 匹配头部声明的正则表达式
HEADER_REGEX = re.compile(
    r'const boost::array\<boost::array\<.*, (\d+)\>, (\d+)\> ([a-zA-Z_\d]+)')

# 需要忽略的文件模式列表
IGNORE_PATTERNS = [
    # 使用 ldexp 和类型转换
    "hypergeometric_1F1_big_double_limited.ipp",
    "hypergeometric_1F1_big_unsolved.ipp",

    # 使用 numeric_limits 和三元操作符
    "beta_small_data.ipp",

    # 不包含任何数据
    "almost_equal.ipp",

    # 没有导数函数的文件
    "bessel_y01_prime_data.ipp",
    "bessel_yn_prime_data.ipp",
    "sph_bessel_prime_data.ipp",
    "sph_neumann_prime_data.ipp",

    # 不被 scipy special 测试所需要的数据文件
    "ibeta_derivative_",
    r"ellint_d2?_",
    "jacobi_",
    "heuman_lambda_",
    "hypergeometric_",
    "nct_",
    r".*gammap1m1_",
    "trig_",
    "powm1_data.ipp",
]


def _raw_data(line):
    # 按逗号分割行，提取数据项列表
    items = line.split(',')
    l = []
    for item in items:
        # 使用正则表达式匹配数据项
        m = ITEM_REGEX.search(item)
        if m:
            q = m.group(0)
            l.append(q)
    return l


def parse_ipp_file(filename):
    # 打印文件名
    print(filename)
    # 打开文件并读取所有行
    with open(filename) as a:
        lines = a.readlines()
    data = {}
    i = 0
    while (i < len(lines)):
        line = lines[i]
        m = HEADER_REGEX.search(line)
        if m:
            # 提取数组维度信息
            d = int(m.group(1))
            n = int(m.group(2))
            print(f"d = {d}, n = {n}")
            cdata = []
            i += 1
            line = lines[i]
            # 跳过 C++ 注释
            while CXX_COMMENT.match(line):
                i += 1
                line = lines[i]
            # 匹配数据行
            while DATA_REGEX.match(line):
                cdata.append(_raw_data(line))
                i += 1
                line = lines[i]
                # 跳过 C++ 注释
                while CXX_COMMENT.match(line):
                    i += 1
                    line = lines[i]
            # 检查数据数量是否正确
            if not len(cdata) == n:
                raise ValueError(f"parsed data: {len(cdata)}, expected {n}")
            # 将数据存入字典
            data[m.group(3)] = cdata
        else:
            i += 1

    return data


def dump_dataset(filename, data):
    # 打开文件以写入数据
    fid = open(filename, 'w')
    try:
        # 将数据写入文件
        for line in data:
            fid.write("%s\n" % " ".join(line))
    finally:
        # 关闭文件
        fid.close()


def dump_datasets(filename):
    # 提取文件名和扩展名
    base, ext = os.path.splitext(os.path.basename(filename))
    base += '_%s' % ext[1:]
    # 构建数据目录路径
    datadir = os.path.join(DATA_DIR, base)
    # 创建目录
    os.makedirs(datadir)
    # 解析 IPP 文件并返回数据集
    datasets = parse_ipp_file(filename)
    # 遍历字典 datasets，其中 k 是键，d 是对应的值
    for k, d in datasets.items():
        # 打印当前键 k 和值 d 的长度
        print(k, len(d))
        # 根据当前键 k，构建数据文件的完整路径，以 '.txt' 结尾
        dfilename = os.path.join(datadir, k) + '.txt'
        # 调用 dump_dataset 函数，将数据 d 存储到文件 dfilename 中
        dump_dataset(dfilename, d)
if __name__ == '__main__':
    # 如果当前脚本作为主程序运行，则执行以下代码块
    for filename in sorted(os.listdir(BOOST_SRC)):
        # 遍历 BOOST_SRC 目录下的所有文件名，并按字母顺序排序
        # 注意：忽略以 .ipp 结尾的文件 (例如 powm1_sqrtp1m1_test.hpp)
        if filename.endswith(".ipp"):
            # 检查文件名是否匹配任何忽略模式中的正则表达式
            if any(re.match(pattern, filename) for pattern in IGNORE_PATTERNS):
                # 如果文件名匹配任何忽略模式，则跳过当前循环，继续处理下一个文件
                continue

            # 构建完整的文件路径
            path = os.path.join(BOOST_SRC, filename)
            # 打印文件路径，作为分隔行
            print(f"================= {path} ===============")
            # 调用 dump_datasets 函数处理当前文件
            dump_datasets(path)
```