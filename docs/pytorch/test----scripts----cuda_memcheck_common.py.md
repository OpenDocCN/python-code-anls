# `.\pytorch\test\scripts\cuda_memcheck_common.py`

```
# 定义一个自定义异常类，用于在简单解析器无法解析报告时引发异常
class ParseError(Exception):
    """Whenever the simple parser is unable to parse the report, this exception will be raised"""
    pass


# 定义一个报告类，用于存储报告的文本、错误数量和错误列表
class Report:
    """A report is a container of errors, and a summary on how many errors are found"""

    def __init__(self, text, errors):
        # 初始化报告对象，text 是类似于 "ERROR SUMMARY: 1 error" 的文本
        self.text = text
        # 从文本中提取错误数量并转换为整数
        self.num_errors = int(text.strip().split()[2])
        # 错误列表
        self.errors = errors
        # 检查错误列表长度是否与报告中指定的错误数量匹配
        if len(errors) != self.num_errors:
            # 当错误数量不匹配时，根据特定情况进行处理
            if len(errors) == 10000 and self.num_errors > 10000:
                # 当报告中的错误超过 10000 个时，cuda-memcheck 只显示前 10000 个错误
                self.num_errors = 10000
            else:
                # 否则，抛出解析错误异常
                raise ParseError("Number of errors does not match")


# 定义一个错误类，用于表示报告中的每个错误信息及其回溯信息
class Error:
    """Each error is a section in the output of cuda-memcheck.
    Each error in the report has an error message and a backtrace. It looks like:

    ========= Program hit cudaErrorInvalidValue (error 1) due to "invalid argument" on CUDA API call to cudaGetLastError.
    =========     Saved host backtrace up to driver entry point at error
    =========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 [0x38c7b3]
    =========     Host Frame:/usr/local/cuda/lib64/libcudart.so.10.1 (cudaGetLastError + 0x163) [0x4c493]
    =========     Host Frame:/home/xgao/anaconda3/lib/python3.7/site-packages/torch/lib/libtorch.so [0x5b77a05]
    =========     Host Frame:/home/xgao/anaconda3/lib/python3.7/site-packages/torch/lib/libtorch.so [0x39d6d1d]
    =========     .....
    """

    def __init__(self, lines):
        # 错误消息为段落的第一行
        self.message = lines[0]
        # 回溯信息从第三行开始
        lines = lines[2:]
        # 回溯信息列表，每行都去除首尾空白字符
        self.stack = [l.strip() for l in lines]


# 定义一个解析函数，用于简单解析 cuda-memcheck 的报告
def parse(message):
    """A simple parser that parses the report of cuda-memcheck. This parser is meant to be simple
    and it only split the report into separate errors and a summary. Where each error is further
    splitted into error message and backtrace. No further details are parsed.

    A report contains multiple errors and a summary on how many errors are detected. It looks like:

    ========= CUDA-MEMCHECK
    ========= Program hit cudaErrorInvalidValue (error 1) due to "invalid argument" on CUDA API call to cudaPointerGetAttributes.
    =========     Saved host backtrace up to driver entry point at error
    =========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 [0x38c7b3]
    =========     Host Frame:/usr/local/cuda/lib64/libcudart.so.10.1 (cudaPointerGetAttributes + 0x1a9) [0x428b9]
    =========     Host Frame:/home/xgao/anaconda3/lib/python3.7/site-packages/torch/lib/libtorch.so [0x5b778a9]
    =========     .....
    =========
    ========= Program hit cudaErrorInvalidValue (error 1) due to "invalid argument" on CUDA API call to cudaGetLastError.
    """
    errors = []
    # 初始化一个空列表，用于存放解析出的错误信息

    HEAD = "========="
    # 定义错误信息头部的标识字符串

    headlen = len(HEAD)
    # 计算错误信息头部标识字符串的长度

    started = False
    # 标志变量，指示是否开始解析错误信息

    in_message = False
    # 标志变量，指示当前是否正在处理错误信息内容

    message_lines = []
    # 初始化一个空列表，用于存放当前正在处理的错误信息行的内容

    lines = message.splitlines()
    # 将输入的错误信息字符串按行分割，并存储在列表中

    for l in lines:
        # 遍历每一行错误信息

        if l == HEAD + " CUDA-MEMCHECK":
            # 如果当前行是以预定义的错误信息头部和" CUDA-MEMCHECK"结尾的
            started = True
            # 标志变量标记为已开始解析错误信息
            continue
        
        if not started or not l.startswith(HEAD):
            # 如果尚未开始解析或者当前行不以错误信息头部开头，则继续下一次循环
            continue
        
        l = l[headlen + 1 :]
        # 去除当前行开头的错误信息头部和空格后的内容，获取真正的错误信息内容

        if l.startswith("ERROR SUMMARY:"):
            # 如果当前行以"ERROR SUMMARY:"开头
            return Report(l, errors)
            # 返回一个报告对象，将当前行和解析出的错误列表作为参数

        if not in_message:
            # 如果当前不在处理错误信息内容的行
            in_message = True
            # 设置标志变量，表示开始处理错误信息内容
            message_lines = [l]
            # 将当前行作为错误信息内容的第一行

        elif l == "":
            # 如果当前行为空行
            errors.append(Error(message_lines))
            # 将当前累积的错误信息内容作为一个错误对象添加到错误列表中
            in_message = False
            # 设置标志变量，表示结束处理当前错误信息内容

        else:
            # 如果当前行不为空且在处理错误信息内容中
            message_lines.append(l)
            # 将当前行添加到错误信息内容列表中

    raise ParseError("No error summary found")
    # 如果未找到错误汇总信息，则抛出解析错误异常
```