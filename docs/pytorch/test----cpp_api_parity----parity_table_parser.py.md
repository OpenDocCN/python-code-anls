# `.\pytorch\test\cpp_api_parity\parity_table_parser.py`

```
# 导入命名元组模块
from collections import namedtuple

# 定义一个命名元组 ParityStatus，表示实现和文档的匹配状态
ParityStatus = namedtuple("ParityStatus", ["has_impl_parity", "has_doc_parity"])

"""
This function expects the parity tracker Markdown file to have the following format:


## package1_name

API | Implementation Parity | Doc Parity
------------- | ------------- | -------------
API_Name|No|No
...

## package2_name

API | Implementation Parity | Doc Parity
------------- | ------------- | -------------
API_Name|No|No
...


The returned dict has the following format:


Dict[package_name]
    -> Dict[api_name]
        -> ParityStatus

"""

# 解析给定的 Parity Tracker Markdown 文件，返回一个嵌套字典表示各包和其 API 的匹配状态
def parse_parity_tracker_table(file_path):
    
    # 内部函数：解析匹配状态选择字符串
    def parse_parity_choice(str):
        if str in ["Yes", "No"]:
            return str == "Yes"
        else:
            raise RuntimeError(
                f'{str} is not a supported parity choice. The valid choices are "Yes" and "No".'
            )

    # 创建空字典用于存储解析后的匹配状态数据
    parity_tracker_dict = {}

    # 打开并读取指定路径的文件
    with open(file_path) as f:
        all_text = f.read()
        # 按照分隔符 '##' 将文本分割为不同的包块
        packages = all_text.split("##")
        
        # 遍历每个包块
        for package in packages[1:]:
            # 对每个包块的行进行处理，去除空格并剔除空行
            lines = [line.strip() for line in package.split("\n") if line.strip() != ""]
            # 第一行为包名
            package_name = lines[0]
            # 检查是否重复出现相同的包名，若是则引发运行时错误
            if package_name in parity_tracker_dict:
                raise RuntimeError(
                    f"Duplicated package name `{package_name}` found in {file_path}"
                )
            else:
                # 否则在字典中创建新的包名条目
                parity_tracker_dict[package_name] = {}
            
            # 处理每个 API 的状态行
            for api_status in lines[3:]:
                # 拆分每行中的 API 名称、实现匹配状态和文档匹配状态
                api_name, has_impl_parity_str, has_doc_parity_str = (
                    x.strip() for x in api_status.split("|")
                )
                # 将 API 名称和其对应的 ParityStatus 存入字典
                parity_tracker_dict[package_name][api_name] = ParityStatus(
                    has_impl_parity=parse_parity_choice(has_impl_parity_str),
                    has_doc_parity=parse_parity_choice(has_doc_parity_str),
                )

    # 返回最终的匹配状态字典
    return parity_tracker_dict
```