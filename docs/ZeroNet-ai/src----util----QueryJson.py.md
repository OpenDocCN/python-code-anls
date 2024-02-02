# `ZeroNet\src\util\QueryJson.py`

```py
# 导入 json 模块
import json
# 导入 re 模块
import re
# 导入 os 模块

# 查询文件内容
def queryFile(file_path, filter_path, filter_key=None, filter_val=None):
    # 存储查询结果
    back = []
    # 读取 JSON 文件内容
    data = json.load(open(file_path))
    # 如果过滤路径为空，则返回整个数据
    if filter_path == ['']:
        return [data]
    # 遍历过滤路径，逐层获取数据
    for key in filter_path:  # Get to the point
        data = data.get(key)
        # 如果数据为空，则返回空
        if not data:
            return

    # 如果数据是列表
    if type(data) == list:
        # 遍历列表中的每一行数据
        for row in data:
            # 如果有过滤值，则根据过滤键和值进行过滤
            if filter_val:  # Filter by value
                if row[filter_key] == filter_val:
                    back.append(row)
            # 如果没有过滤值，则直接添加到结果中
            else:
                back.append(row)
    # 如果数据不是列表
    else:
        # 将数据封装成字典格式
        back.append({"value": data})

    # 返回查询结果
    return back


# 在 JSON 文件中查找内容
# 返回格式为列表，每个元素为一个字典
def query(path_pattern, filter):
    # 如果过滤条件中包含 "="，则根据值进行过滤
    if "=" in filter:  # Filter by value
        filter_path, filter_val = filter.split("=")
        filter_path = filter_path.split(".")
        filter_key = filter_path.pop()  # 最后一个元素是键
        filter_val = int(filter_val)
    # 如果没有 "="，则不进行过滤
    else:  # No filter
        filter_path = filter
        filter_path = filter_path.split(".")
        filter_key = None
        filter_val = None

    # 如果路径中包含 "/*/"，则进行通配符搜索
    if "/*/" in path_pattern:  # Wildcard search
        root_dir, file_pattern = path_pattern.replace("\\", "/").split("/*/")
    # 如果没有 "/*/"，则不进行通配符搜索
    else:  # No wildcard
        root_dir, file_pattern = re.match("(.*)/(.*?)$", path_pattern.replace("\\", "/")).groups()
    # 遍历指定目录下的所有文件和子目录
    for root, dirs, files in os.walk(root_dir, topdown=False):
        # 将路径中的反斜杠替换为斜杠
        root = root.replace("\\", "/")
        # 获取当前文件所在的相对路径
        inner_path = root.replace(root_dir, "").strip("/")
        # 遍历当前目录下的所有文件
        for file_name in files:
            # 如果文件名不匹配指定的文件模式，则跳过当前文件
            if file_pattern != file_name:
                continue
            # 尝试查询文件内容，并根据过滤条件进行筛选
            try:
                res = queryFile(root + "/" + file_name, filter_path, filter_key, filter_val)
                # 如果查询结果为空，则跳过当前文件
                if not res:
                    continue
            # 捕获异常，处理 JSON 加载错误
            except Exception:  # Json load error
                continue
            # 遍历查询结果中的每一行数据
            for row in res:
                # 将当前文件的相对路径添加到每一行数据中
                row["inner_path"] = inner_path
                # 通过生成器返回每一行数据
                yield row
# 如果当前模块被直接执行，而非被导入其他模块，则执行以下代码
if __name__ == "__main__":
    # 遍历查询结果列表中的每一行
    for row in list(query("../../data/12Hw8rTgzrNo4DSh2AkqwPRqDyTticwJyH/data/users/*/data.json", "")):
        # 打印每一行的内容
        print(row)
```