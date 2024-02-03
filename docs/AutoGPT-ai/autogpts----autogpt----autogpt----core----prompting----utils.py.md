# `.\AutoGPT\autogpts\autogpt\autogpt\core\prompting\utils.py`

```py
import ast  # 导入 ast 模块，用于处理抽象语法树
import json  # 导入 json 模块，用于处理 JSON 数据


def to_numbered_list(
    items: list[str], no_items_response: str = "", **template_args
) -> str:
    # 如果 items 列表不为空
    if items:
        # 使用列表推导式生成带编号的列表项字符串，并使用换行符连接成一个字符串
        return "\n".join(
            f"{i+1}. {item.format(**template_args)}" for i, item in enumerate(items)
        )
    else:
        # 如果 items 列表为空，则返回指定的 no_items_response
        return no_items_response


def json_loads(json_str: str):
    # TODO: this is a hack function for now. We'll see what errors show up in testing.
    #   Can hopefully just replace with a call to ast.literal_eval.
    # Can't use json.loads because the function API still sometimes returns json strings
    #   with minor issues like trailing commas.
    try:
        # 从 json_str 中提取出 JSON 格式的子字符串
        json_str = json_str[json_str.index("{") : json_str.rindex("}") + 1]
        # 使用 ast.literal_eval 将 JSON 字符串转换为 Python 对象
        return ast.literal_eval(json_str)
    except json.decoder.JSONDecodeError as e:
        try:
            # 如果出现 JSONDecodeError，尝试使用 ast.literal_eval
            print(f"json decode error {e}. trying literal eval")
            return ast.literal_eval(json_str)
        except Exception:
            # 如果出现异常，打开调试器
            breakpoint()
```