# `.\pytorch\tools\vscode_settings.py`

```py
#!/usr/bin/env python3
# 指定 Python 解释器路径，使脚本可以在环境中独立运行

from pathlib import Path

try:
    # 尝试导入 json5 库，用于解析 JSON 格式的配置文件
    import json5 as json  # type: ignore[import]

    HAS_JSON5 = True
except ImportError:
    # 如果导入失败，则回退到标准 json 库
    import json  # type: ignore[no-redef]

    HAS_JSON5 = False


# 定义根文件夹路径为当前脚本文件的上级目录
ROOT_FOLDER = Path(__file__).absolute().parent.parent
# 定义 VS Code 的配置文件夹路径
VSCODE_FOLDER = ROOT_FOLDER / ".vscode"
# 定义推荐的设置文件路径
RECOMMENDED_SETTINGS = VSCODE_FOLDER / "settings_recommended.json"
# 定义当前用户设置文件路径
SETTINGS = VSCODE_FOLDER / "settings.json"


# 定义函数 deep_update，用于递归更新字典的设置
def deep_update(d: dict, u: dict) -> dict:  # type: ignore[type-arg]
    for k, v in u.items():
        if isinstance(v, dict):
            # 如果值是字典，则递归更新
            d[k] = deep_update(d.get(k, {}), v)
        elif isinstance(v, list):
            # 如果值是列表，则合并列表
            d[k] = d.get(k, []) + v
        else:
            # 否则直接赋值
            d[k] = v
    return d


# 定义主函数 main，用于读取和更新 VS Code 的设置文件
def main() -> None:
    # 加载推荐的设置文件内容为字典
    recommended_settings = json.loads(RECOMMENDED_SETTINGS.read_text())
    try:
        # 尝试读取当前用户设置文件的文本内容
        current_settings_text = SETTINGS.read_text()
    except FileNotFoundError:
        # 如果找不到当前用户设置文件，则将其内容初始化为空字典字符串
        current_settings_text = "{}"

    try:
        # 尝试解析当前用户设置文件的 JSON 内容为字典
        current_settings = json.loads(current_settings_text)
    except ValueError as ex:  # json.JSONDecodeError is a subclass of ValueError
        if HAS_JSON5:
            # 如果使用 json5 库解析失败，则抛出异常提示解析失败
            raise SystemExit("Failed to parse .vscode/settings.json.") from ex
        raise SystemExit(
            # 如果使用标准 json 库解析失败，则提示可能存在注释或尾随逗号，并建议安装 json5 库
            "Failed to parse .vscode/settings.json. "
            "Maybe it contains comments or trailing commas. "
            "Try `pip install json5` to install an extended JSON parser."
        ) from ex

    # 深度更新当前用户设置文件的内容
    settings = deep_update(current_settings, recommended_settings)

    # 将更新后的设置内容写入到设置文件中，格式化为带缩进的 JSON 格式，并添加尾随换行符
    SETTINGS.write_text(
        json.dumps(
            settings,
            indent=4,
        )
        + "\n",  # 添加一个尾随换行符
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
```