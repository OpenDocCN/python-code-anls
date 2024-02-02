# `basic-computer-games\00_Utilities\markdown_todo.py`

```py
# 导入 os 模块
import os
# 导入 Dict 和 List 类型提示
from typing import Dict, List

# 判断指定语言的实现是否存在
def has_implementation(lang: str, file_list: List[str], subdir_list: List[str]) -> bool:
    # 如果语言是 C#，则判断文件列表中是否存在以 .cs 结尾的文件
    if lang == "csharp":
        return any(file.endswith(".cs") for file in file_list)
    # 如果语言是 VB.NET，则判断文件列表中是否存在以 .vb 结尾的文件
    elif lang == "vbnet":
        return any(file.endswith(".vb") for file in file_list)
    # 其他语言则判断文件列表长度是否大于1或子目录列表长度是否大于0
    else:
        return len(file_list) > 1 or len(subdir_list) > 0

# 获取数据
def get_data(checklist_orig: List[str], root_dir: str = "..") -> List[List[str]]:
    """
    获取数据

    Parameters
    ----------
    root_dir : str
        要开始的根目录。
    """
    # 创建语言位置字典，键为语言，值为在检查列表中的位置
    lang_pos: Dict[str, int] = {
        lang: i for i, lang in enumerate(checklist_orig[1:], start=1)
    }
    # 已完成的字符串列表
    strings_done: List[List[str]] = []

    # 忽略的文件夹列表
    ignore_folders = [
        ".git",
        "00_Utilities",
        ".github",
        ".mypy_cache",
        ".pytest_cache",
        "00_Alternate_Languages",
        "00_Common",
        "buildJvm",
        "htmlcov",
    ]

    # 上一个游戏
    prev_game = ""

    # 空白框列表，用于初始化检查列表
    empty_boxes = ["⬜️" for _ in checklist_orig]
    # 复制空白框列表，作为检查列表的初始状态
    checklist = empty_boxes[:]
    # 遍历指定目录下的所有文件和子目录
    for dir_name, subdir_list, file_list in sorted(os.walk(root_dir)):
        # 将目录名按照路径分隔符分割，获取游戏和语言信息
        split_dir = dir_name.split(os.path.sep)

        # 如果目录名长度为2且游戏不在忽略列表中
        if len(split_dir) == 2 and split_dir[1] not in ignore_folders:
            # 如果前一个游戏为空
            if prev_game == "":
                # 设置前一个游戏为当前游戏，并将游戏信息添加到检查列表
                prev_game = split_dir[1]
                checklist[0] = f"{split_dir[1]:<30}"

            # 如果前一个游戏不等于当前游戏
            if prev_game != split_dir[1]:
                # 将检查列表添加到已完成的字符串列表中，并重新初始化检查列表
                strings_done.append(checklist)
                checklist = [
                    f"{f'[{split_dir[1]}](../{split_dir[1]})':<30}",
                ] + empty_boxes[1:]
                prev_game = split_dir[1]
        # 如果目录名长度为3且第二个元素不是".git"，且第三个元素在语言位置字典中
        elif (
            len(split_dir) == 3 and split_dir[1] != ".git" and split_dir[2] in lang_pos
        ):
            # 根据语言位置字典中的位置判断是否有实现，设置对应的标志
            out = (
                "✅"
                if has_implementation(split_dir[2], file_list, subdir_list)
                else "⬜️"
            )
            # 如果语言不在语言位置字典中或者位置超出检查列表长度
            if split_dir[2] not in lang_pos or lang_pos[split_dir[2]] >= len(checklist):
                # 输出找不到语言的信息，并设置对应位置的标志为未完成
                print(f"Could not find {split_dir[2]}: {dir_name}")
                checklist[lang_pos[split_dir[2]]] = "⬜️"
                continue
            # 设置检查列表中对应语言位置的标志
            checklist[lang_pos[split_dir[2]]] = out
    # 返回已完成的字符串列表
    return strings_done
# 定义一个函数，用于将数据写入文件
def write_file(path: str, languages: List[str], strings_done: List[List[str]]) -> None:
    # 创建一个由破折号组成的数组，长度为语言数量加一
    dashes_arr = ["---"] * (len(languages) + 1)
    # 将数组的第一个元素替换为30个破折号
    dashes_arr[0] = "-" * 30
    # 将破折号数组连接成字符串，用竖线分隔
    dashes = " | ".join(dashes_arr)
    # 创建写入字符串的标题行
    write_string = f"# TODO list\n {'game':<30}| {' | '.join(languages)}\n{dashes}\n"
    # 对已完成的字符串列表进行排序，并将每个列表元素连接成字符串
    sorted_strings = list(
        map(lambda l: " | ".join(l) + "\n", sorted(strings_done, key=lambda x: x[0]))
    )
    # 将排序后的字符串列表连接到写入字符串中
    write_string += "".join(sorted_strings)
    # 将破折号连接到写入字符串中
    write_string += f"{dashes}\n"
    # 创建语言索引范围
    language_indices = range(1, len(languages) + 1)
    # 获取已完成的游戏数量
    nb_games = len(strings_done)
    # 将游戏数量和每种语言已完成的数量连接到写入字符串中
    write_string += (
        f"{f'Sum of {nb_games}':<30} | "
        + " | ".join(
            [
                f"{sum(row[lang] == '✅' for row in strings_done)}"
                for lang in language_indices
            ]
        )
        + "\n"
    )

    # 打开文件并写入字符串
    with open(path, "w", encoding="utf-8") as f:
        f.write(write_string)


# 如果作为主程序运行
if __name__ == "__main__":
    # 定义语言字典
    languages = {
        "csharp": "C#",
        "java": "Java",
        "javascript": "JS",
        "kotlin": "Kotlin",
        "lua": "Lua",
        "perl": "Perl",
        "python": "Python",
        "ruby": "Ruby",
        "rust": "Rust",
        "vbnet": "VB.NET",
    }
    # 获取已完成的数据
    strings_done = get_data(["game"] + list(languages.keys()))
    # 调用写入文件函数，将数据写入到TODO.md文件中
    write_file("TODO.md", list(languages.values()), strings_done)
```