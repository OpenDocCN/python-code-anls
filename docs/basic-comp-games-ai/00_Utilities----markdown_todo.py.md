# `basic-computer-games\00_Utilities\markdown_todo.py`

```

# 导入 os 模块
import os
# 导入类型提示模块 Dict 和 List
from typing import Dict, List

# 判断指定语言是否有实现文件
def has_implementation(lang: str, file_list: List[str], subdir_list: List[str]) -> bool:
    # 如果语言是 C#，则判断文件列表中是否有以 .cs 结尾的文件
    if lang == "csharp":
        return any(file.endswith(".cs") for file in file_list)
    # 如果语言是 VB.NET，则判断文件列表中是否有以 .vb 结尾的文件
    elif lang == "vbnet":
        return any(file.endswith(".vb") for file in file_list)
    # 其他语言，判断文件列表长度是否大于1或者子目录列表长度是否大于0
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
    # 创建语言位置字典
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

    prev_game = ""

    empty_boxes = ["⬜️" for _ in checklist_orig]
    checklist = empty_boxes[:]

    # 遍历根目录下的文件和文件夹
    for dir_name, subdir_list, file_list in sorted(os.walk(root_dir)):
        # 分割目录名
        split_dir = dir_name.split(os.path.sep)

        # 如果目录名长度为2且不在忽略列表中
        if len(split_dir) == 2 and split_dir[1] not in ignore_folders:
            # 如果前一个游戏为空，则设置为当前游戏
            if prev_game == "":
                prev_game = split_dir[1]
                checklist[0] = f"{split_dir[1]:<30}"

            # 如果前一个游戏不等于当前游戏
            if prev_game != split_dir[1]:
                # 添加到已完成的字符串列表中
                strings_done.append(checklist)
                # 重置检查列表
                checklist = [
                    f"{f'[{split_dir[1]}](../{split_dir[1]})':<30}",
                ] + empty_boxes[1:]
                prev_game = split_dir[1]
        # 如果目录名长度为3且不在忽略列表中且语言在语言位置字典中
        elif (
            len(split_dir) == 3 and split_dir[1] != ".git" and split_dir[2] in lang_pos
        ):
            # 判断是否有实现，有则标记为✅，否则标记为⬜️
            out = (
                "✅"
                if has_implementation(split_dir[2], file_list, subdir_list)
                else "⬜️"
            )
            # 如果语言不在语言位置字典中或者位置大于检查列表长度
            if split_dir[2] not in lang_pos or lang_pos[split_dir[2]] >= len(checklist):
                print(f"Could not find {split_dir[2]}: {dir_name}")
                checklist[lang_pos[split_dir[2]]] = "⬜️"
                continue
            checklist[lang_pos[split_dir[2]]] = out
    return strings_done

# 写入文件
def write_file(path: str, languages: List[str], strings_done: List[List[str]]) -> None:
    # 创建破折号数组
    dashes_arr = ["---"] * (len(languages) + 1)
    dashes_arr[0] = "-" * 30
    dashes = " | ".join(dashes_arr)
    write_string = f"# TODO list\n {'game':<30}| {' | '.join(languages)}\n{dashes}\n"
    # 对已完成的字符串列表进行排序
    sorted_strings = list(
        map(lambda l: " | ".join(l) + "\n", sorted(strings_done, key=lambda x: x[0]))
    )
    write_string += "".join(sorted_strings)
    write_string += f"{dashes}\n"
    language_indices = range(1, len(languages) + 1)
    nb_games = len(strings_done)
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

    with open(path, "w", encoding="utf-8") as f:
        f.write(write_string)

# 主函数
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
    # 获取数据
    strings_done = get_data(["game"] + list(languages.keys()))
    # 写入文件
    write_file("TODO.md", list(languages.values()), strings_done)

```