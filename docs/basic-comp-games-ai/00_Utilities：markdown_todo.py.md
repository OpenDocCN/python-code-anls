# `00_Utilities\markdown_todo.py`

```
    checklist = checklist_orig.copy()  # 复制输入的检查列表
    result = []  # 存储结果的列表

    for subdir, dirs, files in os.walk(root_dir):  # 遍历根目录及其子目录下的所有文件和文件夹
        if not checklist:  # 如果检查列表为空
            break  # 跳出循环

        if all(item in subdir for item in checklist):  # 如果检查列表中的所有项都在当前子目录中
            file_list = [f for f in files if not f.startswith('.')]  # 获取当前子目录下的文件列表
            subdir_list = [d for d in dirs if not d.startswith('.')]  # 获取当前子目录下的子目录列表

            if has_implementation("csharp", file_list, subdir_list):  # 如果当前子目录下有 C# 文件
                result.append([subdir, "csharp"])  # 将当前子目录和语言类型添加到结果列表中
                checklist.remove("csharp")  # 从检查列表中移除该语言类型

            if has_implementation("vbnet", file_list, subdir_list):  # 如果当前子目录下有 VB.NET 文件
                result.append([subdir, "vbnet"])  # 将当前子目录和语言类型添加到结果列表中
                checklist.remove("vbnet")  # 从检查列表中移除该语言类型

    return result  # 返回结果列表
    """
    lang_pos: Dict[str, int] = {
        lang: i for i, lang in enumerate(checklist_orig[1:], start=1)
    }
    # 创建一个字典，键为字符串类型，值为整数类型，用于存储语言和对应的位置信息

    strings_done: List[List[str]] = []
    # 创建一个列表，列表中的元素是字符串列表，用于存储已处理的字符串

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
    # 创建一个列表，用于存储需要忽略的文件夹名称

    prev_game = ""
    # 创建一个变量，用于存储前一个游戏的信息
    # 创建一个包含与checklist_orig相同数量的空白框的列表
    empty_boxes = ["⬜️" for _ in checklist_orig]
    # 复制empty_boxes列表，得到一个新的checklist列表
    checklist = empty_boxes[:]

    # 遍历root_dir目录及其子目录中的所有文件和文件夹
    for dir_name, subdir_list, file_list in sorted(os.walk(root_dir)):
        # 将目录名按照os.path.sep分割成列表
        split_dir = dir_name.split(os.path.sep)

        # 如果split_dir列表长度为2且split_dir[1]不在ignore_folders列表中
        if len(split_dir) == 2 and split_dir[1] not in ignore_folders:
            # 如果prev_game为空字符串
            if prev_game == "":
                # 将prev_game设置为split_dir[1]，并将checklist的第一个元素设置为split_dir[1]的左对齐字符串
                prev_game = split_dir[1]
                checklist[0] = f"{split_dir[1]:<30}"

            # 如果prev_game不等于split_dir[1]
            if prev_game != split_dir[1]:
                # 将当前checklist添加到strings_done列表中
                strings_done.append(checklist)
                # 重新设置checklist列表，包括split_dir[1]的链接和其他空白框
                checklist = [
                    f"{f'[{split_dir[1]}](../{split_dir[1]})':<30}",
                ] + empty_boxes[1:]
                # 将prev_game设置为split_dir[1]
                prev_game = split_dir[1]
        elif (
            len(split_dir) == 3 and split_dir[1] != ".git" and split_dir[2] in lang_pos
        ):
            # 检查条件：split_dir 列表长度为3，且split_dir的第二个元素不是".git"，并且split_dir的第三个元素在lang_pos中
            out = (
                "✅"
                if has_implementation(split_dir[2], file_list, subdir_list)
                else "⬜️"
            )
            # 如果has_implementation函数返回True，则out为"✅"，否则为"⬜️"
            if split_dir[2] not in lang_pos or lang_pos[split_dir[2]] >= len(checklist):
                # 如果split_dir的第三个元素不在lang_pos中，或者lang_pos中split_dir的第三个元素对应的值大于等于checklist的长度
                print(f"Could not find {split_dir[2]}: {dir_name}")
                # 打印错误信息
                checklist[lang_pos[split_dir[2]]] = "⬜️"
                # 将checklist中对应位置的值设为"⬜️"
                continue
            checklist[lang_pos[split_dir[2]]] = out
            # 将checklist中对应位置的值设为out
    return strings_done
    # 返回strings_done列表


def write_file(path: str, languages: List[str], strings_done: List[List[str]]) -> None:
    # 定义一个由"---"组成的列表，长度为languages的长度加1
    dashes_arr = ["---"] * (len(languages) + 1)
    # 将dashes_arr的第一个元素设为"-"重复30次
    dashes_arr[0] = "-" * 30
    # 将dashes_arr中的元素用" | "连接起来，赋值给dashes
    # 创建待写入文件的字符串，包括表头和已完成的任务列表
    write_string = f"# TODO list\n {'game':<30}| {' | '.join(languages)}\n{dashes}\n"
    # 对已完成的任务列表按照游戏名称排序，并转换成字符串列表
    sorted_strings = list(
        map(lambda l: " | ".join(l) + "\n", sorted(strings_done, key=lambda x: x[0]))
    )
    # 将排序后的任务列表添加到待写入文件的字符串中
    write_string += "".join(sorted_strings)
    # 添加分隔线
    write_string += f"{dashes}\n"
    # 计算每种语言已完成的任务数量，并添加到待写入文件的字符串中
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

    # 打开文件并写入待写入文件的字符串
    with open(path, "w", encoding="utf-8") as f:
if __name__ == "__main__":
    # 定义一个包含编程语言名称和对应缩写的字典
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
    # 调用 get_data 函数，传入包含"game"和编程语言名称的列表，获取数据
    strings_done = get_data(["game"] + list(languages.keys()))
    # 调用 write_file 函数，将获取的数据写入到"TODO.md"文件中
    write_file("TODO.md", list(languages.values()), strings_done)
```