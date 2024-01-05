# `d:/src/tocomm/Bert-VITS2\update_status.py`

```
import os  # 导入os模块，用于操作文件和目录
import gradio as gr  # 导入gradio模块，用于构建交互式界面

lang_dict = {"EN(英文)": "_en", "ZH(中文)": "_zh", "JP(日语)": "_jp"}  # 定义一个字典，用于存储语言对应的后缀


def raw_dir_convert_to_path(target_dir: str, lang):
    # 将目标目录的末尾的斜杠去除，并赋值给res变量
    res = target_dir.rstrip("/").rstrip("\\")
    # 如果目标目录不以"raw"或"./raw"开头，则在目标目录前加上"./raw"，并赋值给res变量
    if (not target_dir.startswith("raw")) and (not target_dir.startswith("./raw")):
        res = os.path.join("./raw", res)
    # 如果res变量的末尾不是"_zh"、"_jp"、"_en"，则在末尾加上对应语言的后缀，并赋值给res变量
    if (
        (not res.endswith("_zh"))
        and (not res.endswith("_jp"))
        and (not res.endswith("_en"))
    ):
        res += lang_dict[lang]
    # 返回处理后的目录路径
    return res


def update_g_files():
    g_files = []
    cnt = 0
    # 遍历指定目录下的所有文件和文件夹
    for root, dirs, files in os.walk(os.path.abspath("./logs")):
        # 遍历当前目录下的所有文件
        for file in files:
            # 判断文件名是否以"G_"开头且以".pth"结尾
            if file.startswith("G_") and file.endswith(".pth"):
                # 将符合条件的文件路径添加到g_files列表中
                g_files.append(os.path.join(root, file))
                # 统计符合条件的文件个数
                cnt += 1
    # 打印g_files列表
    print(g_files)
    # 返回更新模型列表完成的提示信息和模型文件路径列表
    return f"更新模型列表完成, 共找到{cnt}个模型", gr.Dropdown.update(choices=g_files)


def update_c_files():
    c_files = []
    cnt = 0
    # 遍历指定目录下的所有文件和文件夹
    for root, dirs, files in os.walk(os.path.abspath("./logs")):
        # 遍历当前目录下的所有文件
        for file in files:
            # 判断文件名是否为"config.json"
            if file.startswith("config.json"):
                # 将符合条件的文件路径添加到c_files列表中
                c_files.append(os.path.join(root, file))
                # 统计符合条件的文件个数
                cnt += 1
    # 打印c_files列表
    print(c_files)
    return f"更新模型列表完成, 共找到{cnt}个配置文件", gr.Dropdown.update(choices=c_files)
```
这段代码是一个函数的返回语句。它返回一个字符串，其中包含找到的配置文件的数量，并使用这个数量来更新一个下拉菜单的选项。

```
def update_model_folders():
    subdirs = []
    cnt = 0
    for root, dirs, files in os.walk(os.path.abspath("./logs")):
        for dir_name in dirs:
            if os.path.basename(dir_name) != "eval":
                subdirs.append(os.path.join(root, dir_name))
                cnt += 1
    print(subdirs)
    return f"更新模型文件夹列表完成, 共找到{cnt}个文件夹", gr.Dropdown.update(choices=subdirs)
```
这段代码定义了一个名为`update_model_folders`的函数。它遍历指定目录下的子目录，并将不是以"eval"命名的子目录的路径添加到一个列表中。同时，它还计算找到的文件夹的数量。最后，它打印出子目录列表，并返回一个包含找到的文件夹数量的字符串，并使用这些文件夹路径来更新一个下拉菜单的选项。

```
def update_wav_lab_pairs():
    wav_count = tot_count = 0
    for root, _, files in os.walk("./raw"):
        for file in files:
            # print(file)
```
这段代码定义了一个名为`update_wav_lab_pairs`的函数。它遍历指定目录下的文件，并对每个文件执行一些操作。目前，这段代码被注释掉了，所以它不会执行任何操作。
            file_path = os.path.join(root, file)  # 获取文件的完整路径
            if file.lower().endswith(".wav"):  # 判断文件是否以.wav结尾
                lab_file = os.path.splitext(file_path)[0] + ".lab"  # 将.wav文件的路径转换为.lab文件的路径
                if os.path.exists(lab_file):  # 判断.lab文件是否存在
                    wav_count += 1  # 如果.lab文件存在，则.wav文件计数加1
                tot_count += 1  # 总文件计数加1
    return f"{wav_count} / {tot_count}"  # 返回.wav文件计数和总文件计数的字符串表示


def update_raw_folders():
    subdirs = []  # 存储子目录的列表
    cnt = 0  # 计数器
    script_path = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的绝对路径
    raw_path = os.path.join(script_path, "raw")  # 将"raw"目录添加到当前脚本路径中
    print(raw_path)  # 打印"raw"目录的路径
    os.makedirs(raw_path, exist_ok=True)  # 创建"raw"目录，如果目录已存在则忽略
    for root, dirs, files in os.walk(raw_path):  # 遍历"raw"目录下的所有文件和子目录
        for dir_name in dirs:  # 遍历子目录
            relative_path = os.path.relpath(
                os.path.join(root, dir_name), script_path  # 获取子目录相对于当前脚本路径的相对路径
    )  # 获取相对路径
```
这行代码是一个函数调用的结束括号，用于获取相对路径。

```
    subdirs.append(relative_path)
```
将相对路径添加到subdirs列表中。

```
    cnt += 1
```
计数器cnt加1，用于记录找到的文件夹数量。

```
print(subdirs)
```
打印subdirs列表，输出找到的文件夹列表。

```
return (
    f"更新raw音频文件夹列表完成, 共找到{cnt}个文件夹",
    gr.Dropdown.update(choices=subdirs),
    gr.Textbox.update(value=update_wav_lab_pairs()),
)
```
返回一个元组，包含三个元素。第一个元素是一个字符串，表示更新文件夹列表完成，并显示找到的文件夹数量。第二个元素是一个函数调用，用于更新下拉菜单的选项。第三个元素是一个函数调用，用于更新文本框的值。
```