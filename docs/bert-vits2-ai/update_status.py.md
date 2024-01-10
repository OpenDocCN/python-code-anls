# `Bert-VITS2\update_status.py`

```
# 导入 os 模块和 gradio 模块
import os
import gradio as gr

# 定义语言到后缀的映射字典
lang_dict = {"EN(英文)": "_en", "ZH(中文)": "_zh", "JP(日语)": "_jp"}

# 将原始目录转换为路径
def raw_dir_convert_to_path(target_dir: str, lang):
    # 去除目标目录末尾的斜杠
    res = target_dir.rstrip("/").rstrip("\\")
    # 如果目标目录不以 "raw" 或 "./raw" 开头，则添加前缀
    if (not target_dir.startswith("raw")) and (not target_dir.startswith("./raw")):
        res = os.path.join("./raw", res)
    # 如果目标目录不以 "_zh"、"_jp"、"_en" 结尾，则根据语言添加后缀
    if (
        (not res.endswith("_zh"))
        and (not res.endswith("_jp"))
        and (not res.endswith("_en"))
    ):
        res += lang_dict[lang]
    return res

# 更新 G 文件列表
def update_g_files():
    g_files = []
    cnt = 0
    # 遍历 "./logs" 目录下的文件和子目录
    for root, dirs, files in os.walk(os.path.abspath("./logs")):
        for file in files:
            # 如果文件以 "G_" 开头且以 ".pth" 结尾，则添加到列表中
            if file.startswith("G_") and file.endswith(".pth"):
                g_files.append(os.path.join(root, file))
                cnt += 1
    # 打印找到的文件列表
    print(g_files)
    # 返回更新模型列表的消息和模型文件列表
    return f"更新模型列表完成, 共找到{cnt}个模型", gr.Dropdown.update(choices=g_files)

# 更新 C 文件列表
def update_c_files():
    c_files = []
    cnt = 0
    # 遍历 "./logs" 目录下的文件和子目录
    for root, dirs, files in os.walk(os.path.abspath("./logs")):
        for file in files:
            # 如果文件名以 "config.json" 开头，则添加到列表中
            if file.startswith("config.json"):
                c_files.append(os.path.join(root, file))
                cnt += 1
    # 打印找到的文件列表
    print(c_files)
    # 返回更新模型列表的消息和配置文件列表
    return f"更新模型列表完成, 共找到{cnt}个配置文件", gr.Dropdown.update(choices=c_files)

# 更新模型文件夹列表
def update_model_folders():
    subdirs = []
    cnt = 0
    # 遍历 "./logs" 目录下的文件和子目录
    for root, dirs, files in os.walk(os.path.abspath("./logs")):
        for dir_name in dirs:
            # 如果子目录名不是 "eval"，则添加到列表中
            if os.path.basename(dir_name) != "eval":
                subdirs.append(os.path.join(root, dir_name))
                cnt += 1
    # 打印找到的文件夹列表
    print(subdirs)
    # 返回更新模型文件夹列表的消息和文件夹列表
    return f"更新模型文件夹列表完成, 共找到{cnt}个文件夹", gr.Dropdown.update(choices=subdirs)

# 更新 WAV 和标签对列表
def update_wav_lab_pairs():
    wav_count = tot_count = 0
    # 遍历指定目录下的所有文件和子目录
    for root, _, files in os.walk("./raw"):
        # 遍历当前目录下的所有文件
        for file in files:
            # 拼接文件的完整路径
            file_path = os.path.join(root, file)
            # 判断文件是否以.wav结尾
            if file.lower().endswith(".wav"):
                # 构建对应的.lab文件路径
                lab_file = os.path.splitext(file_path)[0] + ".lab"
                # 如果.lab文件存在，则wav_count加1
                if os.path.exists(lab_file):
                    wav_count += 1
                # 总文件数加1
                tot_count += 1
    # 返回格式化的字符串，表示.wav文件数量和总文件数量
    return f"{wav_count} / {tot_count}"
# 更新 raw 文件夹列表
def update_raw_folders():
    # 初始化子文件夹列表
    subdirs = []
    # 初始化计数器
    cnt = 0
    # 获取当前脚本的绝对路径
    script_path = os.path.dirname(os.path.abspath(__file__))
    # 拼接得到 raw 文件夹的路径
    raw_path = os.path.join(script_path, "raw")
    # 打印 raw 文件夹的路径
    print(raw_path)
    # 如果 raw 文件夹不存在，则创建
    os.makedirs(raw_path, exist_ok=True)
    # 遍历 raw 文件夹及其子文件夹
    for root, dirs, files in os.walk(raw_path):
        # 遍历每个子文件夹
        for dir_name in dirs:
            # 获取子文件夹相对于脚本路径的相对路径
            relative_path = os.path.relpath(
                os.path.join(root, dir_name), script_path
            )
            # 将相对路径添加到子文件夹列表中
            subdirs.append(relative_path)
            # 计数器加一
            cnt += 1
    # 打印子文件夹列表
    print(subdirs)
    # 返回更新完成的提示信息、下拉框的更新结果和文本框的更新结果
    return (
        f"更新raw音频文件夹列表完成, 共找到{cnt}个文件夹",
        gr.Dropdown.update(choices=subdirs),
        gr.Textbox.update(value=update_wav_lab_pairs()),
    )
```