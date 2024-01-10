# `Bert-VITS2\webui_preprocess.py`

```
# 导入 gradio 库
import gradio as gr
# 导入 webbrowser、os、json、subprocess、shutil 库
import webbrowser
import os
import json
import subprocess
import shutil

# 获取数据目录下的各个文件路径
def get_path(data_dir):
    start_path = os.path.join("./data", data_dir)
    lbl_path = os.path.join(start_path, "esd.list")
    train_path = os.path.join(start_path, "train.list")
    val_path = os.path.join(start_path, "val.list")
    config_path = os.path.join(start_path, "configs", "config.json")
    return start_path, lbl_path, train_path, val_path, config_path

# 生成配置文件
def generate_config(data_dir, batch_size):
    # 断言数据集名称不为空
    assert data_dir != "", "数据集名称不能为空"
    # 获取数据目录下的各个文件路径
    start_path, _, train_path, val_path, config_path = get_path(data_dir)
    # 如果配置文件存在，则加载配置文件
    if os.path.isfile(config_path):
        config = json.load(open(config_path, "r", encoding="utf-8"))
    else:
        config = json.load(open("configs/config.json", "r", encoding="utf-8"))
    # 设置训练和验证文件路径以及批处理大小
    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path
    config["train"]["batch_size"] = batch_size
    out_path = os.path.join(start_path, "configs")
    # 如果输出路径不存在，则创建输出路径
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    model_path = os.path.join(start_path, "models")
    # 如果模型路径不存在，则创建模型路径
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    # 如果 config.yml 文件不存在，则复制 default_config.yml 为 config.yml
    if not os.path.exists("config.yml"):
        shutil.copy(src="default_config.yml", dst="config.yml")
    return "配置文件生成完成"

# 重采样音频文件
def resample(data_dir):
    # 断言数据集名称不为空
    assert data_dir != "", "数据集名称不能为空"
    # 获取数据目录下的各个文件路径
    start_path, _, _, _, config_path = get_path(data_dir)
    in_dir = os.path.join(start_path, "raw")
    out_dir = os.path.join(start_path, "wavs")
    # 运行 resample_legacy.py 脚本进行音频文件重采样
    subprocess.run(
        f"python resample_legacy.py "
        f"--sr 44100 "
        f"--in_dir {in_dir} "
        f"--out_dir {out_dir} ",
        shell=True,
    )
    return "音频文件预处理完成"

# 预处理文本数据
def preprocess_text(data_dir):
    # 断言数据集名称不为空
    assert data_dir != "", "数据集名称不能为空"
    # 获取数据目录下的各个文件路径
    start_path, lbl_path, train_path, val_path, config_path = get_path(data_dir)
    # 读取标签文件的所有行
    lines = open(lbl_path, "r", encoding="utf-8").readlines()
    # 以写入模式打开标签文件
    with open(lbl_path, "w", encoding="utf-8") as f:
        # 遍历每一行
        for line in lines:
            # 从每一行中提取路径、说话者、语言和文本信息
            path, spk, language, text = line.strip().split("|")
            # 修改路径格式，并将路径指向指定目录下的文件
            path = os.path.join(start_path, "wavs", os.path.basename(path)).replace(
                "\\", "/"
            )
            # 将处理后的行写入标签文件
            f.writelines(f"{path}|{spk}|{language}|{text}\n")
    # 运行预处理文本的 Python 脚本，传入参数并以 shell 模式运行
    subprocess.run(
        f"python preprocess_text.py "
        f"--transcription-path {lbl_path} "
        f"--train-path {train_path} "
        f"--val-path {val_path} "
        f"--config-path {config_path}",
        shell=True,
    )
    # 返回预处理完成的消息
    return "标签文件预处理完成"
# 生成 BERT 特征文件
def bert_gen(data_dir):
    # 断言数据集名称不为空
    assert data_dir != "", "数据集名称不能为空"
    # 获取路径信息
    _, _, _, _, config_path = get_path(data_dir)
    # 运行 bert_gen.py 脚本，传入配置文件路径
    subprocess.run(
        f"python bert_gen.py " f"--config {config_path}",
        shell=True,
    )
    # 返回特征文件生成完成的消息
    return "BERT 特征文件生成完成"


# 如果作为主程序运行，则打开浏览器访问指定地址，并启动应用
if __name__ == "__main__":
    # 打开浏览器访问指定地址
    webbrowser.open("http://127.0.0.1:7860")
    # 启动应用，不共享，指定服务器端口为 7860
    app.launch(share=False, server_port=7860)
```