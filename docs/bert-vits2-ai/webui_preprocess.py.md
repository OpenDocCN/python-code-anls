# `Bert-VITS2\webui_preprocess.py`

```

# 导入必要的库
import gradio as gr
import webbrowser
import os
import json
import subprocess
import shutil

# 获取数据集路径
def get_path(data_dir):
    start_path = os.path.join("./data", data_dir)
    lbl_path = os.path.join(start_path, "esd.list")
    train_path = os.path.join(start_path, "train.list")
    val_path = os.path.join(start_path, "val.list")
    config_path = os.path.join(start_path, "configs", "config.json")
    return start_path, lbl_path, train_path, val_path, config_path

# 生成配置文件
def generate_config(data_dir, batch_size):
    assert data_dir != "", "数据集名称不能为空"
    start_path, _, train_path, val_path, config_path = get_path(data_dir)
    if os.path.isfile(config_path):
        config = json.load(open(config_path, "r", encoding="utf-8"))
    else:
        config = json.load(open("configs/config.json", "r", encoding="utf-8"))
    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path
    config["train"]["batch_size"] = batch_size
    out_path = os.path.join(start_path, "configs")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    model_path = os.path.join(start_path, "models")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    if not os.path.exists("config.yml"):
        shutil.copy(src="default_config.yml", dst="config.yml")
    return "配置文件生成完成"

# 音频文件预处理
def resample(data_dir):
    assert data_dir != "", "数据集名称不能为空"
    start_path, _, _, _, config_path = get_path(data_dir)
    in_dir = os.path.join(start_path, "raw")
    out_dir = os.path.join(start_path, "wavs")
    subprocess.run(
        f"python resample_legacy.py "
        f"--sr 44100 "
        f"--in_dir {in_dir} "
        f"--out_dir {out_dir} ",
        shell=True,
    )
    return "音频文件预处理完成"

# 预处理标签文件
def preprocess_text(data_dir):
    assert data_dir != "", "数据集名称不能为空"
    start_path, lbl_path, train_path, val_path, config_path = get_path(data_dir)
    lines = open(lbl_path, "r", encoding="utf-8").readlines()
    with open(lbl_path, "w", encoding="utf-8") as f:
        for line in lines:
            path, spk, language, text = line.strip().split("|")
            path = os.path.join(start_path, "wavs", os.path.basename(path)).replace(
                "\\", "/"
            )
            f.writelines(f"{path}|{spk}|{language}|{text}\n")
    subprocess.run(
        f"python preprocess_text.py "
        f"--transcription-path {lbl_path} "
        f"--train-path {train_path} "
        f"--val-path {val_path} "
        f"--config-path {config_path}",
        shell=True,
    )
    return "标签文件预处理完成"

# 生成BERT特征文件
def bert_gen(data_dir):
    assert data_dir != "", "数据集名称不能为空"
    _, _, _, _, config_path = get_path(data_dir)
    subprocess.run(
        f"python bert_gen.py " f"--config {config_path}",
        shell=True,
    )
    return "BERT 特征文件生成完成"

# 主程序入口
if __name__ == "__main__":
    # 创建 Gradio 应用
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                # 显示说明文档
                _ = gr.Markdown(
                    value="# Bert-VITS2 数据预处理\n"
                    "## 预先准备：\n"
                    # 省略部分内容
                )
                # 输入数据集名称
                data_dir = gr.Textbox(
                    label="数据集名称",
                    placeholder="你放置在 data 文件夹下的数据集所在文件夹的名称，如 data/genshin 则填 genshin",
                )
                # 显示状态信息
                info = gr.Textbox(label="状态信息")
                # 生成配置文件部分
                _ = gr.Markdown(value="## 第一步：生成配置文件")
                with gr.Row():
                    batch_size = gr.Slider(
                        label="批大小（Batch size）：24 GB 显存可用 12",
                        value=8,
                        minimum=1,
                        maximum=64,
                        step=1,
                    )
                    generate_config_btn = gr.Button(value="执行", variant="primary")
                # 预处理音频文件部分
                _ = gr.Markdown(value="## 第二步：预处理音频文件")
                resample_btn = gr.Button(value="执行", variant="primary")
                # 预处理标签文件部分
                _ = gr.Markdown(value="## 第三步：预处理标签文件")
                preprocess_text_btn = gr.Button(value="执行", variant="primary")
                # 生成BERT特征文件部分
                _ = gr.Markdown(value="## 第四步：生成 BERT 特征文件")
                bert_gen_btn = gr.Button(value="执行", variant="primary")
                # 训练模型及部署部分
                _ = gr.Markdown(
                    value="## 训练模型及部署：\n"
                    # 省略部分内容
                )

        # 绑定按钮点击事件
        generate_config_btn.click(
            generate_config, inputs=[data_dir, batch_size], outputs=[info]
        )
        resample_btn.click(resample, inputs=[data_dir], outputs=[info])
        preprocess_text_btn.click(preprocess_text, inputs=[data_dir], outputs=[info])
        bert_gen_btn.click(bert_gen, inputs=[data_dir], outputs=[info])

    # 打开浏览器并启动应用
    webbrowser.open("http://127.0.0.1:7860")
    app.launch(share=False, server_port=7860)

```