# `d:/src/tocomm/Bert-VITS2\webui_preprocess.py`

```
import gradio as gr  # 导入 gradio 库，用于构建交互式界面
import webbrowser  # 导入 webbrowser 库，用于打开网页
import os  # 导入 os 库，用于操作文件和目录
import json  # 导入 json 库，用于处理 JSON 数据
import subprocess  # 导入 subprocess 库，用于执行外部命令
import shutil  # 导入 shutil 库，用于文件和目录的高级操作


def get_path(data_dir):
    # 构建文件路径
    start_path = os.path.join("./data", data_dir)  # 拼接目录路径
    lbl_path = os.path.join(start_path, "esd.list")  # 拼接文件路径
    train_path = os.path.join(start_path, "train.list")  # 拼接文件路径
    val_path = os.path.join(start_path, "val.list")  # 拼接文件路径
    config_path = os.path.join(start_path, "configs", "config.json")  # 拼接文件路径
    return start_path, lbl_path, train_path, val_path, config_path  # 返回构建的文件路径


def generate_config(data_dir, batch_size):
    assert data_dir != "", "数据集名称不能为空"  # 断言，确保数据集名称不为空
    start_path, _, train_path, val_path, config_path = get_path(data_dir)  # 获取文件路径的返回值
    if os.path.isfile(config_path):
        # 如果配置文件存在，则从配置文件中加载配置信息
        config = json.load(open(config_path, "r", encoding="utf-8"))
    else:
        # 如果配置文件不存在，则从默认配置文件中加载配置信息
        config = json.load(open("configs/config.json", "r", encoding="utf-8"))
    # 设置训练文件路径
    config["data"]["training_files"] = train_path
    # 设置验证文件路径
    config["data"]["validation_files"] = val_path
    # 设置批量大小
    config["train"]["batch_size"] = batch_size
    # 设置输出路径
    out_path = os.path.join(start_path, "configs")
    # 如果输出路径不存在，则创建输出路径
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    # 设置模型路径
    model_path = os.path.join(start_path, "models")
    # 如果模型路径不存在，则创建模型路径
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    # 将配置信息写入配置文件
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    # 如果 "config.yml" 文件不存在，则复制 "default_config.yml" 文件为 "config.yml"
    if not os.path.exists("config.yml"):
        shutil.copy(src="default_config.yml", dst="config.yml")
    # 返回配置文件生成完成的提示信息
    return "配置文件生成完成"
def resample(data_dir):
    # 断言数据集名称不为空
    assert data_dir != "", "数据集名称不能为空"
    # 获取路径信息
    start_path, _, _, _, config_path = get_path(data_dir)
    # 输入目录
    in_dir = os.path.join(start_path, "raw")
    # 输出目录
    out_dir = os.path.join(start_path, "wavs")
    # 运行命令行命令，对音频文件进行重采样
    subprocess.run(
        f"python resample_legacy.py "
        f"--sr 44100 "
        f"--in_dir {in_dir} "
        f"--out_dir {out_dir} ",
        shell=True,
    )
    # 返回预处理完成的消息
    return "音频文件预处理完成"


def preprocess_text(data_dir):
    # 断言数据集名称不为空
    assert data_dir != "", "数据集名称不能为空"
    # 获取路径信息
    start_path, lbl_path, train_path, val_path, config_path = get_path(data_dir)
    # 读取标签文件的内容
    lines = open(lbl_path, "r", encoding="utf-8").readlines()
    # 以写入模式打开标签文件
    with open(lbl_path, "w", encoding="utf-8") as f:
        # ... (此处省略了后续的代码)
```

这段代码主要是对音频文件和文本文件进行预处理的函数。其中，`resample`函数用于对音频文件进行重采样，`preprocess_text`函数用于对文本文件进行预处理。

在`resample`函数中，首先通过断言确保数据集名称不为空。然后，通过调用`get_path`函数获取路径信息，包括起始路径、标签路径、训练路径、验证路径和配置路径。接下来，根据起始路径拼接输入目录和输出目录的路径。然后，使用`subprocess.run`函数运行命令行命令，对音频文件进行重采样。最后，返回预处理完成的消息。

在`preprocess_text`函数中，同样通过断言确保数据集名称不为空。然后，通过调用`get_path`函数获取路径信息。接下来，使用`open`函数以只读模式打开标签文件，并读取其内容到`lines`变量中。然后，以写入模式打开标签文件，并使用`with`语句来确保文件在使用完后被正确关闭。在`with`语句块中，可以继续对标签文件进行后续的处理操作。
        for line in lines:
            path, spk, language, text = line.strip().split("|")
            path = os.path.join(start_path, "wavs", os.path.basename(path)).replace(
                "\\", "/"
            )
            f.writelines(f"{path}|{spk}|{language}|{text}\n")
```
这段代码是一个for循环，用于遍历`lines`列表中的每一行。每一行都被拆分成`path`、`spk`、`language`和`text`四个变量。`path`变量通过`os.path.join()`函数将`start_path`、"wavs"和`os.path.basename(path)`拼接起来，并将路径中的反斜杠替换为斜杠。然后，使用`f.writelines()`函数将拼接后的字符串写入文件。

```
    subprocess.run(
        f"python preprocess_text.py "
        f"--transcription-path {lbl_path} "
        f"--train-path {train_path} "
        f"--val-path {val_path} "
        f"--config-path {config_path}",
        shell=True,
    )
```
这段代码使用`subprocess.run()`函数来运行一个子进程。子进程的命令是一个字符串，其中包含了要执行的Python脚本和一些命令行参数。这个子进程的作用是调用`preprocess_text.py`脚本，并传递一些路径参数给它。`shell=True`表示在shell中执行命令。

```
    return "标签文件预处理完成"
```
这行代码返回一个字符串，表示标签文件的预处理已经完成。

```
def bert_gen(data_dir):
    assert data_dir != "", "数据集名称不能为空"
    _, _, _, _, config_path = get_path(data_dir)
```
这是一个函数定义，函数名为`bert_gen`，接受一个参数`data_dir`。在函数内部，使用`assert`语句来确保`data_dir`不为空字符串，如果为空字符串，则抛出一个断言错误。然后，调用`get_path()`函数获取一些路径，并将其中的第五个路径赋值给`config_path`变量。
subprocess.run(
    f"python bert_gen.py " f"--config {config_path}",
    shell=True,
)
```
这段代码使用`subprocess.run()`函数来运行一个命令行命令。命令是通过字符串插值的方式构建的，其中`config_path`是一个变量。`shell=True`表示在shell中执行命令。

```
return "BERT 特征文件生成完成"
```
这行代码返回一个字符串，表示BERT特征文件生成完成。

```
if __name__ == "__main__":
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                _ = gr.Markdown(
                    value="# Bert-VITS2 数据预处理\n"
                    "## 预先准备：\n"
                    "下载 BERT 和 WavLM 模型：\n"
                    "- [中文 RoBERTa](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)\n"
                    "- [日文 DeBERTa](https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm)\n"
                    "- [英文 DeBERTa](https://huggingface.co/microsoft/deberta-v3-large)\n"
                    "- [WavLM](https://huggingface.co/microsoft/wavlm-base-plus)\n"
                    "\n"
                )
```
这段代码定义了一个主程序的入口点。它创建了一个`gr.Blocks()`对象，并使用`with`语句来确保在程序结束时正确关闭该对象。然后，它创建了一个`gr.Row()`对象和一个`gr.Column()`对象，用于布局。在`gr.Column()`对象中，使用`gr.Markdown()`函数创建了一个Markdown文本对象，用于显示一些预处理的说明文本。

注：以上代码片段中的`gr`是一个未定义的变量，可能是一个自定义的模块或库。在注释中无法确定其具体作用。
这段代码是一段注释，用于说明如何准备数据和放置模型文件的目录结构。具体内容如下：

- 将 BERT 模型放置到 `bert` 文件夹下，WavLM 模型放置到 `slm` 文件夹下，覆盖同名文件夹。
- 数据准备：将数据放置在 data 文件夹下，按照如下结构组织：
  ```
  ├── data
  │   ├── {你的数据集名称}
  │   │   ├── esd.list
  │   │   ├── raw
  │   │   │   ├── ****.wav
  │   │   │   ├── ****.wav
  │   │   │   ├── ...
  ```
  其中，`raw` 文件夹下保存所有的音频文件，`esd.list` 文件为标签文本，格式为
  ```
  ****.wav|{说话人名}|{语言 ID}|{标签文本}
  ```
  其中，`****.wav` 是音频文件名，`{说话人名}` 是说话人的名称，`{语言 ID}` 是语言的标识，`{标签文本}` 是对应音频的标签文本。
                    "\n"
                    "例如：\n"
                    "```\n"
                    "vo_ABDLQ001_1_paimon_02.wav|派蒙|ZH|没什么没什么，只是平时他总是站在这里，有点奇怪而已。\n"
                    "noa_501_0001.wav|NOA|JP|そうだね、油断しないのはとても大事なことだと思う\n"
                    "Albedo_vo_ABDLQ002_4_albedo_01.wav|Albedo|EN|Who are you? Why did you alarm them?\n"
                    "...\n"
                    "```\n"
                )
```
这段代码是一个多行字符串，用于提供示例数据的格式和内容。

```
                data_dir = gr.Textbox(
                    label="数据集名称",
                    placeholder="你放置在 data 文件夹下的数据集所在文件夹的名称，如 data/genshin 则填 genshin",
                )
```
这段代码定义了一个文本框，用于输入数据集的名称。它有一个标签用于显示在文本框上方，还有一个占位符用于在文本框中显示示例文本。

```
                info = gr.Textbox(label="状态信息")
```
这段代码定义了一个文本框，用于输入状态信息。它有一个标签用于显示在文本框上方。

```
                _ = gr.Markdown(value="## 第一步：生成配置文件")
```
这段代码创建了一个 Markdown 组件，用于显示标题为"第一步：生成配置文件"的文本。

```
                with gr.Row():
                    batch_size = gr.Slider(
                        label="批大小（Batch size）：24 GB 显存可用 12",
                        value=8,
                        minimum=1,
```
这段代码创建了一个滑块组件，用于选择批大小。它有一个标签用于显示在滑块上方，还有一个默认值和最小值。
                        maximum=64,
                        step=1,
                    )
                    generate_config_btn = gr.Button(value="执行", variant="primary")
                _ = gr.Markdown(value="## 第二步：预处理音频文件")
                resample_btn = gr.Button(value="执行", variant="primary")
                _ = gr.Markdown(value="## 第三步：预处理标签文件")
                preprocess_text_btn = gr.Button(value="执行", variant="primary")
                _ = gr.Markdown(value="## 第四步：生成 BERT 特征文件")
                bert_gen_btn = gr.Button(value="执行", variant="primary")
                _ = gr.Markdown(
                    value="## 训练模型及部署：\n"
                    "修改根目录下的 `config.yml` 中 `dataset_path` 一项为 `data/{你的数据集名称}`\n"
                    "- 部署：修改根目录下的 `config.yml` 中 `webui` 下 `model` 一项为 `models/{权重文件名}.pth` （如 G_10000.pth），然后执行 `python webui.py`"
                )

        generate_config_btn.click(
            generate_config, inputs=[data_dir, batch_size], outputs=[info]
        )
        resample_btn.click(resample, inputs=[data_dir], outputs=[info])
```

这段代码是一个界面的布局代码，用于创建按钮和文本框等界面元素，并设置它们的属性和事件。

- `maximum=64`：设置一个按钮的最大值为64。
- `step=1`：设置一个按钮的步长为1。
- `generate_config_btn = gr.Button(value="执行", variant="primary")`：创建一个名为`generate_config_btn`的按钮，显示文本为"执行"，样式为"primary"。
- `_ = gr.Markdown(value="## 第二步：预处理音频文件")`：创建一个名为`_`的Markdown文本框，显示文本为"## 第二步：预处理音频文件"。
- `resample_btn = gr.Button(value="执行", variant="primary")`：创建一个名为`resample_btn`的按钮，显示文本为"执行"，样式为"primary"。
- `_ = gr.Markdown(value="## 第三步：预处理标签文件")`：创建一个名为`_`的Markdown文本框，显示文本为"## 第三步：预处理标签文件"。
- `preprocess_text_btn = gr.Button(value="执行", variant="primary")`：创建一个名为`preprocess_text_btn`的按钮，显示文本为"执行"，样式为"primary"。
- `_ = gr.Markdown(value="## 第四步：生成 BERT 特征文件")`：创建一个名为`_`的Markdown文本框，显示文本为"## 第四步：生成 BERT 特征文件"。
- `bert_gen_btn = gr.Button(value="执行", variant="primary")`：创建一个名为`bert_gen_btn`的按钮，显示文本为"执行"，样式为"primary"。
- `_ = gr.Markdown(value="## 训练模型及部署：\n"...)`：创建一个名为`_`的Markdown文本框，显示一段关于训练模型及部署的说明文本。
- `generate_config_btn.click(...)`：为`generate_config_btn`按钮添加点击事件，当点击按钮时，执行`generate_config`函数，传入`data_dir`和`batch_size`作为输入，将结果输出到`info`。
- `resample_btn.click(...)`：为`resample_btn`按钮添加点击事件，当点击按钮时，执行`resample`函数，传入`data_dir`作为输入，将结果输出到`info`。
# 点击 preprocess_text_btn 按钮，执行 preprocess_text 函数，输入参数为 data_dir，输出结果为 info
preprocess_text_btn.click(preprocess_text, inputs=[data_dir], outputs=[info])
# 点击 bert_gen_btn 按钮，执行 bert_gen 函数，输入参数为 data_dir，输出结果为 info
bert_gen_btn.click(bert_gen, inputs=[data_dir], outputs=[info])

# 在浏览器中打开指定的网址
webbrowser.open("http://127.0.0.1:7860")
# 启动应用程序，不共享资源，使用端口号 7860
app.launch(share=False, server_port=7860)
```