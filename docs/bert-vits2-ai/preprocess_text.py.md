# `d:/src/tocomm/Bert-VITS2\preprocess_text.py`

```
import json  # 导入json模块，用于处理JSON数据
from collections import defaultdict  # 导入defaultdict类，用于创建默认值为列表的字典
from random import shuffle  # 导入shuffle函数，用于随机打乱列表的顺序
from typing import Optional  # 导入Optional类型，用于指定可选的函数参数类型
import os  # 导入os模块，用于与操作系统进行交互

from tqdm import tqdm  # 导入tqdm模块，用于显示进度条
import click  # 导入click模块，用于创建命令行接口
from text.cleaner import clean_text  # 从text.cleaner模块中导入clean_text函数，用于文本清洗
from config import config  # 从config模块中导入config对象，用于读取配置信息
from infer import latest_version  # 从infer模块中导入latest_version函数，用于获取最新版本号

preprocess_text_config = config.preprocess_text_config  # 从config对象中获取preprocess_text_config配置信息

# 创建命令行接口的装饰器，用于解析命令行参数
@click.command()
@click.option(
    "--transcription-path",
    default=preprocess_text_config.transcription_path,  # 设置默认值为preprocess_text_config配置中的transcription_path
    type=click.Path(exists=True, file_okay=True, dir_okay=False),  # 指定参数类型为路径，要求文件存在且不是目录
```

这段代码是一个Python脚本的开头部分，用于导入所需的模块和定义命令行接口的装饰器。其中，`import`语句用于导入需要使用的模块，`from ... import ...`语句用于从模块中导入特定的函数或对象。`preprocess_text_config`是一个变量，用于存储配置信息。`@click.command()`是一个装饰器，用于将下面的函数转换为命令行接口。`@click.option()`是另一个装饰器，用于定义命令行参数。在这段代码中，`--transcription-path`是一个可选的命令行参数，它的默认值为`preprocess_text_config`配置中的`transcription_path`。`type=click.Path(exists=True, file_okay=True, dir_okay=False)`指定了参数的类型为路径，要求文件存在且不是目录。
@click.option("--cleaned-path", default=preprocess_text_config.cleaned_path)
```
- `@click.option`：装饰器，用于定义命令行选项。
- `--cleaned-path`：命令行选项的名称。
- `default=preprocess_text_config.cleaned_path`：指定选项的默认值为`preprocess_text_config.cleaned_path`。

```
@click.option("--train-path", default=preprocess_text_config.train_path)
```
- `--train-path`：命令行选项的名称。
- `default=preprocess_text_config.train_path`：指定选项的默认值为`preprocess_text_config.train_path`。

```
@click.option("--val-path", default=preprocess_text_config.val_path)
```
- `--val-path`：命令行选项的名称。
- `default=preprocess_text_config.val_path`：指定选项的默认值为`preprocess_text_config.val_path`。

```
@click.option(
    "--config-path",
    default=preprocess_text_config.config_path,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
```
- `--config-path`：命令行选项的名称。
- `default=preprocess_text_config.config_path`：指定选项的默认值为`preprocess_text_config.config_path`。
- `type=click.Path(exists=True, file_okay=True, dir_okay=False)`：指定选项的类型为路径，并且要求路径存在且为文件。

```
@click.option("--val-per-lang", default=preprocess_text_config.val_per_lang)
```
- `--val-per-lang`：命令行选项的名称。
- `default=preprocess_text_config.val_per_lang`：指定选项的默认值为`preprocess_text_config.val_per_lang`。

```
@click.option("--max-val-total", default=preprocess_text_config.max_val_total)
```
- `--max-val-total`：命令行选项的名称。
- `default=preprocess_text_config.max_val_total`：指定选项的默认值为`preprocess_text_config.max_val_total`。

```
@click.option("--clean/--no-clean", default=preprocess_text_config.clean)
```
- `--clean/--no-clean`：命令行选项的名称，表示一个布尔值。
- `default=preprocess_text_config.clean`：指定选项的默认值为`preprocess_text_config.clean`。

```
@click.option("-y", "--yml_config")
```
- `-y`：命令行选项的短名称。
- `--yml_config`：命令行选项的长名称。

```
def preprocess(
    transcription_path: str,
    cleaned_path: Optional[str],
    train_path: str,
    val_path: str,
    config_path: str,
    val_per_lang: int,
```
- `def preprocess(...)`：定义了一个名为`preprocess`的函数。
- `transcription_path: str`：函数的第一个参数，类型为字符串。
- `cleaned_path: Optional[str]`：函数的第二个参数，类型为可选的字符串。
- `train_path: str`：函数的第三个参数，类型为字符串。
- `val_path: str`：函数的第四个参数，类型为字符串。
- `config_path: str`：函数的第五个参数，类型为字符串。
- `val_per_lang: int`：函数的第六个参数，类型为整数。
max_val_total: int,  # 定义一个整型变量 max_val_total
clean: bool,  # 定义一个布尔型变量 clean
yml_config: str,  # 定义一个字符串变量 yml_config，这个注释是提醒不要删除这个参数
):
if cleaned_path == "" or cleaned_path is None:  # 如果 cleaned_path 为空字符串或者为 None
    cleaned_path = transcription_path + ".cleaned"  # 将 transcription_path 加上后缀 ".cleaned" 赋值给 cleaned_path

if clean:  # 如果 clean 为真
    with open(cleaned_path, "w", encoding="utf-8") as out_file:  # 打开 cleaned_path 文件，以写入模式，编码为 utf-8，文件对象赋值给 out_file
        with open(transcription_path, "r", encoding="utf-8") as trans_file:  # 打开 transcription_path 文件，以读取模式，编码为 utf-8，文件对象赋值给 trans_file
            lines = trans_file.readlines()  # 读取 trans_file 中的所有行，返回一个包含所有行的列表，赋值给 lines
            # print(lines, ' ', len(lines))
            if len(lines) != 0:  # 如果 lines 列表的长度不为 0
                for line in tqdm(lines):  # 遍历 lines 列表中的每一行，使用 tqdm 进行进度条显示
                    try:
                        utt, spk, language, text = line.strip().split("|")  # 将 line 去除首尾空白字符后，按 "|" 分割成四个部分，分别赋值给 utt, spk, language, text
                        norm_text, phones, tones, word2ph = clean_text(  # 调用 clean_text 函数，传入 text 和 language 作为参数，返回的结果分别赋值给 norm_text, phones, tones, word2ph
                            text, language
                        )
                        out_file.write(  # 将下面的内容写入 out_file 文件
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
```

需要注释的代码：

```
# 以指定格式将变量的值转换为字符串，并将其添加到字符串中
"{}|{}|{}|{}|{}|{}|{}\n".format(
    utt,
    spk,
    language,
    norm_text,
    " ".join(phones),
    " ".join([str(i) for i in tones]),
    " ".join([str(i) for i in word2ph]),
)

# 捕获异常并打印相关信息
except Exception as e:
    print(line)
    print(f"生成训练集和验证集时发生错误！, 详细信息:\n{e}")

# 设置变量 transcription_path 为 cleaned_path 的值
transcription_path = cleaned_path
# 创建一个空的字典，用于存储说话人和句子的映射关系
spk_utt_map = defaultdict(list)
# 创建一个空的字典，用于存储说话人和ID的映射关系
spk_id_map = {}
# 设置变量 current_sid 的初始值为 0
current_sid = 0

# 打开指定路径的文件，以只读模式，并指定编码为 utf-8
with open(transcription_path, "r", encoding="utf-8") as f:
        audioPaths = set()  # 创建一个空集合，用于存储音频文件的路径
        countSame = 0  # 初始化计数器，用于记录相同音频的数量
        countNotFound = 0  # 初始化计数器，用于记录找不到音频的数量
        for line in f.readlines():  # 遍历文件对象f的每一行内容
            utt, spk, language, text, phones, tones, word2ph = line.strip().split("|")  # 将每一行内容按照"|"分隔，并赋值给对应的变量
            if utt in audioPaths:  # 判断当前音频路径是否已经存在于集合audioPaths中
                # 过滤数据集错误：相同的音频匹配多个文本，导致后续bert出问题
                print(f"重复音频文本：{line}")  # 打印出重复音频的文本内容
                countSame += 1  # 计数器countSame加1
                continue  # 跳过当前循环，继续下一次循环
            if not os.path.isfile(utt):  # 判断当前音频文件是否存在
                # 过滤数据集错误：不存在对应音频
                print(f"没有找到对应的音频：{utt}")  # 打印出找不到音频的路径
                countNotFound += 1  # 计数器countNotFound加1
                continue  # 跳过当前循环，继续下一次循环
            audioPaths.add(utt)  # 将当前音频路径添加到集合audioPaths中
            spk_utt_map[language].append(line)  # 将当前行内容添加到以language为键的字典spk_utt_map中
            if spk not in spk_id_map.keys():  # 判断当前说话人是否已经存在于字典spk_id_map的键中
                spk_id_map[spk] = current_sid  # 将当前说话人添加到字典spk_id_map中，并赋值为current_sid
                current_sid += 1  # current_sid加1
        print(f"总重复音频数：{countSame}，总未找到的音频数:{countNotFound}")
```
这行代码用于打印输出一个字符串，其中包含了变量`countSame`和`countNotFound`的值。

```
    train_list = []
    val_list = []
```
这两行代码创建了两个空列表`train_list`和`val_list`。

```
    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        val_list += utts[:val_per_lang]
        train_list += utts[val_per_lang:]
```
这个循环遍历`spk_utt_map`字典的键值对。对于每个键值对，将其值`utts`进行随机打乱。然后，将`utts`列表的前`val_per_lang`个元素添加到`val_list`列表中，将剩余的元素添加到`train_list`列表中。

```
    shuffle(val_list)
    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]
```
这段代码将`val_list`列表进行随机打乱。如果`val_list`列表的长度大于`max_val_total`，则将`val_list`列表的第`max_val_total`个元素及之后的元素添加到`train_list`列表中，并将`val_list`列表截取为前`max_val_total`个元素。

```
    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)
```
这段代码打开一个文件`train_path`，以写入模式打开，并使用UTF-8编码。然后，遍历`train_list`列表中的每个元素`line`，将其写入文件中。

```
    with open(val_path, "w", encoding="utf-8") as f:
```
这段代码打开一个文件`val_path`，以写入模式打开，并使用UTF-8编码。
        for line in val_list:
            f.write(line)
```
这段代码是一个for循环，遍历val_list列表中的每个元素，并将每个元素写入文件f中。

```
    json_config = json.load(open(config_path, encoding="utf-8"))
    json_config["data"]["spk2id"] = spk_id_map
    json_config["data"]["n_speakers"] = len(spk_id_map)
    # 新增写入：写入训练版本、数据集路径
    json_config["version"] = latest_version
    json_config["data"]["training_files"] = os.path.normpath(train_path).replace(
        "\\", "/"
    )
    json_config["data"]["validation_files"] = os.path.normpath(val_path).replace(
        "\\", "/"
    )
```
这段代码读取一个JSON配置文件，并对其中的一些字段进行修改。首先，使用json.load函数打开并解析配置文件，将解析结果存储在json_config变量中。然后，通过修改json_config中的"data"字段下的"spk2id"和"n_speakers"字段，将它们的值分别设置为spk_id_map和spk_id_map的长度。接下来，将最新版本号latest_version赋值给json_config中的"version"字段。最后，使用os.path.normpath函数规范化train_path和val_path的路径，并将路径中的反斜杠替换为正斜杠，然后将结果分别赋值给json_config中的"data"字段下的"training_files"和"validation_files"字段。

```
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(json_config, f, indent=2, ensure_ascii=False)
    print("训练集和验证集生成完成！")
```
这段代码打开一个文件config_path，并以写入模式打开。然后，使用json.dump函数将json_config中的内容以JSON格式写入文件f中，其中indent参数设置缩进为2个空格，ensure_ascii参数设置为False以支持非ASCII字符。最后，打印出"训练集和验证集生成完成！"的提示信息。

```
if __name__ == "__main__":
```
这段代码是一个条件语句，判断当前模块是否作为主程序运行。只有当当前模块作为主程序运行时，才会执行if语句块中的代码。
# 调用preprocess()函数，进行预处理操作
```