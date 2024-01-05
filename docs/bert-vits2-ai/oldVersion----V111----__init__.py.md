# `d:/src/tocomm/Bert-VITS2\oldVersion\V111\__init__.py`

```
"""
1.1.1版本兼容
https://github.com/fishaudio/Bert-VITS2/releases/tag/1.1.1
"""
# 导入torch模块
import torch
# 导入commons模块
import commons
# 导入text模块中的clean_text和clean_text_fix函数
from .text.cleaner import clean_text, clean_text_fix
# 导入text模块中的cleaned_text_to_sequence函数
from .text import cleaned_text_to_sequence
# 导入text模块中的get_bert和get_bert_fix函数
from .text import get_bert, get_bert_fix

# 定义函数get_text，接收参数text, language_str, hps, device
def get_text(text, language_str, hps, device):
    # 调用clean_text函数，对text和language_str进行清洗，返回norm_text, phone, tone, word2ph
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    # 调用cleaned_text_to_sequence函数，将phone, tone, language转换为序列
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    # 如果hps.data.add_blank为True
    if hps.data.add_blank:
        # 在phone序列中插入0
        phone = commons.intersperse(phone, 0)
        # 在tone序列中插入0
        tone = commons.intersperse(tone, 0)
        # 在language序列中插入0
        language = commons.intersperse(language, 0)
        # 遍历word2ph列表
        for i in range(len(word2ph)):
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

```
word2ph[i] = word2ph[i] * 2
```
将`word2ph`字典中索引为`i`的值乘以2。

```
word2ph[0] += 1
```
将`word2ph`字典中索引为0的值加1。

```
bert = get_bert(norm_text, word2ph, language_str, device)
```
调用`get_bert`函数，传入`norm_text`、`word2ph`、`language_str`和`device`作为参数，将返回值赋给变量`bert`。

```
del word2ph
```
删除变量`word2ph`。

```
assert bert.shape[-1] == len(phone), phone
```
断言`bert`的最后一个维度的长度等于`phone`的长度，如果不相等，则抛出异常并显示`phone`的值。

```
if language_str == "ZH":
    bert = bert
    ja_bert = torch.zeros(768, len(phone))
elif language_str == "JP":
    ja_bert = bert
    bert = torch.zeros(1024, len(phone))
else:
    bert = torch.zeros(1024, len(phone))
    ja_bert = torch.zeros(768, len(phone))
```
根据`language_str`的值，分别对`bert`和`ja_bert`进行赋值。如果`language_str`为"ZH"，则`bert`保持不变，`ja_bert`被赋值为一个形状为(768, len(phone))的全零张量；如果`language_str`为"JP"，则`ja_bert`被赋值为`bert`，`bert`被赋值为一个形状为(1024, len(phone))的全零张量；否则，`bert`和`ja_bert`都被赋值为形状分别为(1024, len(phone))和(768, len(phone))的全零张量。

```
assert bert.shape[-1] == len(
    phone
), f"Bert seq len {bert.shape[-1]} != {len(phone)}"
```
断言`bert`的最后一个维度的长度等于`phone`的长度，如果不相等，则抛出异常并显示错误信息。
phone = torch.LongTensor(phone)
# 将phone列表转换为torch的LongTensor类型

tone = torch.LongTensor(tone)
# 将tone列表转换为torch的LongTensor类型

language = torch.LongTensor(language)
# 将language列表转换为torch的LongTensor类型

return bert, ja_bert, phone, tone, language
# 返回bert, ja_bert, phone, tone, language这五个变量
```

```
def get_text_fix(text, language_str, hps, device):
    # 对文本进行清洗，得到规范化的文本、phone列表、tone列表和word2ph字典
    norm_text, phone, tone, word2ph = clean_text_fix(text, language_str)
    # 将phone、tone和language转换为序列
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        # 如果hps.data.add_blank为True，则在phone、tone和language中插入0
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        # 将word2ph中的每个值乘以2
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        # 将word2ph的第一个值加1
        word2ph[0] += 1
    # 获取bert
    bert = get_bert_fix(norm_text, word2ph, language_str, device)
    # 删除word2ph变量
    del word2ph
    # 断言bert的最后一个维度与phone的长度相等
    assert bert.shape[-1] == len(phone), phone
    if language_str == "ZH":
        bert = bert
        ja_bert = torch.zeros(768, len(phone))
    elif language_str == "JP":
        ja_bert = bert
        bert = torch.zeros(1024, len(phone))
    else:
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))
```
- 这段代码根据`language_str`的值来决定`bert`和`ja_bert`的赋值。
- 如果`language_str`的值是"ZH"，则`bert`保持不变，`ja_bert`被赋值为一个形状为(768, len(phone))的全零张量。
- 如果`language_str`的值是"JP"，则`ja_bert`被赋值为`bert`的值，`bert`被赋值为一个形状为(1024, len(phone))的全零张量。
- 如果`language_str`的值既不是"ZH"也不是"JP"，则`bert`被赋值为一个形状为(1024, len(phone))的全零张量，`ja_bert`被赋值为一个形状为(768, len(phone))的全零张量。

```
    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"
```
- 这行代码用于断言`bert`的最后一个维度的大小与`phone`的长度相等。
- 如果不相等，则会抛出一个断言错误，错误信息中包含`bert`的最后一个维度的大小和`phone`的长度。

```
    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, phone, tone, language
```
- 这段代码将`phone`、`tone`和`language`转换为`torch.LongTensor`类型的张量。
- 然后将`bert`、`ja_bert`、`phone`、`tone`和`language`作为结果返回。
# 推断函数，根据给定的参数进行文本推断
def infer(
    text,  # 输入的文本
    sdp_ratio,  # SDP 比例
    noise_scale,  # 噪声比例
    noise_scale_w,  # 噪声比例（权重）
    length_scale,  # 长度比例
    sid,  # 会话 ID
    language,  # 语言
    hps,  # 超参数
    net_g,  # 生成网络
    device,  # 设备
):
    # 获取文本的 BERT 编码、日语 BERT 编码、音素、音调和语言 ID
    bert, ja_bert, phones, tones, lang_ids = get_text(text, language, hps, device)
    # 禁用梯度计算
    with torch.no_grad():
        # 将音素转换为张量，并在第一个维度上添加一个维度
        x_tst = phones.to(device).unsqueeze(0)
        # 将音调转换为张量，并在第一个维度上添加一个维度
        tones = tones.to(device).unsqueeze(0)
        # 将语言 ID 转换为张量，并在第一个维度上添加一个维度
        lang_ids = lang_ids.to(device).unsqueeze(0)
        # 将 BERT 编码转换为张量，并在第一个维度上添加一个维度
        bert = bert.to(device).unsqueeze(0)
        # 将日语 BERT 编码转换为张量，并在第一个维度上添加一个维度
        ja_bert = ja_bert.to(device).unsqueeze(0)
```

这段代码是一个推断函数，根据给定的参数对文本进行推断。具体注释如下：

- `text`: 输入的文本
- `sdp_ratio`: SDP 比例
- `noise_scale`: 噪声比例
- `noise_scale_w`: 噪声比例（权重）
- `length_scale`: 长度比例
- `sid`: 会话 ID
- `language`: 语言
- `hps`: 超参数
- `net_g`: 生成网络
- `device`: 设备

接下来的代码是对输入文本进行处理和转换的步骤：

- `bert, ja_bert, phones, tones, lang_ids = get_text(text, language, hps, device)`: 调用`get_text`函数获取文本的 BERT 编码、日语 BERT 编码、音素、音调和语言 ID。

然后使用`torch.no_grad()`禁用梯度计算，以提高推断的效率。

接下来的代码是将获取到的数据转换为张量，并在第一个维度上添加一个维度：

- `x_tst = phones.to(device).unsqueeze(0)`: 将音素转换为张量，并在第一个维度上添加一个维度
- `tones = tones.to(device).unsqueeze(0)`: 将音调转换为张量，并在第一个维度上添加一个维度
- `lang_ids = lang_ids.to(device).unsqueeze(0)`: 将语言 ID 转换为张量，并在第一个维度上添加一个维度
- `bert = bert.to(device).unsqueeze(0)`: 将 BERT 编码转换为张量，并在第一个维度上添加一个维度
- `ja_bert = ja_bert.to(device).unsqueeze(0)`: 将日语 BERT 编码转换为张量，并在第一个维度上添加一个维度
# 创建一个 LongTensor，其中包含一个元素，该元素是 phones 的大小，然后将其发送到设备上
x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)

# 删除变量 phones，释放内存
del phones

# 创建一个 LongTensor，其中包含一个元素，该元素是根据 sid 在 hps.data.spk2id 中查找到的值，然后将其发送到设备上
speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)

# 调用 net_g.infer 方法，传入多个参数，返回一个结果
# 结果是一个 Tensor，通过链式调用 .data.cpu().float().numpy() 将其转换为 CPU 上的 float 类型的 numpy 数组
audio = (
    net_g.infer(
        x_tst,
        x_tst_lengths,
        speakers,
        tones,
        lang_ids,
        bert,
        ja_bert,
        sdp_ratio=sdp_ratio,
        noise_scale=noise_scale,
        noise_scale_w=noise_scale_w,
        length_scale=length_scale,
    )[0][0, 0]
    .data.cpu()
    .float()
    .numpy()
)
        )
```
这是一个多行代码的结束括号，用于结束前面的代码块。

```
        del x_tst, x_tst_lengths, speakers, tones, lang_ids, bert, ja_bert
```
这行代码用于删除变量x_tst, x_tst_lengths, speakers, tones, lang_ids, bert, ja_bert，释放内存空间。

```
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```
这是一个条件语句，判断当前是否有可用的CUDA设备。如果有可用的CUDA设备，则调用torch.cuda.empty_cache()函数来清空CUDA缓存，释放GPU内存。

```
        return audio
```
这行代码用于返回变量audio作为函数的结果。

```
def infer_fix(
    text,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    sid,
    language,
    hps,
    net_g,
    device,
):
```
这是一个函数定义，函数名为infer_fix，接受的参数有text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid, language, hps, net_g, device。

```
    bert, ja_bert, phones, tones, lang_ids = get_text_fix(text, language, hps, device)
```
这行代码调用了一个名为get_text_fix的函数，并将其返回值赋值给变量bert, ja_bert, phones, tones, lang_ids。函数get_text_fix接受参数text, language, hps, device。
    with torch.no_grad():
```
使用`torch.no_grad()`上下文管理器，表示在该上下文中不需要计算梯度，可以提高代码的执行效率。

```
        x_tst = phones.to(device).unsqueeze(0)
```
将`phones`张量移动到指定的设备上，并在第0维度上添加一个维度。

```
        tones = tones.to(device).unsqueeze(0)
```
将`tones`张量移动到指定的设备上，并在第0维度上添加一个维度。

```
        lang_ids = lang_ids.to(device).unsqueeze(0)
```
将`lang_ids`张量移动到指定的设备上，并在第0维度上添加一个维度。

```
        bert = bert.to(device).unsqueeze(0)
```
将`bert`张量移动到指定的设备上，并在第0维度上添加一个维度。

```
        ja_bert = ja_bert.to(device).unsqueeze(0)
```
将`ja_bert`张量移动到指定的设备上，并在第0维度上添加一个维度。

```
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
```
创建一个包含`phones`张量第0维度大小的长整型张量，并将其移动到指定的设备上。

```
        del phones
```
删除`phones`变量，释放内存。

```
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
```
根据`sid`从`hps.data.spk2id`字典中获取对应的值，创建一个包含该值的长整型张量，并将其移动到指定的设备上。

```
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                ja_bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
```
调用`net_g.infer()`方法，传入多个参数进行推断，返回推断结果赋值给`audio`变量。其中包括`x_tst`、`x_tst_lengths`、`speakers`、`tones`、`lang_ids`、`bert`、`ja_bert`等参数。
# 根据给定的参数生成音频数据
def generate_audio(
    model,
    x_tst,
    x_tst_lengths,
    speakers,
    tones,
    lang_ids,
    bert,
    ja_bert,
    noise_scale_w,
    length_scale,
):
    # 使用模型生成音频数据
    audio = (
        model.generate(
            x_tst,
            x_tst_lengths,
            speakers,
            tones,
            lang_ids,
            bert,
            ja_bert,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        )[0][0, 0]
        .data.cpu()
        .float()
        .numpy()
    )
    # 释放不再使用的变量
    del x_tst, x_tst_lengths, speakers, tones, lang_ids, bert, ja_bert
    # 如果有可用的 GPU，则清空 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # 返回生成的音频数据
    return audio
```