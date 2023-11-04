# BertVITS2源码解析 3

# `webui.py`

这段代码的作用是设置多个日志输出来源的级别，并为它们添加了一个无警告级别（WARNING），以便我们能够在程序运行时遇到警告时进行记录。

具体来说，这段代码：

1. 通过 `import sys，os` 导入操作系统中的两个标准输入输出流 `sys.stdout` 和 `sys.stderr`，以及第三方库中的 `logging` 模块。
2. 通过 `logging.getLogger` 函数获取一个名为 `numba` 的日志输出源的实例，并将其设置为 `logging.WARNING` 级别。
3. 通过 `logging.getLogger` 函数获取一个名为 `markdown_it` 的日志输出源的实例，并将其设置为 `logging.WARNING` 级别。
4. 通过 `logging.getLogger` 函数获取一个名为 `urllib3` 的日志输出源的实例，并将其设置为 `logging.WARNING` 级别。
5. 通过 `logging.getLogger` 函数获取一个名为 `matplotlib` 的日志输出源的实例，并将其设置为 `logging.WARNING` 级别。
6. 通过 `logging.basicConfig` 函数设置一个无警告级别的格式，其中包括：
  ```pyyaml
  - 格式字符串 `"%(name)s"`：将 `%(name)s` 替换为 `name` 对象，如果 `name` 是 `None`，则不替换。
  - 格式字符串 `"%(levelname)s"`：将 `%(levelname)s` 替换为 `levelname` 对象。
  - 级别字符串 `"%(message)s"`：将 `%(message)s` 替换为 `message` 对象。
  ```
  
  `getLogger` 函数获取指定名称的日志输出源实例，`setLevel` 函数将日志输出源的级别设置为指定的警告级别。
  
  `logging.basicConfig` 函数的 `level` 参数指定了日志输出源的最低警告级别，`format` 参数指定了格式字符串，其中包括了时间戳、日志名称、日志级别和消息。


```py
# flake8: noqa: E402

import sys, os
import logging

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

```

这段代码是一个机器学习模型的前处理和后处理代码，属于自然语言处理(NLP)和机器学习(ML)领域。具体来说，它包括以下几个部分：

- 导入必要的模块和函数：import torch, argparse, commons, utils, models, Gradio, webbrowser, numpy as np。
- 定义了一些变量：net_g, which is the variable that stores the instance of the pre-trained language model to use for text generation,argparse.ArgumentParser, which is for parsing command-line arguments, and utils.
- 加载预训练的语言模型：from models import SynthesizerTrn, get_bert。
- 定义函数清理文本：clean_text, which takes a text string as input and returns a cleaned version of that text.
- 加载已经处理过的文本数据：from text.symbols import symbols, symbols.风生水起
- 定义函数将文本序列化为模型可读取的格式：to_sequences, which takes a text string as input and returns a list of integers representing the tokens in the input text.
- 加载已经清洗过的文本数据：from text.cleaner import clean, cleaned_text_to_sequence
- 定义函数生成文本：generate_text, which takes a text sequence作为输入， generates a text string with the pre-trained language model.
- 定义函数输出清洗后的文本数据：outputs, which takes a text sequence作为输入， generates the cleaned text output.
- 加载已经处理过的文本数据：from webbrowser import webbrowser, which is for interacting with a web browser.
- 将函数作为参数传递给gradio:gr.generated，以便在Gradio中使用。
- 保存函数代码为文件：with open("script.py", "w") as f:f.write(str(net_g))
- 运行函数以生成文本：net_g.generate_text("这是一个干净的文本")
- 运行函数以在网页上查看生成的文本：webbrowser.run_tag("http://localhost:5000")

函数 net_g 是一个模型实例，通过调用 model.preprocess() 函数来预处理文本数据，返回经过清理和序列化的文本数据。函数 clean_text() 函数将输入的文本数据进行清洗，并返回一个 cleaned 的文本数据。函数 generate_text() 函数将经过预处理和清理的文本数据序列化为模型可读取的格式，并返回生成的文本数据。函数 outputs.generate_text() 将 generate_text() 函数生成的文本数据作为参数，执行清理和序列化操作，并返回清洗后的文本数据。函数 webbrowser.run_tag() 是用来在网页上查看生成的文本数据的函数，它通过 webbrowser.py 库实现。


```py
import torch
import argparse
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence, get_bert
from text.cleaner import clean_text
import gradio as gr
import webbrowser
import numpy as np

net_g = None

if sys.platform == "darwin" and torch.backends.mps.is_available():
    device = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```

这段代码定义了一个名为 `get_text` 的函数，它接受一个文本 `text`，一个语言字符串 `language_str` 和一个二进制数据集 `hps`。函数的主要作用是预处理文本数据并将其转换成可以用于 BERT 模型的格式。

具体来说，函数首先通过 `clean_text` 函数清洗文本数据，然后通过 `cleaned_text_to_sequence` 函数将文本数据转换为序列数据。接下来，函数根据 `hps.data.add_blank` 的值来决定是否在数据集中添加空白。如果添加空白，则函数会计算每个单词的词向量，并将这些词向量与空白处的单词向量进行拼接。然后，函数使用 `get_bert` 函数获取预处理后的文本数据，并将这些数据与原始单词向量 `word2ph` 组合成文本数据。最后，函数返回处理后的文本数据、BERT 模型、语言分布和原始单词向量。


```py
else:
    device = "cuda"


def get_text(text, language_str, hps):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert = get_bert(norm_text, word2ph, language_str, device)
    del word2ph
    assert bert.shape[-1] == len(phone), phone

    if language_str == "ZH":
        bert = bert
        ja_bert = torch.zeros(768, len(phone))
    elif language_str == "JA":
        ja_bert = bert
        bert = torch.zeros(1024, len(phone))
    else:
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, phone, tone, language


```

该函数的作用是利用PyTorch中的文本预处理技术和深度学习模型BERT，对输入的文本进行预处理和转录，然后对预处理后的文本进行语音合成，并将输出的音频数据进行去噪处理。

具体来说，该函数的实现过程如下：

1. 读取输入文本并获取其语言编号、预处理后的文本数据以及模型的参数。
2. 将输入文本和语言编号转换为PyTorch中的LongTensor对象，并使用这些对象创建一个包含所有模型的参数的张量。
3. 使用BERT模型对输入文本进行预处理和转录，并获取模型的输出。
4. 对预处理后的文本数据进行去噪处理，并返回处理后的音频数据。

该函数的作用是将输入文本转化为音频数据，以便在应用程序中进行语音合成。该函数需要使用PyTorch中的BERT模型和相应的数据预处理技术，因此需要安装相应的PyTorch包。


```py
def infer(text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid, language):
    global net_g
    bert, ja_bert, phones, tones, lang_ids = get_text(text, language, hps)
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        del phones
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
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
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers
        return audio


```

This is a JavaScript code to create a simple葡萄生产者-消费者应用。这个应用包括文本输入框、文本输出框、音频输出。当点击“生成”按钮时，它将在控制台中输出葡萄的相关信息（吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮）。

```pyjavascript
// HTML markup
<html>
 <head>
   <meta charset="utf-8">
   <meta name="viewport" content="width=device-width, initial-scale=1">
   <link rel="stylesheet" href="https://css.opera.com/display-static/8.0/星辰.css">
   <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" integrity="sha384-LLyX+yJLrKGL7Rk6thjhTWtVdY/6KpJuPcVZ7Z7BCk==" crossorigin="anonymous" />
   <script src="https://cdn.jsdelivr.net/npm/remreact@2.90.0/umd/react.production.min.js"></script>
   <script src="https://cdn.jsdelivr.net/npm/react-dom@17.0.0/umd/react-dom.production.min.js"></script>
   <script src="https://cdn.jsdelivr.net/npm/axios@2.90.0/dist/axios.min.js"></script>
   <script>
     const dp = document.documentElement;
     const span = document.createElement(" span");
     span.classList.add("input-text");
     span.style.color = "white";
     span.textContent = "吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。";
     document.body.appendChild(span);

     const rate = dp.getElementsByTagName("speaker")[0].value;
     const sdpRatio = dp.getElementsByTagName("sdp-ratio")[0].value;
     const noiseScale = dp.getElementsByTagName("noise-scale")[0].value;
     const noiseScaleW = dp.getElementsByTagName("noise-scale-w")[0].value;
     const lengthScale = dp.getElementsByTagName("length-scale")[0].value;
     const language = dp.getElementsByTagName("language")[0].value;
     const outputText = document.createElement("p");
     outputText.classList.add("output-text");
     outputText.style.color = "white";
     outputText.textContent = `${language}的葡萄产地和品种是：${{{ Speaking. talking("zh-CN").match(/葡萄产地和品种是：(.+)/)[1] }}\n`;
     document.body.appendChild(outputText);

     const tts_fn = async () => {
       const url = `https://api.openapi.org/graphql`;
       const response = await fetch(url, {
         method: "POST",
         headers: {
           "Content-Type": "application/json",
           "Authorization": `Bearer ${dp.getElementsByTagName("token")[0].value}`
         },
         body: JSON.stringify({
           query: `
             {
               speaker: ${speaker},
               sdp_ratio: ${sdpRatio},
               noise_scale: ${noiseScale},
               noise_scale_w: ${noiseScaleW},
               length_scale: ${lengthScale},
               language: ${language},
               output_audio: true
             }
           `
         })
       });
       const data = await response.json();
       const outputAudio = data.output_audio;
       const responseMessage = data.message;
       outputText.textContent = `${outputAudio ? '' : <audio src="${outputMessage}" controls></audio>`;
     };

     document.addEventListener("keyup", function(e) {
       if (e.key === "Enter") {
         tts_fn();
       }
     });

     tts_fn();
   </script>
 </body>
</html>
```


```py
def tts_fn(text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale):
    slices = text.split("|")
    audio_list = []
    with torch.no_grad():
        for slice in slices:
            audio = infer(slice, sdp_ratio=sdp_ratio, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale, sid=speaker)
            audio_list.append(audio)
            silence = np.zeros(hps.data.sampling_rate)  # 生成1秒的静音
            audio_list.append(silence)  # 将静音添加到列表中
    audio_concat = np.concatenate(audio_list)
    return "Success", (hps.data.sampling_rate, audio_concat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", default="./logs/as/G_8000.pth", help="path of your model"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="./configs/config.json",
        help="path of your config file",
    )
    parser.add_argument(
        "--share", default=False, help="make link public", action="store_true"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="enable DEBUG-LEVEL log"
    )

    args = parser.parse_args()
    if args.debug:
        logger.info("Enable DEBUG-LEVEL log")
        logging.basicConfig(level=logging.DEBUG)
    hps = utils.get_hparams_from_file(args.config)

    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else (
            "mps"
            if sys.platform == "darwin" and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.model, net_g, None, skip_optimizer=True)

    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    languages = ["ZH", "JP"]
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                text = gr.TextArea(
                    label="Text",
                    placeholder="Input Text Here",
                    value="吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。",
                )
                speaker = gr.Dropdown(
                    choices=speakers, value=speakers[0], label="Speaker"
                )
                sdp_ratio = gr.Slider(
                    minimum=0, maximum=1, value=0.2, step=0.1, label="SDP Ratio"
                )
                noise_scale = gr.Slider(
                    minimum=0.1, maximum=2, value=0.6, step=0.1, label="Noise Scale"
                )
                noise_scale_w = gr.Slider(
                    minimum=0.1, maximum=2, value=0.8, step=0.1, label="Noise Scale W"
                )
                length_scale = gr.Slider(
                    minimum=0.1, maximum=2, value=1, step=0.1, label="Length Scale"
                )
                language = gr.Dropdown(
                    choices=languages, value=languages[0], label="Language"
                )
                btn = gr.Button("Generate!", variant="primary")
            with gr.Column():
                text_output = gr.Textbox(label="Message")
                audio_output = gr.Audio(label="Output Audio")

        btn.click(
            tts_fn,
            inputs=[
                text,
                speaker,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                language,
            ],
            outputs=[text_output, audio_output],
        )

    webbrowser.open("http://127.0.0.1:7860")
    app.launch(share=args.share)

```

---
license: apache-2.0
datasets:
- cc100
- wikipedia
language:
- ja
widget:
- text: 東北大学で[MASK]の研究をしています。
---

# BERT base Japanese (unidic-lite with whole word masking, CC-100 and jawiki-20230102)

This is a [BERT](https://github.com/google-research/bert) model pretrained on texts in the Japanese language.

This version of the model processes input texts with word-level tokenization based on the Unidic 2.1.2 dictionary (available in [unidic-lite](https://pypi.org/project/unidic-lite/) package), followed by the WordPiece subword tokenization.
Additionally, the model is trained with the whole word masking enabled for the masked language modeling (MLM) objective.

The codes for the pretraining are available at [cl-tohoku/bert-japanese](https://github.com/cl-tohoku/bert-japanese/).

## Model architecture

The model architecture is the same as the original BERT base model; 12 layers, 768 dimensions of hidden states, and 12 attention heads.

## Training Data

The model is trained on the Japanese portion of [CC-100 dataset](https://data.statmt.org/cc-100/) and the Japanese version of Wikipedia.
For Wikipedia, we generated a text corpus from the [Wikipedia Cirrussearch dump file](https://dumps.wikimedia.org/other/cirrussearch/) as of January 2, 2023.
The corpus files generated from CC-100 and Wikipedia are 74.3GB and 4.9GB in size and consist of approximately 392M and 34M sentences, respectively.

For the purpose of splitting texts into sentences, we used [fugashi](https://github.com/polm/fugashi) with [mecab-ipadic-NEologd](https://github.com/neologd/mecab-ipadic-neologd) dictionary (v0.0.7).

## Tokenization

The texts are first tokenized by MeCab with the Unidic 2.1.2 dictionary and then split into subwords by the WordPiece algorithm.
The vocabulary size is 32768.

We used [fugashi](https://github.com/polm/fugashi) and [unidic-lite](https://github.com/polm/unidic-lite) packages for the tokenization.

## Training

We trained the model first on the CC-100 corpus for 1M steps and then on the Wikipedia corpus for another 1M steps.
For training of the MLM (masked language modeling) objective, we introduced whole word masking in which all of the subword tokens corresponding to a single word (tokenized by MeCab) are masked at once.

For training of each model, we used a v3-8 instance of Cloud TPUs provided by [TPU Research Cloud](https://sites.research.google/trc/about/).

## Licenses

The pretrained models are distributed under the Apache License 2.0.

## Acknowledgments

This model is trained with Cloud TPUs provided by [TPU Research Cloud](https://sites.research.google/trc/about/) program.


---
language:
- zh
tags:
- bert
license: "apache-2.0"
---

# Please use 'Bert' related functions to load this model!

## Chinese BERT with Whole Word Masking
For further accelerating Chinese natural language processing, we provide **Chinese pre-trained BERT with Whole Word Masking**.

**[Pre-Training with Whole Word Masking for Chinese BERT](https://arxiv.org/abs/1906.08101)**
Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Ziqing Yang, Shijin Wang, Guoping Hu

This repository is developed based on：https://github.com/google-research/bert

You may also interested in,
- Chinese BERT series: https://github.com/ymcui/Chinese-BERT-wwm
- Chinese MacBERT: https://github.com/ymcui/MacBERT
- Chinese ELECTRA: https://github.com/ymcui/Chinese-ELECTRA
- Chinese XLNet: https://github.com/ymcui/Chinese-XLNet
- Knowledge Distillation Toolkit - TextBrewer: https://github.com/airaria/TextBrewer

More resources by HFL: https://github.com/ymcui/HFL-Anthology

## Citation
If you find the technical report or resource is useful, please cite the following technical report in your paper.
- Primary: https://arxiv.org/abs/2004.13922
```py
@inproceedings{cui-etal-2020-revisiting,
    title = "Revisiting Pre-Trained Models for {C}hinese Natural Language Processing",
    author = "Cui, Yiming  and
      Che, Wanxiang  and
      Liu, Ting  and
      Qin, Bing  and
      Wang, Shijin  and
      Hu, Guoping",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.58",
    pages = "657--668",
}
```
- Secondary: https://arxiv.org/abs/1906.08101
```py
@article{chinese-bert-wwm,
  title={Pre-Training with Whole Word Masking for Chinese BERT},
  author={Cui, Yiming and Che, Wanxiang and Liu, Ting and Qin, Bing and Yang, Ziqing and Wang, Shijin and Hu, Guoping},
  journal={arXiv preprint arXiv:1906.08101},
  year={2019}
 }
```


# `monotonic_align/core.py`

这段代码使用 Numba JIT 编译器来优化一个名为 "maximum_path" 的函数。函数的输入参数包括一个二维路径数组 "paths" 和一个二维数值数组 "values"，以及两个时间步 "t_ys" 和 "t_xs"。函数返回一个整数。

具体来说，这段代码定义了一个名为 "maximum_path_jit" 的函数，它接受四个参数：路径数组 "paths"、数值数组 "values"、时间步 "t_ys" 和时间步 "t_xs"。函数的主要作用是计算给定路径和数值数组中的数据，使得所有路径中的时间步都包含相同的最小值，即到达时间的最小值。

函数的实现包括以下几个步骤：

1. 初始化变量，包括路径数组的第一个元素、数值数组的第一个元素、路径数组的形状为 1 的一维片段和数值数组的第一个元素，以及一个变量 "max_neg_val"，表示最大负值。

2. 遍历路径数组的每个元素和数值数组的每个元素。

3. 对于路径数组的每个元素，对于每个时间步 t_y 和时间步 t_x，按照时间步 t_x 和路径数组元素 at least t_y - t_y中的一致，计算当前时间步的值。

4. 对于数值数组的每个元素，初始化其值为路径数组中对应元素的最大负值。

5. 遍历路径数组的每个元素和数值数组的每个元素，对于每个时间步 t_y 和时间步 t_x，按照时间步 t_x 和路径数组元素 at least t_y - t_y中的一致，计算当前时间步的值。

6. 对于路径数组的每个元素，遍历时间步 t_y 和时间步 t_x，对于每个时间步 t_x，计算当前时间步的值。

7. 对于数值数组的每个元素，按照时间步 t_y 和路径数组中对应元素的一致性，更新其值。


```py
import numba


@numba.jit(
    numba.void(
        numba.int32[:, :, ::1],
        numba.float32[:, :, ::1],
        numba.int32[::1],
        numba.int32[::1],
    ),
    nopython=True,
    nogil=True,
)
def maximum_path_jit(paths, values, t_ys, t_xs):
    b = paths.shape[0]
    max_neg_val = -1e9
    for i in range(int(b)):
        path = paths[i]
        value = values[i]
        t_y = t_ys[i]
        t_x = t_xs[i]

        v_prev = v_cur = 0.0
        index = t_x - 1

        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                if x == y:
                    v_cur = max_neg_val
                else:
                    v_cur = value[y - 1, x]
                if x == 0:
                    if y == 0:
                        v_prev = 0.0
                    else:
                        v_prev = max_neg_val
                else:
                    v_prev = value[y - 1, x - 1]
                value[y, x] += max(v_prev, v_cur)

        for y in range(t_y - 1, -1, -1):
            path[y, index] = 1
            if index != 0 and (
                index == y or value[y - 1, index] < value[y - 1, index - 1]
            ):
                index = index - 1

```

# `monotonic_align/__init__.py`

这段代码实现了一个名为“maximum_path”的函数，接受两个参数：一个表示负注意力分配掩码的二维数组，以及一个表示路径的二维数组。函数的主要目的是计算负注意力分配的最大值，并返回从根节点到当前节点的路径。

具体实现步骤如下：

1. 从PyTorch的ZeroProduct对负注意力分配掩码的每个元素求最大值，然后将结果保存到路径数组的对应元素中。
2. 从ZeroProduct中获取路径数组的起始索引，然后将起始索引和路径数组的对应元素连在一起，得到完整的路径。
3. 调用函数`maximum_path_jit`，该函数将创建一个按路径顺序排列的路径数组，并在路径上执行“`torch.no_grad`”以避免梯度计算过程中的梯度消失。
4. 从路径数组中获取前一个元素，将其设置为根节点，然后继续从路径数组中获取下一个元素，重复以上步骤，直到达到负注意力分配掩码的末尾。
5. 返回路径数组的最后一个元素，即从根节点到当前节点的路径。

该函数的作用是计算一个负注意力分配网络中两个注意力头之间的路径，并返回从根节点到当前节点的路径。这对于训练中学习如何计算路径很有用。


```py
from numpy import zeros, int32, float32
from torch import from_numpy

from .core import maximum_path_jit


def maximum_path(neg_cent, mask):
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = neg_cent.data.cpu().numpy().astype(float32)
    path = zeros(neg_cent.shape, dtype=int32)

    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(int32)
    maximum_path_jit(path, neg_cent, t_t_max, t_s_max)
    return from_numpy(path).to(device=device, dtype=dtype)

```

# `text/chinese.py`

这段代码的作用是：

1. 导入两个模块：os 和 re。
2. 从 pypinyin 导入 lazy\_pinyin。
3. 从 pypinyin 导入 Style。
4. 从 text.symbols 和 text.tone\_sandhi 导入一些工具函数。
5. 获取当前文件夹的路径。
6. 从 OpenCPOP 风格库导入 ToneSandhi。
7. 将 `cn2an` 和 `pinyin_to_symbol_map` 对象进行初始化。
8. 遍历当前文件夹下的所有文本文件，并将文件名和内容存储到映射中。
9. 支持 Pinyin 风格。
10. 支持 OpenCPOP 风格。


```py
import os
import re

import cn2an
from pypinyin import lazy_pinyin, Style

from text.symbols import punctuation
from text.tone_sandhi import ToneSandhi

current_file_path = os.path.dirname(__file__)
pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}

```

这段代码是一个命名字典，通过psg库实现了对中文分词的功能。通过psg库对一段中文文本进行分词，得到一个以冒号分割的词语列表，方便对词语进行操作。


```py
import jieba.posseg as psg


rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "“": "'",
    "”": "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}

```

这段代码的作用是定义了一个名为 `tone_modifier` 的变量，其值为一个经过修改后的文本。修改的方式包括：

1. 删除原文本中的标点符号，使用正则表达式 `re.escape(p)` 将其转义，其中 `p` 是文本中的一个字符串。
2. 将文本中的所有中文使用拼音代替，使用正则表达式 `re.escape(p).upper()` 转义，其中 `p` 是文本中的一个字符串。
3. 在替换语句中使用 `re.sub` 函数将所有中文拼音和替换后的字符串组合成的字符串替换掉原文本中的所有中文。
4. 使用正则表达式将文本中的标点符号替换成它本身，使用字符串方法 `replace()` 实现。


```py
tone_modifier = ToneSandhi()


def replace_punctuation(text):
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(
        r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text
    )

    return replaced_text


```



这段代码定义了一个名为 `g2p` 的函数，它接受一个字符串参数 `text`。函数的作用是检查文本中的语言信息，包括拼音和声调，并将它们转化为数字形式。

函数首先使用正则表达式匹配文本中的所有拼音，并将它们存储在一个列表中。然后，它使用 `re.split` 函数将拼音分离出来，并将它们转换为小写。接下来，它调用另一个名为 `_g2p` 的函数，这个函数也接受一个列表作为参数，并将它们与文本中的其他语言信息一起传递给 `_get_initials_finals` 函数。最后，它将生成的拼音、声调和拼音转换为数字形式，并将它们存储在结果列表中。

函数的完整实现如下：

```pypython

def g2p(text):
   pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
   sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
   phones, tones, word2ph = _g2p(sentences)
   assert sum(word2ph) == len(phones)
   assert len(word2ph) == len(text)  # Sometimes it will crash,you can add a try-catch.
   phones = ["_"] + phones + ["_"]
   tones = [0] + tones + [0]
   word2ph = [1] + word2ph + [1]
   return phones, tones, word2ph

def _get_initials_finals(word):
   initials = []
   finals = []
   orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
   orig_finals = lazy_pinyin(
       word, neutral_tone_with_five=True, style=Style.FINALS_TONE3
   )
   for c, v in zip(orig_initials, orig_finals):
       initials.append(c)
       finals.append(v)
   return initials, finals
```

其中，`lazy_pinyin` 函数用于生成拼音。拼音是指汉语拼音，包括声调。`neutral_tone_with_five` 是设置为使用五个中立元音的选项，这是根据音调是否准确来计算拼音的。`Style.INITIALS` 和 `Style.FINALS_TONE3` 指定了用于初始化和终止的拼音样式。


```py
def g2p(text):
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    phones, tones, word2ph = _g2p(sentences)
    assert sum(word2ph) == len(phones)
    assert len(word2ph) == len(text)  # Sometimes it will crash,you can add a try-catch.
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph


def _get_initials_finals(word):
    initials = []
    finals = []
    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(
        word, neutral_tone_with_five=True, style=Style.FINALS_TONE3
    )
    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    return initials, finals


```

This is a Python function that takes in a transcript of a Chinese sentence, segmented by lines and translated into Pinyin. It then normalizes the segmentation, tone mapping, and phone/word mapping for the segmented words.

The function takes in three arguments:

- `transcript`: The input transcript of the Chinese sentence, split by lines.
- `segmentation_map`: A dictionary that maps each segment of the sentence to its corresponding Chinese word.
- `pinyin_to_symbol_map`: A dictionary that maps each Pinyin sound to its corresponding symbol in the Pinyin dictionary.
- `phones_list`: A list that stores the phone calls in the segmented sentence, split by the换线符 (,).
- `tones_list`: A list that stores the tones in the segmented sentence, where each tone is represented by its corresponding index in the Pinyin dictionary.
- `word2ph`: A dictionary that maps each Chinese word to its corresponding index in the phone/word mapping dictionary.

The function returns three values:

- `phones_list`: The normalized phone calls in the segmented sentence.
- `tones_list`: The normalized tones in the segmented sentence.
- `word2ph`: The normalized word/phone mapping in the segmented sentence.

The function first preprocesses the input by segmenting the transcript into its constituent words, which are then converted to their Pinyin representation using the `pinyin_to_symbol_map` dictionary.

It then performs phone normalization by removing any calls that contain more than one character, as well as converting the tones to their corresponding index in the Pinyin dictionary.

Finally, it creates a dictionary that maps each Chinese word to its corresponding index in the `word2ph` dictionary and stores the normalized phone calls, tones, and word/phone mapping in the returned values.


```py
def _g2p(segments):
    phones_list = []
    tones_list = []
    word2ph = []
    for seg in segments:
        # Replace all English words in the sentence
        seg = re.sub("[a-zA-Z]+", "", seg)
        seg_cut = psg.lcut(seg)
        initials = []
        finals = []
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)
        for word, pos in seg_cut:
            if pos == "eng":
                continue
            sub_initials, sub_finals = _get_initials_finals(word)
            sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
            initials.append(sub_initials)
            finals.append(sub_finals)

            # assert len(sub_initials) == len(sub_finals) == len(word)
        initials = sum(initials, [])
        finals = sum(finals, [])
        #
        for c, v in zip(initials, finals):
            raw_pinyin = c + v
            # NOTE: post process for pypinyin outputs
            # we discriminate i, ii and iii
            if c == v:
                assert c in punctuation
                phone = [c]
                tone = "0"
                word2ph.append(1)
            else:
                v_without_tone = v[:-1]
                tone = v[-1]

                pinyin = c + v_without_tone
                assert tone in "12345"

                if c:
                    # 多音节
                    v_rep_map = {
                        "uei": "ui",
                        "iou": "iu",
                        "uen": "un",
                    }
                    if v_without_tone in v_rep_map.keys():
                        pinyin = c + v_rep_map[v_without_tone]
                else:
                    # 单音节
                    pinyin_rep_map = {
                        "ing": "ying",
                        "i": "yi",
                        "in": "yin",
                        "u": "wu",
                    }
                    if pinyin in pinyin_rep_map.keys():
                        pinyin = pinyin_rep_map[pinyin]
                    else:
                        single_rep_map = {
                            "v": "yu",
                            "e": "e",
                            "i": "y",
                            "u": "w",
                        }
                        if pinyin[0] in single_rep_map.keys():
                            pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

                assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
                phone = pinyin_to_symbol_map[pinyin].split(" ")
                word2ph.append(len(phone))

            phones_list += phone
            tones_list += [int(tone)] * len(phone)
    return phones_list, tones_list, word2ph


```

这段代码的主要作用是 normalize（使正常）和 preprocess（预处理）中文文本，以使其适合用于 BERT 模型输入。

具体来说，代码首先使用正则表达式从文本中提取出数字，并将它们替换为拼音。接下来，代码使用 replace\_punctuation() 函数删除字符串中的标点符号。最后，代码将生成的文本输入 get\_bert\_feature() 函数，以获取 BERT 模型对其的文本表示。

如果将文本 "啊！但是《原神》是由，米哈\游自主，  [研发]的一款全.新开放世界.冒险游戏" 经过上述预处理后，输出的结果将是非常适合 BERT 模型输入的文本。


```py
def text_normalize(text):
    numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    text = replace_punctuation(text)
    return text


def get_bert_feature(text, word2ph):
    from text import chinese_bert

    return chinese_bert.get_bert_feature(text, word2ph)


if __name__ == "__main__":
    from text.chinese_bert import get_bert_feature

    text = "啊！但是《原神》是由,米哈\游自主，  [研发]的一款全.新开放世界.冒险游戏"
    text = text_normalize(text)
    print(text)
    phones, tones, word2ph = g2p(text)
    bert = get_bert_feature(text, word2ph)

    print(phones, tones, word2ph, bert.shape)


```

这段代码定义了一个变量 `text` 并将其设置为字符串类型 `"这是一个示例文本：，你好！这是一个测试。"`。然后，定义了一个函数 `g2p_paddle`，并将 `text` 作为参数传递给它。最后，使用 `print` 函数将 `g2p_paddle` 的返回值输出，结果为 `"这是一个示例文本：你好这是一个测试。"`。


```py
# # 示例用法
# text = "这是一个示例文本：,你好！这是一个测试...."
# print(g2p_paddle(text))  # 输出: 这是一个示例文本你好这是一个测试

```

# `text/chinese_bert.py`

这段代码使用了PyTorch和transformers库实现了一个文本预处理和文本分类任务。

首先，通过导入 torch 和 sys，并使用 AutoTokenizer 和 AutoModelForMaskedLM 从 transformers 中加载了预训练的 BERT 模型，并定义了一个名为 get_bert_feature 的函数，用于获取 BERT 模型中的文本特征。

接着，代码创建了一个名为 models 的字典，用于存储 BERT 模型及其参数。

然后，代码定义了一个名为 main 的函数，作为程序的入口点。在这个函数中，首先加载了 BERT 模型，并创建了一个自定义的 AutoTokenizer 实例，用于从 BERT 模型中编码文本。接着，定义了一个名为 get_bert_feature 的函数，用于从 BERT 模型中提取文本特征。这个函数需要一个参数 text，一个参数 word2ph，表示从左到右的词素映射，以及一个参数 device，用于指定要使用的设备(例如，CPU 或 GPU)。如果 device 为 CPU，则需要在命令行中指定该设备。

在 main 函数中，还定义了一个名为 bert_feature 的函数，用于从 BERT 模型中提取文本特征。这个函数会尝试使用 device 指定的硬件设备，如果没有可用的硬件设备，则默认使用 CPU 设备。

接着，代码遍历 main 函数中的 get_bert_feature 函数，并将提取到的文本特征存储到 models 字典中。

最后，代码创建了一个名为 model_tokenizer 的函数，用于从 bert 模型中提取词素。


```py
import torch
import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("./bert/chinese-roberta-wwm-ext-large")

models = dict()


def get_bert_feature(text, word2ph, device=None):
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if device not in models.keys():
        models[device] = AutoModelForMaskedLM.from_pretrained(
            "./bert/chinese-roberta-wwm-ext-large"
        ).to(device)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = models[device](**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()

    assert len(word2ph) == len(text) + 2
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T


```

这段代码的主要作用是计算一个文本数据集中的词和对应的拼音。其中，词和拼音分别以数组的形式存储，每个拼音对应一个词。

具体来说，第一行先引入了PyTorch库，然后定义了一个名为"__main__"的函数，这是PyTorch库中的一个特殊函数，表示当脚本运行时，如果__name__不等于 "__main__"，就执行函数中的内容。

接下来，定义了一个包含38个词和1024个词的PyTorch张量，分别命名为"word_level_feature"和"word2phone"。其中，"word_level_feature"是一个4维的张量，每个维度包含1024个单词的1024维特征。而"word2phone"则是一个包含38个单词的列表，每个单词包含1个数字，即每个单词对应一个拼音。

接着，使用for循环计算每个单词的词和对应的拼音，并将它们存储在"phone_level_feature"数组中。其中，在每个循环中，首先使用`word_level_feature`中的该单词的4维张量，然后将它重复这个单词的1次，即得到一个与输入单词相同大小但维度为1的张量。最后将这些张量连接成一个36维的数组，并将其存储为"phone_level_feature"数组的元素。

最终的结果是，"phone_level_feature"数组，其中每个元素都是一个包含36个单词的数组，每个数组都是用1024维的CPU张量计算得到的。


```py
if __name__ == "__main__":
    import torch

    word_level_feature = torch.rand(38, 1024)  # 12个词,每个词1024维特征
    word2phone = [
        1,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        2,
        2,
        1,
    ]

    # 计算总帧数
    total_frames = sum(word2phone)
    print(word_level_feature.shape)
    print(word2phone)
    phone_level_feature = []
    for i in range(len(word2phone)):
        print(word_level_feature[i].shape)

        # 对每个词重复word2phone[i]次
        repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    print(phone_level_feature.shape)  # torch.Size([36, 1024])

```

# `text/cleaner.py`

这段代码定义了一个名为 `clean_text` 的函数，它接受一个字符串参数 `text` 和一个语言参数 `language`，并返回经过清理后的文本、电话号码和文本中的拼音。

clean_text 的实现主要依赖于两个参数：语言模块映射和经过预处理后的文本。通过调用 `chinese` 和 `clean_text_bert` 函数，可以加载不同语言的预处理模块，并使用这些模块对输入的文本进行预处理。

具体来说，代码中的 `clean_text_bert` 函数接受一个经过预处理后的文本和语言参数，使用该语言的预处理模块对文本进行处理，并获取 BERT 模型的编码结果。然后，代码将编码结果和语言对应的拼音作为输出，并使用 `g2p` 函数将拼音映射到具体的目标拼音。

最终，代码中的 `clean_text` 函数在 `clean_text_bert` 的基础上，对传入的文本进行语言无关的预处理，并输出经过清理后的文本、电话号码和拼音。


```py
from text import chinese, japanese, cleaned_text_to_sequence


language_module_map = {"ZH": chinese, "JP": japanese}


def clean_text(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph


def clean_text_bert(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    bert = language_module.get_bert_feature(norm_text, word2ph)
    return phones, tones, bert


```

这段代码定义了一个名为 `text_to_sequence` 的函数，它接受两个参数 `text` 和 `language`，分别表示要转换为序列的文本和目标语言。

函数实现中，首先调用一个名为 `clean_text` 的函数来处理传入的文本，该函数可以清除文本中的标点符号、数字等不符合规范的内容，同时可以分离出文本中的单词和电话号码。

接着，函数调用了另一个名为 `cleaned_text_to_sequence` 的函数，该函数接受两个已经处理过的参数 `phones` 和 `tones`，分别表示电话号码和语气音调，以及目标语言。

最后，函数返回经过处理后的结果，可以将其存储为另一个函数或文件等。

该函数的作用是将传入的文本转化为目标语言的序列数据，其中 `clean_text` 和 `cleaned_text_to_sequence` 函数是实现该功能的关键部分。


```py
def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass

```

# `text/english.py`

It seems like you are trying to load a dictionary that has been indexed with a考研成功预测 algorithm. However, I don't have any information about this specific implementation, so I'm unable to provide any additional guidance.

Can you please provide more context or information about what you are trying to do? This will help me provide more detailed assistance.


```py
import pickle
import os
import re
from g2p_en import G2p

from text import symbols

current_file_path = os.path.dirname(__file__)
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")
_g2p = G2p()

arpa = {
    "AH0",
    "S",
    "AH1",
    "EY2",
    "AE2",
    "EH0",
    "OW2",
    "UH0",
    "NG",
    "B",
    "G",
    "AY0",
    "M",
    "AA0",
    "F",
    "AO0",
    "ER2",
    "UH1",
    "IY1",
    "AH2",
    "DH",
    "IY0",
    "EY1",
    "IH0",
    "K",
    "N",
    "W",
    "IY2",
    "T",
    "AA1",
    "ER1",
    "EH2",
    "OY0",
    "UH2",
    "UW1",
    "Z",
    "AW2",
    "AW1",
    "V",
    "UW2",
    "AA2",
    "ER",
    "AW0",
    "UW0",
    "R",
    "OW1",
    "EH1",
    "ZH",
    "AE0",
    "IH2",
    "IH",
    "Y",
    "JH",
    "P",
    "AY1",
    "EY0",
    "OY2",
    "TH",
    "HH",
    "D",
    "ER0",
    "CH",
    "AO1",
    "AE1",
    "AO2",
    "OY1",
    "AY2",
    "IH1",
    "OW0",
    "L",
    "SH",
}


```

这段代码定义了一个名为 `post_replace_ph` 的函数，用于将给定的拼音字符串(ph)进行替换操作，其作用如下：

1. 如果给定的拼音字符串(ph)在字典 `rep_map` 中存在，则将该字符串替换为该字典中的键值对，即 `rep_map[ph]`。
2. 如果给定的拼音字符串(ph)在字符串常量列表(symbols)中存在，则返回该字符串，否则将字符串替换为“UNK”(未定义)。

函数的实现比较简单，主要使用了 Python 内置的字符串方法和字典数据结构。通过遍历字典 `rep_map`，查找给定的拼音字符串(ph)是否存在于字典中，如果存在则进行替换操作，否则返回“UNK”。


```py
def post_replace_ph(ph):
    rep_map = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "...": "…",
        "v": "V",
    }
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    if ph not in symbols:
        ph = "UNK"
    return ph


```

这段代码的作用是读取一个名为 CMU_DICT_PATH 的文件，里面的每一行包含一个单词，每个单词由一个音节和若干个辅音音节组成。代码读取每个单词，将其拆分成音节和电话号码两部分，并将电话号码加入对应单词的列表中，最终返回一个字典类型的 g2p_dict 变量。


```py
def read_dict():
    g2p_dict = {}
    start_line = 49
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= start_line:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0]

                syllable_split = word_split[1].split(" - ")
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    g2p_dict[word].append(phone_split)

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


```

这段代码定义了一个名为 `cache_dict` 的函数和一个名为 `get_dict` 的函数。

`cache_dict` 函数的作用是将一个名为 `g2p_dict` 的字典存储到一个名为 `file_path` 的文件中，存储格式为二进制写入(wb)。具体来说，它使用 `with` 语句打开一个写二进制文件的设备(比如一个文件)，然后使用 `pickle.dump` 函数将 `g2p_dict` 对象写入到文件中。

`get_dict` 函数的作用是读取一个名为 `CACHE_PATH` 的缓存文件中的内容，如果缓存文件存在，就从文件中读取，否则就从 `read_dict` 函数中读取，并将读取到的字典存储到 `g2p_dict` 中，最后将更新后的 `g2p_dict` 写回到缓存文件中。它的实现方式是先检查缓存文件是否存在，如果存在，就使用 `with` 语句打开一个读二进制文件的设备，并使用 `pickle.load` 函数从文件中读取字典，否则就调用 `read_dict` 函数读取字典，并将更新后的字典存储到 `g2p_dict` 中，最后将更新后的 `g2p_dict` 写回到缓存文件中。


```py
def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


def get_dict():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict()
        cache_dict(g2p_dict, CACHE_PATH)

    return g2p_dict


```

这段代码的作用是定义了两个函数refine_ph和refine_syllables，以及一个名为eng_dict的字典。

refine_ph函数用于将一个音节(phn)精炼成一个新的音节(phn_tone)，并且只基于这个音节的最后一位(如果它是数字的话)，如果该音节不是数字，则将其转换成一个小写字母并增加1。

refine_syllables函数用于将多个音节(syllables)精炼成一个新的列表(phonemes)，每个音节都对应一个音素(phoneme)，并记录下其浊度和强度(tone and strength)，这些浊度和强度基于上面精炼出来的音节。

eng_dict是一个字典，用于存储经过refine_ph和refine_syllables函数处理过的单词及其对应的音节和浊度。


```py
eng_dict = get_dict()


def refine_ph(phn):
    tone = 0
    if re.search(r"\d$", phn):
        tone = int(phn[-1]) + 1
        phn = phn[:-1]
    return phn.lower(), tone


def refine_syllables(syllables):
    tones = []
    phonemes = []
    for phn_list in syllables:
        for i in range(len(phn_list)):
            phn = phn_list[i]
            phn, tone = refine_ph(phn)
            phonemes.append(phn)
            tones.append(tone)
    return phonemes, tones


```

这段代码定义了两个函数，text_normalize() 和 g2p()。

text_normalize()函数的作用是返回一个经过正常化的文本。函数内部暂时没有做具体的工作，可以直接返回。

g2p()函数的作用是将一个文本转换为拼音列表。函数接收一个文本参数，首先通过re.split()函数将文本分割为单词，然后通过refine_syllables()函数对单词进行音节划分。接着，对每个单词，使用refine_ph()函数将其转换为拼音。最终，返回经过处理后的单词和拼音列表。


```py
def text_normalize(text):
    # todo: eng text normalize
    return text


def g2p(text):
    phones = []
    tones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.upper() in eng_dict:
            phns, tns = refine_syllables(eng_dict[w.upper()])
            phones += phns
            tones += tns
        else:
            phone_list = list(filter(lambda p: p != " ", _g2p(w)))
            for ph in phone_list:
                if ph in arpa:
                    ph, tn = refine_ph(ph)
                    phones.append(ph)
                    tones.append(tn)
                else:
                    phones.append(ph)
                    tones.append(0)
    # todo: implement word2ph
    word2ph = [1 for i in phones]

    phones = [post_replace_ph(i) for i in phones]
    return phones, tones, word2ph


```

这段代码是一个Python脚本，主要作用是输出一个名为"English words to phonemes"的英文单词表。

具体来说，代码会执行以下操作：

1. 读取一个名为"eng_dict.py"的Python文件，并导入其中定义的英文单词表。
2. 输出GAN-based universal vocoder的名称。
3. 输出GAN-based universal vocoder的英文单词。
4. 将所有英文单词添加到一个名为"all_phones"的集合中。
5. 通过循环遍历英文单词表中的所有单词，并将其拆分成音节。
6. 对于每个音节，将其添加到"all_phones"中。
7. 输出"all_phones"中的所有元素。


```py
if __name__ == "__main__":
    # print(get_dict())
    # print(eng_word_to_phoneme("hello"))
    print(g2p("In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder."))
    # all_phones = set()
    # for k, syllables in eng_dict.items():
    #     for group in syllables:
    #         for ph in group:
    #             all_phones.add(ph)
    # print(all_phones)

```

# `text/english_bert_mock.py`

这段代码使用了PyTorch库来实现，其主要目的是获取BERT模型的特征。BERT是一种预训练的自然语言处理模型，其可以从文本中提取出丰富的特征信息。这里，我们定义了一个名为`get_bert_feature`的函数，该函数接受两个参数：`norm_text`和`word2ph`。其中，`norm_text`表示经过标准化处理后的文本，`word2ph`表示一个词到预训练模型的拼音映射。

函数实现中，我们首先从`torch.zeros`创建一个大小为`1024`的二维内存，用于存储BERT模型的特征。接着，我们对`norm_text`和`word2ph`分别进行一些处理，这里我们省略了具体的实现。最后，我们得到了一个大小为`sum(word2ph)`的二维内存，其中每个元素的值都为`1`，这意味着这些内存区域都是填充完毕的。

总的来说，这段代码的主要目的是创建一个用于存储BERT模型特征的内存区域，该区域的大小为预训练模型的总词数。


```py
import torch


def get_bert_feature(norm_text, word2ph):
    return torch.zeros(1024, sum(word2ph))

```

# `text/japanese.py`

这段代码的作用是将日语文本转换为听觉表示，使其与Julius兼容。具体来说，它实现了以下功能：

1. 导入必要的库：re、unicodedata、AutoTokenizer、MeCab和num2words。
2. 定义了一个名为"text"的函数，该函数接受一个字符串参数。
3. 在"text"函数中，首先尝试导入MeCab库，如果成功，则跳过错误。否则， raise错误并返回。
4. 导入unidicat库的from num2words import num2words。
5. 使用num2words库将文本中的数字转换为日本假名。
6. 使用re库的pattern功能从字符串中删除标点符号。
7. 将删除标点符号后的文本和日本假名合并，并将结果存储在"text"函数中。
8. 最后，使用AutoTokenizer库的add_special_tokens函数对结果进行特殊标记。


```py
# Convert Japanese text to phonemes which is
# compatible with Julius https://github.com/julius-speech/segmentation-kit
import re
import unicodedata

from transformers import AutoTokenizer

from text import punctuation, symbols

try:
    import MeCab
except ImportError as e:
    raise ImportError("Japanese requires mecab-python3 and unidic-lite.") from e
from num2words import num2words

```

It appears that you're Japanese characters, but with some characters in the text可能有误。 If you have any specific question or request, please let me know and I'll do my best to assist you.



```py
_CONVRULES = [
    # Conversion of 2 letters
    "アァ/ a a",
    "イィ/ i i",
    "イェ/ i e",
    "イャ/ y a",
    "ウゥ/ u:",
    "エェ/ e e",
    "オォ/ o:",
    "カァ/ k a:",
    "キィ/ k i:",
    "クゥ/ k u:",
    "クャ/ ky a",
    "クュ/ ky u",
    "クョ/ ky o",
    "ケェ/ k e:",
    "コォ/ k o:",
    "ガァ/ g a:",
    "ギィ/ g i:",
    "グゥ/ g u:",
    "グャ/ gy a",
    "グュ/ gy u",
    "グョ/ gy o",
    "ゲェ/ g e:",
    "ゴォ/ g o:",
    "サァ/ s a:",
    "シィ/ sh i:",
    "スゥ/ s u:",
    "スャ/ sh a",
    "スュ/ sh u",
    "スョ/ sh o",
    "セェ/ s e:",
    "ソォ/ s o:",
    "ザァ/ z a:",
    "ジィ/ j i:",
    "ズゥ/ z u:",
    "ズャ/ zy a",
    "ズュ/ zy u",
    "ズョ/ zy o",
    "ゼェ/ z e:",
    "ゾォ/ z o:",
    "タァ/ t a:",
    "チィ/ ch i:",
    "ツァ/ ts a",
    "ツィ/ ts i",
    "ツゥ/ ts u:",
    "ツャ/ ch a",
    "ツュ/ ch u",
    "ツョ/ ch o",
    "ツェ/ ts e",
    "ツォ/ ts o",
    "テェ/ t e:",
    "トォ/ t o:",
    "ダァ/ d a:",
    "ヂィ/ j i:",
    "ヅゥ/ d u:",
    "ヅャ/ zy a",
    "ヅュ/ zy u",
    "ヅョ/ zy o",
    "デェ/ d e:",
    "ドォ/ d o:",
    "ナァ/ n a:",
    "ニィ/ n i:",
    "ヌゥ/ n u:",
    "ヌャ/ ny a",
    "ヌュ/ ny u",
    "ヌョ/ ny o",
    "ネェ/ n e:",
    "ノォ/ n o:",
    "ハァ/ h a:",
    "ヒィ/ h i:",
    "フゥ/ f u:",
    "フャ/ hy a",
    "フュ/ hy u",
    "フョ/ hy o",
    "ヘェ/ h e:",
    "ホォ/ h o:",
    "バァ/ b a:",
    "ビィ/ b i:",
    "ブゥ/ b u:",
    "フャ/ hy a",
    "ブュ/ by u",
    "フョ/ hy o",
    "ベェ/ b e:",
    "ボォ/ b o:",
    "パァ/ p a:",
    "ピィ/ p i:",
    "プゥ/ p u:",
    "プャ/ py a",
    "プュ/ py u",
    "プョ/ py o",
    "ペェ/ p e:",
    "ポォ/ p o:",
    "マァ/ m a:",
    "ミィ/ m i:",
    "ムゥ/ m u:",
    "ムャ/ my a",
    "ムュ/ my u",
    "ムョ/ my o",
    "メェ/ m e:",
    "モォ/ m o:",
    "ヤァ/ y a:",
    "ユゥ/ y u:",
    "ユャ/ y a:",
    "ユュ/ y u:",
    "ユョ/ y o:",
    "ヨォ/ y o:",
    "ラァ/ r a:",
    "リィ/ r i:",
    "ルゥ/ r u:",
    "ルャ/ ry a",
    "ルュ/ ry u",
    "ルョ/ ry o",
    "レェ/ r e:",
    "ロォ/ r o:",
    "ワァ/ w a:",
    "ヲォ/ o:",
    "ディ/ d i",
    "デェ/ d e:",
    "デャ/ dy a",
    "デュ/ dy u",
    "デョ/ dy o",
    "ティ/ t i",
    "テェ/ t e:",
    "テャ/ ty a",
    "テュ/ ty u",
    "テョ/ ty o",
    "スィ/ s i",
    "ズァ/ z u a",
    "ズィ/ z i",
    "ズゥ/ z u",
    "ズャ/ zy a",
    "ズュ/ zy u",
    "ズョ/ zy o",
    "ズェ/ z e",
    "ズォ/ z o",
    "キャ/ ky a",
    "キュ/ ky u",
    "キョ/ ky o",
    "シャ/ sh a",
    "シュ/ sh u",
    "シェ/ sh e",
    "ショ/ sh o",
    "チャ/ ch a",
    "チュ/ ch u",
    "チェ/ ch e",
    "チョ/ ch o",
    "トゥ/ t u",
    "トャ/ ty a",
    "トュ/ ty u",
    "トョ/ ty o",
    "ドァ/ d o a",
    "ドゥ/ d u",
    "ドャ/ dy a",
    "ドュ/ dy u",
    "ドョ/ dy o",
    "ドォ/ d o:",
    "ニャ/ ny a",
    "ニュ/ ny u",
    "ニョ/ ny o",
    "ヒャ/ hy a",
    "ヒュ/ hy u",
    "ヒョ/ hy o",
    "ミャ/ my a",
    "ミュ/ my u",
    "ミョ/ my o",
    "リャ/ ry a",
    "リュ/ ry u",
    "リョ/ ry o",
    "ギャ/ gy a",
    "ギュ/ gy u",
    "ギョ/ gy o",
    "ヂェ/ j e",
    "ヂャ/ j a",
    "ヂュ/ j u",
    "ヂョ/ j o",
    "ジェ/ j e",
    "ジャ/ j a",
    "ジュ/ j u",
    "ジョ/ j o",
    "ビャ/ by a",
    "ビュ/ by u",
    "ビョ/ by o",
    "ピャ/ py a",
    "ピュ/ py u",
    "ピョ/ py o",
    "ウァ/ u a",
    "ウィ/ w i",
    "ウェ/ w e",
    "ウォ/ w o",
    "ファ/ f a",
    "フィ/ f i",
    "フゥ/ f u",
    "フャ/ hy a",
    "フュ/ hy u",
    "フョ/ hy o",
    "フェ/ f e",
    "フォ/ f o",
    "ヴァ/ b a",
    "ヴィ/ b i",
    "ヴェ/ b e",
    "ヴォ/ b o",
    "ヴュ/ by u",
    # Conversion of 1 letter
    "ア/ a",
    "イ/ i",
    "ウ/ u",
    "エ/ e",
    "オ/ o",
    "カ/ k a",
    "キ/ k i",
    "ク/ k u",
    "ケ/ k e",
    "コ/ k o",
    "サ/ s a",
    "シ/ sh i",
    "ス/ s u",
    "セ/ s e",
    "ソ/ s o",
    "タ/ t a",
    "チ/ ch i",
    "ツ/ ts u",
    "テ/ t e",
    "ト/ t o",
    "ナ/ n a",
    "ニ/ n i",
    "ヌ/ n u",
    "ネ/ n e",
    "ノ/ n o",
    "ハ/ h a",
    "ヒ/ h i",
    "フ/ f u",
    "ヘ/ h e",
    "ホ/ h o",
    "マ/ m a",
    "ミ/ m i",
    "ム/ m u",
    "メ/ m e",
    "モ/ m o",
    "ラ/ r a",
    "リ/ r i",
    "ル/ r u",
    "レ/ r e",
    "ロ/ r o",
    "ガ/ g a",
    "ギ/ g i",
    "グ/ g u",
    "ゲ/ g e",
    "ゴ/ g o",
    "ザ/ z a",
    "ジ/ j i",
    "ズ/ z u",
    "ゼ/ z e",
    "ゾ/ z o",
    "ダ/ d a",
    "ヂ/ j i",
    "ヅ/ z u",
    "デ/ d e",
    "ド/ d o",
    "バ/ b a",
    "ビ/ b i",
    "ブ/ b u",
    "ベ/ b e",
    "ボ/ b o",
    "パ/ p a",
    "ピ/ p i",
    "プ/ p u",
    "ペ/ p e",
    "ポ/ p o",
    "ヤ/ y a",
    "ユ/ y u",
    "ヨ/ y o",
    "ワ/ w a",
    "ヰ/ i",
    "ヱ/ e",
    "ヲ/ o",
    "ン/ N",
    "ッ/ q",
    "ヴ/ b u",
    "ー/:",
    # Try converting broken text
    "ァ/ a",
    "ィ/ i",
    "ゥ/ u",
    "ェ/ e",
    "ォ/ o",
    "ヮ/ w a",
    "ォ/ o",
    # Symbols
    "、/ ,",
    "。/ .",
    "！/ !",
    "？/ ?",
    "・/ ,",
]

```

这段代码的作用是定义了两个正则表达式（regex）模式，然后实现了将日本假名（katakana）文本转换为假名音节的功能。

首先，定义了两个名为`_RULEMAP1`和`_RULEMAP2`的函数，它们都使用正则表达式模式匹配 rules 列表中的两个元素。通过遍历 rules 列表，创建了一个映射，将规则Key（字段名）映射到规则Value（匹配模式）。这个映射关系的键和值都被制作成元组形式，然后使用了两个列表（子列表）来分别处理规则。

接下来，定义了一个名为`kata2phoneme`的函数，这个函数接受一个日本假名文本作为参数，返回一个包含假名音节的新文本。函数内部首先将输入文本的边转型为字符串，然后使用正则表达式模式 `_RULEMAP2` 来查找两个及以上的匹配项。如果找到了匹配项，则从匹配结果中提取出从第二个匹配项开始的子串，然后继续寻找下一个匹配项。如果匹配项数目小于2，则直接返回输入文本的第一个匹配项。

接下来，使用了 `re.compile` 函数，将上述两个正则表达式模式编译成正则表达式对象，以便在后续的文本处理过程中使用。最后，在 `kata2phoneme` 函数中，通过调用 `re.sub` 函数，将匹配日本假名文本的第一个匹配项替换为相应的假名音节，并将结果返回。


```py
_COLON_RX = re.compile(":+")
_REJECT_RX = re.compile("[^ a-zA-Z:,.?]")


def _makerulemap():
    l = [tuple(x.split("/")) for x in _CONVRULES]
    return tuple({k: v for k, v in l if len(k) == i} for i in (1, 2))


_RULEMAP1, _RULEMAP2 = _makerulemap()


def kata2phoneme(text: str) -> str:
    """Convert katakana text to phonemes."""
    text = text.strip()
    res = []
    while text:
        if len(text) >= 2:
            x = _RULEMAP2.get(text[:2])
            if x is not None:
                text = text[2:]
                res += x.split(" ")[1:]
                continue
        x = _RULEMAP1.get(text[0])
        if x is not None:
            text = text[1:]
            res += x.split(" ")[1:]
            continue
        res.append(text[0])
        text = text[1:]
    # res = _COLON_RX.sub(":", res)
    return res


```

这段代码的作用是定义了一个名为`hira2kata`的函数，接受一个字符串参数`text`，并返回一个新的字符串。函数实现的过程如下：

1. 首先定义了两个变量`_KATAKANA`和`_HIRAGANA`，这两个变量是通过创建一个字符串列表，并使用`chr(ord("的象征")),ord(" final"))`函数得到的。其中`chr(ord("的象征"))`创建了一个包含`ord("的象征")`到`ord("final")`的字符集，`list()`函数将这个字符串列表赋值给`_KATAKANA`，`join()`函数将列表中的所有字符连接成一个字符串，最终得到一个包含从`ord("symbol")`到`ord("final")`的字符的字符串列表。而`_HIRAGANA`则是相反的过程，通过`chr(ord("symbol")),list()`函数得到一个包含从`ord("symbol")`到`ord("final")`的字符集，`join()`函数将这个字符串列表赋值给`_HIRAGANA`，最终得到一个包含从`ord("symbol")`到`ord("final")`的字符的字符串列表。

2. 接着定义了一个名为`_HIRA2KATATRANS`的变量，这个变量是通过调用`str.maketrans()`函数得到的。这个函数的作用是将`_HIRAGANA`字符串列表中的所有字符，替换为`_KATAKANA`字符串列表中的相应字符。

3. 定义了一个名为`hira2kata()`的函数，这个函数接收一个字符串参数`text`，并使用`text.translate(_HIRA2KATATRANS)`方法将`text`字符串中的所有字符，从`_HIRA2KATATRANS`字符串列表中替换为相应的从`_KATAKANA`字符串列表中得来的字符。

4. 定义了一个名为`_SYMBOL_TOKENS`和`_NO_YOMI_TOKENS`的集合变量，这两个变量是通过在列表中查找特定的字符而定义的。`_SYMBOL_TOKENS`是一个包含从`chr(ord("symbol"))`到`chr(ord("final"))`的字符集合，而`_NO_YOMI_TOKENS`则是一个包含从`chr(ord(" NO_"))`到`chr(ord(" PI"))`的字符集合。

5. 定义了一个名为`_TAGGER`的类变量，这个类实现了一个`MeCab.Tagger`接口，这个接口定义了一个将给定的字符串转换为另一个字符串的函数。


```py
_KATAKANA = "".join(chr(ch) for ch in range(ord("ァ"), ord("ン") + 1))
_HIRAGANA = "".join(chr(ch) for ch in range(ord("ぁ"), ord("ん") + 1))
_HIRA2KATATRANS = str.maketrans(_HIRAGANA, _KATAKANA)


def hira2kata(text: str) -> str:
    text = text.translate(_HIRA2KATATRANS)
    return text.replace("う゛", "ヴ")


_SYMBOL_TOKENS = set(list("・、。？！"))
_NO_YOMI_TOKENS = set(list("「」『』―（）［］[]"))
_TAGGER = MeCab.Tagger()


```

这段代码定义了一个名为 `text2kata` 的函数，它接受一个字符串参数 `text`。函数的作用是将输入的字符串 `text` 转换成日本假名。

函数内部首先调用一个名为 `_TAGGER.parse` 的函数，它接受一个字符串参数，表示输入的文字。这个函数的作用是将输入的文字转换成文句赌aches，也就是日本假名。

接下来，函数内部循环遍历 `parsed.split("\n")` 得到的所有文句赌花。对于每个文句赌花，函数先判断其是否为 EOS（end of string）标点。如果是，函数跳出循环。否则，函数将花分解成辞典（辞典是一种树形数据结构，其中每个键都是一个字符）并返回其中所有可以找到的词。这些词如果是特殊假名（如 "获得感"、"共产党" 中的 "acity"、"出血" 等），函数将特殊标记并返回。如果词是有效的日本假名，函数将其添加到结果列表中。如果无论怎么解析都不行，函数会将词作为孤儿添加到结果列表中。

最后，函数接受一个参数 `hira2kata`，它接受一个字符串参数，表示一个已经解析好的日本假名。函数将接收来的日本假名中的所有可以找到的词添加到结果列表中，并返回结果列表的唯一字符串。


```py
def text2kata(text: str) -> str:
    parsed = _TAGGER.parse(text)
    res = []
    for line in parsed.split("\n"):
        if line == "EOS":
            break
        parts = line.split("\t")

        word, yomi = parts[0], parts[1]
        if yomi:
            res.append(yomi)
        else:
            if word in _SYMBOL_TOKENS:
                res.append(word)
            elif word in ("っ", "ッ"):
                res.append("ッ")
            elif word in _NO_YOMI_TOKENS:
                pass
            else:
                res.append(word)
    return hira2kata("".join(res))


```

This is a JavaScript object that maps日语 Hiragana letters to their English equivalents. It includes the letter pairs for each of the 108 characters in the Japanese kana system.

Here is the object:

```py 
const kanjiToEnglish = {
 "@": "アット",
 "a": "エー",
 "b": "ビー",
 "c": "シー",
 "d": "ディー",
 "e": "イー",
 "f": "アイファー",
 "g": "ジーズス",
 "h": "不负 flow through letter flow",
 "i": "hhé个 guy",
 "j": " Jeff",
 "k": "电子商务",
 "l": "刑的一种，女人",
 "m": "监，肌肉，minimal effort movement",
 "n": "ながす",
 "o": "哦",
 "p": "悲剧",
 "q": "奇妙的",
 "r": "科学研究， Round trip",
 "s": "すすます",
 "t": "超意，Top level yield",
 "u": "公共交通，Union Transport",
 "v": "北，video,viewer",
 "w": "抽象",
 "x": "徐々な,from Alice to Unreal Engine",
 "y": "游行车，yōuka error",
 "z": "生命的意义， end up in front of a wall",
 "α": "究极，α-发出",
 "β": "力，the power of being simple"
};
```

样本使用

你可以使用这个对象检查一个日语单词的英语翻译，以下是一个示例：

```py 
const kanjiToEnglish = {
 "@": "アット",
 "a": "抽烟",
 "b": "只猴子",
 "c": "打开",
 "d": " gate "
};

const kanji = "邀，在日本，在齐，和中国";

console.log(kanjiToEnglish[ kanji ]); // "open"
```

完整的 API 定义：

```py 
const kanjiToEnglish = {
 @: "αλφβγδε faint return",
 綱： "system for returning蛾磷methyl取代的吧！"
};

const kanji = "邀，在日本，在齐，和中国";

console.log(kanjiToEnglish[ kanji ]); // "open"
```

这只是其中的一个例子，你可以在需要时使用这个对象。


```py
_ALPHASYMBOL_YOMI = {
    "#": "シャープ",
    "%": "パーセント",
    "&": "アンド",
    "+": "プラス",
    "-": "マイナス",
    ":": "コロン",
    ";": "セミコロン",
    "<": "小なり",
    "=": "イコール",
    ">": "大なり",
    "@": "アット",
    "a": "エー",
    "b": "ビー",
    "c": "シー",
    "d": "ディー",
    "e": "イー",
    "f": "エフ",
    "g": "ジー",
    "h": "エイチ",
    "i": "アイ",
    "j": "ジェー",
    "k": "ケー",
    "l": "エル",
    "m": "エム",
    "n": "エヌ",
    "o": "オー",
    "p": "ピー",
    "q": "キュー",
    "r": "アール",
    "s": "エス",
    "t": "ティー",
    "u": "ユー",
    "v": "ブイ",
    "w": "ダブリュー",
    "x": "エックス",
    "y": "ワイ",
    "z": "ゼット",
    "α": "アルファ",
    "β": "ベータ",
    "γ": "ガンマ",
    "δ": "デルタ",
    "ε": "イプシロン",
    "ζ": "ゼータ",
    "η": "イータ",
    "θ": "シータ",
    "ι": "イオタ",
    "κ": "カッパ",
    "λ": "ラムダ",
    "μ": "ミュー",
    "ν": "ニュー",
    "ξ": "クサイ",
    "ο": "オミクロン",
    "π": "パイ",
    "ρ": "ロー",
    "σ": "シグマ",
    "τ": "タウ",
    "υ": "ウプシロン",
    "φ": "ファイ",
    "χ": "カイ",
    "ψ": "プサイ",
    "ω": "オメガ",
}


```

这段代码的作用是定义了两个函数，分别为 `japanese_convert_numbers_to_words` 和 `japanese_convert_alpha_symbols_to_words`。

第一个函数 `japanese_convert_numbers_to_words` 接收一个字符串参数 `text`，并使用正则表达式 `re.compile("[0-9]{1,3}(,[0-9]{3})+")` 将数字匹配提取出来，然后通过字符映射替换掉数字中多余的零，再通过 `re.compile(r"([$¥£€])([0-9.]*[0-9])")` 将货币名称匹配提取出来，最后通过 `re.compile(r"[0-9]+(\.[0-9]+)?")` 将匹配的数字中允许带小数点。最终的结果将会被返回。

第二个函数 `japanese_convert_alpha_symbols_to_words` 接收一个字符串参数 `text`，并使用正则表达式 `re.compile(r"([^}]+)")` 将所有非空铝字符符号替换成空格，最后的结果将会被返回。


```py
_NUMBER_WITH_SEPARATOR_RX = re.compile("[0-9]{1,3}(,[0-9]{3})+")
_CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}
_CURRENCY_RX = re.compile(r"([$¥£€])([0-9.]*[0-9])")
_NUMBER_RX = re.compile(r"[0-9]+(\.[0-9]+)?")


def japanese_convert_numbers_to_words(text: str) -> str:
    res = _NUMBER_WITH_SEPARATOR_RX.sub(lambda m: m[0].replace(",", ""), text)
    res = _CURRENCY_RX.sub(lambda m: m[2] + _CURRENCY_MAP.get(m[1], m[1]), res)
    res = _NUMBER_RX.sub(lambda m: num2words(m[0], lang="ja"), res)
    return res


def japanese_convert_alpha_symbols_to_words(text: str) -> str:
    return "".join([_ALPHASYMBOL_YOMI.get(ch, ch) for ch in text.lower()])


```



该函数将输入的 Japanese 文本转换为 phonemes(音节)，即日语拼音。函数的实现包括两个步骤：

1. 将文本转换为 Unicode 编码格式，并且只处理 Unicode 范围内编码的文本。
2. 将文本中的字符转换为浮点数，然后将其转换为日语假名。

函数的实现基于两个辅助函数：

1. `is_japanese_character()`函数：该函数接收一个字符作为输入，并返回一个布尔值。该函数定义了日语文字系统的 Unicode 范围，然后将其与给定的字符进行比较。如果字符在任何一个范围内，则返回 True，否则返回 False。

2. `text2kata()`函数：该函数接收一个 Unicode 编码的文本，并返回一个列表，其中包含日语假名。该函数根据一些辅助字符表来将文本转换为假名。

3. `kata2phoneme()`函数：该函数接收一个 Unicode 编码的文本，并返回一个列表，其中包含每个假名对应的音节。该函数根据一些辅助字符表和电话语音学规则将文本转换为假名，并尝试将其转换为电话语音。

最终的结果是将输入的文本转换为日语假名列表。


```py
def japanese_text_to_phonemes(text: str) -> str:
    """Convert Japanese text to phonemes."""
    res = unicodedata.normalize("NFKC", text)
    res = japanese_convert_numbers_to_words(res)
    # res = japanese_convert_alpha_symbols_to_words(res)
    res = text2kata(res)
    res = kata2phoneme(res)
    return res


def is_japanese_character(char):
    # 定义日语文字系统的 Unicode 范围
    japanese_ranges = [
        (0x3040, 0x309F),  # 平假名
        (0x30A0, 0x30FF),  # 片假名
        (0x4E00, 0x9FFF),  # 汉字 (CJK Unified Ideographs)
        (0x3400, 0x4DBF),  # 汉字扩展 A
        (0x20000, 0x2A6DF),  # 汉字扩展 B
        # 可以根据需要添加其他汉字扩展范围
    ]

    # 将字符的 Unicode 编码转换为整数
    char_code = ord(char)

    # 检查字符是否在任何一个日语范围内
    for start, end in japanese_ranges:
        if start <= char_code <= end:
            return True

    return False


```

这段代码是一个Python代码，定义了一个名为rep_map的字典，包含了20个用“，”作为键的键值对，分别表示为中文字符。接下来定义了一个名为replace_punctuation的函数，该函数接收一个中文字符串作为参数，将其中的标点符号（包括中文和英文的标点）全部转义，然后将其余下的字符串连接起来，再将转义后的字符串转换为小写，最后将其返回。函数的核心实现部分是正则表达式模式替换以及字符串拼接，通过组合正则表达式模式和字符串拼接，实现了对中文字符串进行标点转义、转义后字符串拼接以及中文标点符号转换成英文标点等操作。最终返回转义并转小写字母后的字符串。


```py
rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
}


def replace_punctuation(text):
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(
        r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF"
        + "".join(punctuation)
        + r"]+",
        "",
        replaced_text,
    )

    return replaced_text


```



This code defines two functions, `text_normalize` and `distribute_phone`.

The `text_normalize` function takes a text string as input and returns a normalized version of that text, which is encoded in the NFKC (Native Japanese Fonts Compatibility) encoding. It then converts the text to Japanese characters and replaces any Japanese characters with their English equivalent，删去所有标点符号，最后将结果拼接回字符串。

The `distribute_phone` function takes two parameters, `n_phone` and `n_word`, and returns a list of `n_phone` numbers corresponding to the number of times each word is emphasized in the text, it normalizes the phone distribution of the text, it is not very clear what it does exactly.


```py
def text_normalize(text):
    res = unicodedata.normalize("NFKC", text)
    res = japanese_convert_numbers_to_words(res)
    # res = "".join([i for i in res if is_japanese_character(i)])
    res = replace_punctuation(res)
    return res


def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


```

这段代码是一个用于将文本数据中的日语单词转换为对应语音序列的函数。具体来说，它使用预训练的 BERT 模型中的日本语数据集，将传入的文本数据进行 tokenize 操作，即将文本数据中的每个单词转换为一个数字编码。然后，它将这个数字编码序列映射到预定义的日语假名（phonemes）上，从而得到该文本数据对应的日本语语音序列。

下面是代码的更详细的解释：

1. `AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")`:

这一行代码使用预训练的 BERT模型，并从 `./bert/bert-base-japanese-v3` 这个文件中加载预训练的模型。这个模型是用于日本语文本分类任务的，具有比较长的参数量，因此在进行具体文本分析时需要进行剪枝，使用的是 `AutoTokenizer` 类来简化操作。

2. `g2p(norm_text)`:

这一行代码定义了一个函数 `g2p`，用于将传入的文本数据中的日本语单词转换为对应的假名编码。

3. `tokenizer.tokenize(norm_text)`:

这一行代码使用 `tokenizer` 对传入的 `norm_text` 参数进行 tokenize 操作，即将文本数据中的每个单词转换为一个数字编码。这里使用的 `tokenize` 方法是 `AutoTokenizer` 类中的一个静态方法，用于从模型中获取预定义的日语词汇，并根据需要对词汇进行清洗和分词操作。

4. `phs = []`、`ph_groups = []`、`word2ph = []`、`phons = []`、`tones = []`、`word2ph = []`:

这几行代码定义了几个变量，用于对文本数据中的日语单词进行分组和处理。这里，`phs` 是一个空列表，用于存储所有分组后的单词列表；`ph_groups` 是一个空列表，用于存储所有分组后的假名列表；`word2ph` 是一个空列表，用于存储所有分组后的数字编码列表；`phons` 是一个空列表，用于存储所有分组后的声调列表；`tones` 是一个空列表，用于存储所有分组后的音节列表；`word2ph` 是一个空列表，用于存储所有分组后的数字编码列表。

5. `distribute_phone(phone_len, word_len)`:

这一行代码定义了一个函数 `distribute_phone`，用于对给定的文本数据中的语音信号进行分布，即把文本数据中的每个单词的编码向量分配给符号。这个函数的输入参数是语音信号的长度 `phone_len` 和文本数据的长度 `word_len`，它根据这两个参数对编码向量进行分


```py
tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")


def g2p(norm_text):
    tokenized = tokenizer.tokenize(norm_text)
    phs = []
    ph_groups = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))
    word2ph = []
    for group in ph_groups:
        phonemes = kata2phoneme(text2kata("".join(group)))
        # phonemes = [i for i in phonemes if i in symbols]
        for i in phonemes:
            assert i in symbols, (group, norm_text, tokenized)
        phone_len = len(phonemes)
        word_len = len(group)

        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa

        phs += phonemes
    phones = ["_"] + phs + ["_"]
    tones = [0 for i in phones]
    word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph


```

这段代码的作用是使用BERT模型对一段日本语文本进行分析和 tokenize，然后输出以下信息：

1. 定义了一个if语句，判断当前是否为主要的程序文件。
2. 从./bert/bert-base-japanese-v3目录中加载了预训练的BERT模型。
3. 将文本数据进行normalization处理，使得可以输入到模型中。
4. 通过g2p函数将文本转换成了日本语的拼音数据，包括电话号码和音节。
5. 通过get_bert_feature函数从加载的BERT模型中获取了文本的特征。
6. 打印了日本语的拼音数据、单词映射以及BERT模型的输出形状。


```py
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")
    text = "hello,こんにちは、世界！……"
    from text.japanese_bert import get_bert_feature

    text = text_normalize(text)
    print(text)
    phones, tones, word2ph = g2p(text)
    bert = get_bert_feature(text, word2ph)

    print(phones, tones, word2ph, bert.shape)

```