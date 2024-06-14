# `.\demo\gradiodemo.py`

```
import torch  # 导入PyTorch库
from scipy.spatial.distance import cosine  # 从SciPy库中导入余弦距离计算函数
from transformers import AutoModel, AutoTokenizer  # 从transformers库中导入AutoModel和AutoTokenizer
import gradio as gr  # 导入gradio库，用于构建交互界面

# Import our models. The package will take care of downloading the models automatically
# 使用AutoTokenizer从预训练模型中加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
# 使用AutoModel从预训练模型中加载模型
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

def simcse(text1, text2, text3):
    # Tokenize input texts
    # 将输入文本进行分词
    texts = [
        text1,
        text2,
        text3
    ]
    # 使用tokenizer对文本进行编码，padding=True表示填充到最长的文本长度，truncation=True表示截断超过最大长度的文本
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Get the embeddings
    # 获取文本的嵌入表示
    with torch.no_grad():  # 关闭梯度计算，因为这里不需要进行模型参数更新
        # 通过模型获取隐藏状态，并从中获取池化后的输出
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    # Calculate cosine similarities
    # 计算余弦相似度
    # 余弦相似度的取值范围为[-1, 1]，数值越高表示文本越相似
    cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])  # 计算第一对文本的余弦相似度
    cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])  # 计算第二对文本的余弦相似度
    return {"cosine similarity":cosine_sim_0_1}, {"cosine similarity":cosine_sim_0_2}  # 返回余弦相似度作为字典

inputs = [
    gr.inputs.Textbox(lines=5, label="Input Text One"),  # 文本输入框，用于输入第一个文本
    gr.inputs.Textbox(lines=5, label="Input Text Two"),  # 文本输入框，用于输入第二个文本
    gr.inputs.Textbox(lines=5, label="Input Text Three")  # 文本输入框，用于输入第三个文本
]

outputs = [
    gr.outputs.Label(type="confidences",label="Cosine similarity between text one and two"),  # 输出标签，显示第一对文本的余弦相似度
    gr.outputs.Label(type="confidences", label="Cosine similarity between text one and three")  # 输出标签，显示第二对文本的余弦相似度
]

title = "SimCSE"  # 界面标题
description = "demo for Princeton-NLP SimCSE. To use it, simply add your text, or click one of the examples to load them. Read more at the links below."  # 界面描述
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2104.08821'>SimCSE: Simple Contrastive Learning of Sentence Embeddings</a> | <a href='https://github.com/princeton-nlp/SimCSE'>Github Repo</a></p>"  # 界面底部的文章链接
examples = [
    ["There's a kid on a skateboard.",  # 示例输入文本列表
    "A kid is skateboarding.",
    "A kid is inside the house."]
]

# 构建交互界面，接收输入文本，显示余弦相似度
gr.Interface(simcse, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()
```