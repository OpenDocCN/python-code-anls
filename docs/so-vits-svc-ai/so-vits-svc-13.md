# SO-VITS-SVC源码解析 13

# `vdecoder/nsf_hifigan/utils.py`

这段代码的主要作用是定义一个名为 `plot_spectrogram` 的函数，用于在 Python 中绘制一个光谱图(spectrogram)。

具体来说，它实现了以下几个步骤：

1. 导入所需的模块和库：`import glob, os, matplotlib, matplotlib.pylab, torch, torch.nn.utils, weight_norm`

2. 定义了 `plot_spectrogram` 函数。函数接收一个 `spectrogram` 参数，它是一个音频信号的幅度图像。函数使用 `import matplotlib` 和 `import torch` 模块，来实现对 `matplotlib` 和 `torch` 模块的引用。

3. 在函数内部，使用 `torch.nn.utils.weight_norm` 模块对 `torch` 对象进行预处理，以便在函数中可以方便地使用 `weight_norm` 方法对 `torch` 对象进行加权平均。

4. 调用 `plot_spectrogram` 函数，并将 `spectrogram` 参数传递给它。函数返回一个包含 `fig` 和 `ax` 对象的元组。函数使用 `plt.subplots` 方法创建一个新的图像窗口，然后使用 `imshow` 函数将 `spectrogram` 图形化显示。函数使用 `plt.colorbar` 函数来显示曲线的颜色条纹。

5. 最后，函数使用 `fig.canvas.draw` 方法来绘制图像，并使用 `plt.close` 方法关闭绘图窗口。

因此，这段代码的作用是定义了一个可以绘制音频信号光谱图的函数，用于对音频信号进行可视化处理。


```py
import glob
import os

import matplotlib
import matplotlib.pylab as plt
import torch
from torch.nn.utils import weight_norm

matplotlib.use("Agg")


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


```



这段代码定义了四个函数，它们在机器学习领域中可能有不同的用途。这里简要介绍一下每个函数的作用：

1. `init_weights`函数接受一个参数`m`，表示一个张量(2D或3D)，以及两个参数`mean`和`std`，分别表示张量中的均值和方差。这个函数的作用是在创建张量时对张量中的数据进行初始化，确保张量的数据在同一分布下。如果张量中的数据已经对齐，`mean`和`std`参数将用于对张量中的数据进行归一化。

2. `apply_weight_norm`函数与`init_weights`函数类似，只不过它不对张量中的数据进行归一化。这个函数的作用是应用一个常见的权重规范化技术，将张量中的数据缩放到一个指定的大小。

3. `get_padding`函数接受两个参数，一个是`kernel_size`，表示一个卷积核的大小，另一个是`dilation`，表示对卷积核的尺寸进行步长的程度。这个函数的作用是在使用卷积操作时确定一个合适的步长，以避免边缘效应。如果使用的是随机初始化的卷积核，该函数将返回一个填充卷积核的尺寸。

4. `mean_initialization`函数接受一个参数`m`，表示一个张量。这个函数的作用是在创建张量时设置一个默认的均值为`mean`，并将张量中的所有数据归一化到均值上。如果张量中的数据已经对齐，该函数将使用张量中的数据初始化均值。


```py
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


```

这段代码定义了三种函数：load_checkpoint、save_checkpoint和del_old_checkpoints。它们的主要作用是加载和保存checkpoint文件，以及在特定情况下删除旧的checkpoint文件。

1. load_checkpoint函数：

这个函数接受两个参数：一个文件路径和一个设备。它首先检查文件是否存在，然后加载并返回checkpoint文件中的数据。如果文件存在，它将打印文件名。然后，它将使用map_location参数将checkpoint文件中的数据加载到指定的设备上。最后，它将打印"Complete."并返回加载的checkpoint数据。

2. save_checkpoint函数：

这个函数接受一个文件路径和一个对象（通常是模型或数据集）。它将保存该对象到指定的文件路径，并打印"Saving checkpoint to {}'".然后，它使用torch.save函数将对象保存到文件中。最后，它打印"Complete."并返回保存的checkpoint数据。

3. del_old_checkpoints函数：

这个函数需要两个参数：一个checkpoint目录和一个前缀（通常是数字）和一个最大模型数（通常是2）。它将查找目录中的所有旧的checkpoint文件，并删除那些早于前缀模型的文件。如果目录中存在更多旧的checkpoint文件，它将删除除了第n个模型以外的所有文件。


```py
def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def del_old_checkpoints(cp_dir, prefix, n_models=2):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern) # get checkpoint paths
    cp_list = sorted(cp_list)# sort by iter
    if len(cp_list) > n_models: # if more than n_models models are found
        for cp in cp_list[:-n_models]:# delete the oldest models other than lastest n_models
            open(cp, 'w').close()# empty file contents
            os.unlink(cp)# delete file (move to trash when using Colab)


```

该函数的作用是检查指定的文件夹中是否存在某个名为 "???" 的文件，如果文件存在，则返回该文件的路径。具体实现方式如下：

1. 首先，通过 os.path.join() 方法将文件夹路径和文件名拼接起来，得到目标文件路径为 cp_dir/prefix + '???'。

2. 接着，使用 glob.glob() 方法在文件夹路径中查找所有文件名以 "???" 开头的文件。这里使用的是 join() 方法将文件名和文件夹路径拼接起来，以便作为过滤条件。

3. 通过 sorted() 方法对找到的文件进行排序，取出最后一个文件，即该文件是文件夹中的主要文件，返回它的路径。

4. 最后，函数返回 None，表示没有找到匹配的文件。


```py
def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


```

# `vencoder/CNHubertLarge.py`

这段代码定义了一个名为CNHubertLarge的类，继承自SpeechEncoder类，用于实现中文语音编码任务。

具体来说，它通过引入torch库和fairseq库，加载了一个预训练的中文胡宇b大模型，并初始化了一个具有1024个隐藏层 dimension 的模型。

在训练过程中，如果设备为 cpu，则使用 cuda 设备进行推理；否则，使用cpu设备。

在模型编码音频信号时，将其先将特征图的维度从 2 转换为 1，这样可以将 feature 图的维度与输入的 audio signal 保持一致，方便输入到 model 中。

然后传递输入到 model 的 extract_features 方法中，返回模型的第一个隐藏层 output。

最后，使用 de-normalization 方法将输出结果返回。


```py
import torch
from fairseq import checkpoint_utils

from vencoder.encoder import SpeechEncoder


class CNHubertLarge(SpeechEncoder):
    def __init__(self, vec_path="pretrain/chinese-hubert-large-fairseq-ckpt.pt", device=None):
        super().__init__()
        print("load model(s) from {}".format(vec_path))
        self.hidden_dim = 1024
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
          [vec_path],
          suffix="",
        )
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        self.model = models[0].to(self.dev)
        self.model.eval()

    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
          "source": feats.to(wav.device),
          "padding_mask": padding_mask.to(wav.device)
        }
        with torch.no_grad():
            logits = self.model.extract_features(**inputs)
        return logits[0].transpose(1, 2)
```

# `vencoder/ContentVec256L12_Onnx.py`

这段代码是一个基于ONNX运行时库和PyTorch实现的代码，主要目的是创建一个名为"ContentVec256L12_Onnx"的类，用于在输入音频波形（即音频信号）上进行语音编码。

具体来说，这段代码执行以下操作：

1. 加载预训练的词汇表（即已经训练好的模型）和一个用于保存模型文件的路径。

2. 如果指定了设备是GPU，则创建一个利用GPU的设备对象，否则创建一个CPU设备对象。

3. 加载预训练的语音编码器模型，这个模型包含一个隐含层（即图中的"SpeechEncoder"部分）。

4. 创建一个名为"ContentVec256L12"的类，继承自SpeechEncoder类，这个类包含一个名为"encoder"的函数，接受一个音频波形作为输入，并返回一个经过编码的音频信号。

5. 在"encoder"函数中，将输入的音频波形通过ONNX运行时库获得的输入数据结构（即Numpy数组）转换为P Torch张量，然后将这个张量提供给ONNX运行时库。

6. 使用ONNX运行时库中的InferenceSession函数加载预训练的词汇表，并将它提供给SpeechEncoder类中的encoder函数。

7. 在SpeechEncoder类中，初始化模型参数，包括隐藏层维度和输入音频设备。

8. 在SpeechEncoder类中的encoder函数中，将输入的音频波形和模型参数一起提供给计算图，然后执行计算图上的操作，并将结果返回。


```py
import onnxruntime
import torch

from vencoder.encoder import SpeechEncoder


class ContentVec256L12_Onnx(SpeechEncoder):
    def __init__(self, vec_path="pretrain/vec-256-layer-12.onnx", device=None):
        super().__init__()
        print("load model(s) from {}".format(vec_path))
        self.hidden_dim = 256
        if device is None:
            self.dev = torch.device("cpu")
        else:
            self.dev = torch.device(device)

        if device == 'cuda' or device == torch.device("cuda"):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.model = onnxruntime.InferenceSession(vec_path, providers=providers)

    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        feats = feats.unsqueeze(0).cpu().detach().numpy()
        onnx_input = {self.model.get_inputs()[0].name: feats}
        logits = self.model.run(None, onnx_input)
        return torch.tensor(logits[0]).transpose(1, 2).to(self.dev)

```

# `vencoder/ContentVec256L9.py`

这段代码是一个基于PyTorch的语音编码器实现。它使用Fairseq库从预训练的500个WAV文件中加载一个名为"pretrain/checkpoint_best_legacy_500.pt"的模型，并使用CUDA（如果可用）或CPU设备来运行它。

代码中定义了一个名为"ContentVec256L9"的类，它继承自SpeechEncoder类。这个类继承了SpeechEncoder类中的__init__方法，用于加载预训练的模型、设置模型参数和初始化模型。

在SpeechEncoder类中，使用checkpoint_utils库从预训练模型中加载模型、设置设备参数，并将模型设置为非托管（model.to("cpu")）。然后加载了属于SpeechEncoder类之一的第一个模型，并将其设置为托管（model.model.to("cpu")）。

最后，在SpeechEncoder类中的一个名为"encoder"的方法中，将输入的WAV数据转换为包含两个通道的单通道数据，并将其输入到加载的第一个模型中。该模型还包括一个带有输出层9的卷积层，用于获取最终的语音编码。


```py
import torch
from fairseq import checkpoint_utils

from vencoder.encoder import SpeechEncoder


class ContentVec256L9(SpeechEncoder):
    def __init__(self, vec_path="pretrain/checkpoint_best_legacy_500.pt", device=None):
        super().__init__()
        print("load model(s) from {}".format(vec_path))
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
          [vec_path],
          suffix="",
        )
        self.hidden_dim = 256
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        self.model = models[0].to(self.dev)
        self.model.eval()

    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
          "source": feats.to(wav.device),
          "padding_mask": padding_mask.to(wav.device),
          "output_layer": 9,  # layer 9
        }
        with torch.no_grad():
            logits = self.model.extract_features(**inputs)
            feats = self.model.final_proj(logits[0])
        return feats.transpose(1, 2)

```

# `vencoder/ContentVec256L9_Onnx.py`

这段代码是一个基于ONNX Runtime和PyTorch的语音编码器。它的主要作用是实现一个将输入音频（WAV格式）转换为输出文本的功能。

具体来说，这段代码包括以下几个步骤：

1. 加载预训练的ONNX模型，这个模型具有两个隐藏层，每层有256个节点。

2. 如果要使用CPU作为设备，就创建一个CPU设备对象；如果使用GPU（CUDA）作为设备，就创建一个GPU设备对象。

3. 在__init__方法中，初始化一个内容向量（vector）变量，这个向量用于保存模型参数。

4. 在encoder方法中，将输入的WAV音频数据转换为包含一个批次（batch）时间戳的1维向量。这个批次向量被传递给模型，然后使用model.run()函数来运行模型以获得编码结果。最后，将编码结果转换为PyTorch张量类型，然后使其与device指定的事器（如nullDevice）一起使用，以便将其移动到设备上。


```py
import onnxruntime
import torch

from vencoder.encoder import SpeechEncoder


class ContentVec256L9_Onnx(SpeechEncoder):
    def __init__(self, vec_path="pretrain/vec-256-layer-9.onnx", device=None):
        super().__init__()
        print("load model(s) from {}".format(vec_path))
        self.hidden_dim = 256
        if device is None:
            self.dev = torch.device("cpu")
        else:
            self.dev = torch.device(device)
        if device == 'cpu' or device == torch.device("cpu") or device is None:
            providers = ['CPUExecutionProvider']
        elif device == 'cuda' or device == torch.device("cuda"):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.model = onnxruntime.InferenceSession(vec_path, providers=providers)

    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        feats = feats.unsqueeze(0).cpu().detach().numpy()
        onnx_input = {self.model.get_inputs()[0].name: feats}
        logits = self.model.run(None, onnx_input)
        return torch.tensor(logits[0]).transpose(1, 2).to(self.dev)
    
```

# `vencoder/ContentVec768L12.py`

这段代码是一个基于PyTorch实现的SpeechEncoder类，用于将文本转换为语音信号。它的实现基于一个预训练的Fairseq模型，使用七个64个细胞的LSTM作为encoder，同时使用一个预训练的预处理语料。在训练时，它使用一个未指定的device，如果是CUDA可用，则将在设备上进行训练。

该类的一个实例可以被创建，并且在__init__方法中设置了一个VectorSpeechEncoder类的一个隐含层，其大小为768。之后，它使用checkpoint_utils库从预训练模型中加载了整个模型、任务和配置文件。如果设备没有被指定，则将其设置为第一个可用的CUDA设备。

在编码器部分，它包含一个从输入文本到语音信号的forward pass。在该forward pass中，它将输入文本的每个单词转换为一个包含多个特征的点，然后将这些点输入到已经训练好的LSTM模型中。通过提取模型的第一个隐藏层中的输出，它得到一个带有预处理文本的二维表示，这个表示被转换为从左到右的分数，表示每个单词的音素。最后，它将这些分数的逆序嵌入到输入文本中，以生成完整的语音信号。

由于该代码片段缺少完整的类和函数定义，因此无法提供更多有关该代码实际实现的确切细节。


```py
import torch
from fairseq import checkpoint_utils

from vencoder.encoder import SpeechEncoder


class ContentVec768L12(SpeechEncoder):
    def __init__(self, vec_path="pretrain/checkpoint_best_legacy_500.pt", device=None):
        super().__init__()
        print("load model(s) from {}".format(vec_path))
        self.hidden_dim = 768
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
          [vec_path],
          suffix="",
        )
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        self.model = models[0].to(self.dev)
        self.model.eval()

    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
          "source": feats.to(wav.device),
          "padding_mask": padding_mask.to(wav.device),
          "output_layer": 12,  # layer 12
        }
        with torch.no_grad():
            logits = self.model.extract_features(**inputs)
        return logits[0].transpose(1, 2)

```

# `vencoder/ContentVec768L12_Onnx.py`

这段代码的作用是实现一个名为"ContentVec768L12_Onnx"的类，该类继承自SpeechEncoder类。这个类接受一个用于存储预训练向量的文件路径，以及一个用于在训练时或预测时使用的工作设备。

具体来说，这个类包含了一个__init__方法，用于初始化对象并加载预训练模型。如果工作设备（设备类型）为'cuda'，则加载CUDA执行器；否则，加载CPU执行器。加载预训练模型后，将模型存储在self.model中，以便在编码过程中进行调用。

另外，这个类包含一个名为encoder的函数，用于将输入的波形（wav）编码为文本向量。这个函数的输入是经过预处理的一个波形，输出是一个文本向量，表示编码后的语音信号。


```py
import onnxruntime
import torch

from vencoder.encoder import SpeechEncoder


class ContentVec768L12_Onnx(SpeechEncoder):
    def __init__(self, vec_path="pretrain/vec-768-layer-12.onnx", device=None):
        super().__init__()
        print("load model(s) from {}".format(vec_path))
        self.hidden_dim = 768
        if device is None:
            self.dev = torch.device("cpu")
        else:
            self.dev = torch.device(device)

        if device == 'cuda' or device == torch.device("cuda"):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
            
        self.model = onnxruntime.InferenceSession(vec_path, providers=providers)

    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        feats = feats.unsqueeze(0).cpu().detach().numpy()
        onnx_input = {self.model.get_inputs()[0].name: feats}
        logits = self.model.run(None, onnx_input)
        return torch.tensor(logits[0]).transpose(1, 2).to(self.dev)

```

# `vencoder/ContentVec768L9_Onnx.py`

这段代码的作用是定义一个名为 ContentVec768L9_Onnx 的类，该类继承自 PyTorch 中的 SpeechEncoder 类。这个类的目标是实现一个将输入文本转换为向量表示的函数，以便在向量表示的基础上进行后续的处理。

具体来说，这段代码执行以下操作：

1. 加载一个名为 "pretrain/vec-768-layer-9.onnx" 的 ONNX 模型，这个模型已经在 ONNX 文档中定义。
2. 如果指定了设备（用 None 表示），则使用 CPU 设备。否则，使用 GPU 设备。
3. 在 encoder 方法中，将输入的 WAV 音频数据转换为特征向量，这个向量有 2 层，因为是双通道音频。
4. 如果输入的特征向量只有 1 层，则将其转换为 1 个维度，表示其只需要一个时间步。
5. 在 encoder 方法中，将 ONNX 模型的输入设置为上述转换后的特征向量，并运行模型的 forward 方法。
6. 将训练好的 ONNX 模型保存到文件 "model.pth"。

这段代码的主要目的是实现一个将输入文本转换为向量表示的函数，以便在向量表示的基础上进行后续的处理。这个函数可以被用来执行各种任务，如语音识别、语音合成等。


```py
import onnxruntime
import torch

from vencoder.encoder import SpeechEncoder


class ContentVec768L9_Onnx(SpeechEncoder):
    def __init__(self,vec_path = "pretrain/vec-768-layer-9.onnx",device=None):
        super().__init__()
        print("load model(s) from {}".format(vec_path))
        self.hidden_dim = 768
        if device is None:
            self.dev = torch.device("cpu")
        else:
            self.dev = torch.device(device)

        if device == 'cuda' or device == torch.device("cuda"):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
            
        self.model = onnxruntime.InferenceSession(vec_path, providers=providers)

    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        feats = feats.unsqueeze(0).cpu().detach().numpy()
        onnx_input = {self.model.get_inputs()[0].name: feats}
        logits = self.model.run(None, onnx_input)
        return torch.tensor(logits[0]).transpose(1, 2).to(self.dev)

```

# `vencoder/DPHubert.py`

这段代码定义了一个名为 "DPHubert" 的类，继承自名为 "SpeechEncoder" 的类。该类表示一个利用 DP（DeepPointing）和 Hubert 模型来进行语音编码的模块。

在类的初始化函数 "__init__" 中，首先调用父类的初始化函数，然后根据所使用的 GPU 设备类型加载预训练模型。如果使用的是 CPU 设备，则不会做任何 GPU 相关的设置。然后加载预训练模型的参数，并将这些参数保存到变量 "self.model" 中。

接着定义了 "encoder" 函数，该函数以语音信号为输入，并将其转换为模型可以处理的 format。如果输入信号只有两个通道（即 double channels），则需要将其转换为单通道。然后对输入信号进行 mean 操作，并将其输入到模型的 "config" 参数中。最后，将输入信号输入到模型中，获取模型的输出，并返回输出的一维数据。

最后，在代码的最后部分，使用 torch.no_grad 清除梯度计算，并使用模型对输入数据进行编码。


```py
import torch

from vencoder.dphubert.model import wav2vec2_model
from vencoder.encoder import SpeechEncoder


class DPHubert(SpeechEncoder):
    def __init__(self, vec_path="pretrain/DPHuBERT-sp0.75.pth", device=None):
        super().__init__()
        print("load model(s) from {}".format(vec_path))
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        ckpt = torch.load(vec_path)
        self.hidden_dim = 768
        self.model = wav2vec2_model(**ckpt["config"]).to(self.dev)
        self.model.load_state_dict(ckpt["state_dict"], strict=False)

    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats[None, :]
        with torch.no_grad():
            with torch.inference_mode():
                units = self.model(feats)[0]
                return units.transpose(1,2)

```

# `vencoder/encoder.py`

这段代码定义了一个名为SpeechEncoder的类，旨在实现语音编码器。SpeechEncoder类包含了一个初始化方法(__init__)，一个编码方法(encoder)，以及一个嵌入维度(hidden_dim)。

在初始化方法(__init__)中，SpeechEncoder实例化了一个SpeechEncoder模型，该模型包含一个嵌入维度为768的隐藏层。SpeechEncoder实例没有指定输入数据文件和设备。

SpeechEncoder的编码方法(encoder)实现了将输入信号(wav)转换为嵌入向量的功能。具体来说，该方法将输入信号与一个预先训练好的模型(SpeechEncoder模型)进行融合，通过替换原始信号中的部分元素来更新原始信号的嵌入向量，从而实现语音编码。

SpeechEncoder类中没有定义具体的实现细节，因此无法得知该类是否支持更多的功能。


```py
class SpeechEncoder(object):
    def __init__(self, vec_path="pretrain/checkpoint_best_legacy_500.pt", device=None):
        self.model = None  # This is Model
        self.hidden_dim = 768
        pass


    def encoder(self, wav):
        """
        input: wav:[signal_length]
        output: embedding:[batchsize,hidden_dim,wav_frame]
        """
        pass

```

# `vencoder/HubertSoft.py`

这段代码是一个基于PyTorch的语音编码器实现，它使用Hubert Soft模型作为其基础模型。具体来说，这段代码定义了一个名为HubertSoft的类，继承自SpeechEncoder类，用于实现将音频信号转换为文本输出的过程。

在HubertSoft类中，代码首先引入了PyTorch支持和Hubert模型的预训练设置，然后定义了一个构造函数，用于初始化Hubert模型，并指定使用的硬件设备（如果使用的是GPU，则设备为0）。

在HubertSoft的encoder方法中，将传入的音频信号（wav格式）输入到Hubert模型的 forward 方法中，得到一个包含文本输出的结果。如果输入音频信号是双通道的，则需要将其平均化成为一个通道的音频信号。

另外，为了提高模型的训练速度和准确性，HubertSoft还定义了一个优化器，在训练时使用。


```py
import torch

from vencoder.encoder import SpeechEncoder
from vencoder.hubert import hubert_model


class HubertSoft(SpeechEncoder):
    def __init__(self, vec_path="pretrain/hubert-soft-0d54a1f4.pt", device=None):
        super().__init__()
        print("load model(s) from {}".format(vec_path))
        hubert_soft = hubert_model.hubert_soft(vec_path)
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        self.hidden_dim = 256
        self.model = hubert_soft.to(self.dev)

    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats[None,None,:]  
        with torch.no_grad():
            with torch.inference_mode():
                units = self.model.units(feats)
                return units.transpose(1,2)

```

# `vencoder/HubertSoft_Onnx.py`

这段代码实现了一个基于ONNX Runtime的语音编码器，该编码器使用Hubert Soft模型的预训练权重在CPU或GPU上运行。具体来说，它继承了SpeechEncoder类，并在其构造函数中加载了模型和初始化了一些变量，包括隐藏层维度和设备。

在encoder函数中，将输入的WAV格式数据转换为1维的feats数据，并将其输入到模型中。由于输入数据是double channels，因此feats数据也应该是2维的。在函数内部，还实现了输入数据的 unsqueeze() 操作，并使用cpu() 方法将其转换为numpy数组。最后，将模型返回的logits转换为PyTorch张量，并使用device属性指定输出设备（即GPU）。


```py
import onnxruntime
import torch

from vencoder.encoder import SpeechEncoder


class HubertSoft_Onnx(SpeechEncoder):
    def __init__(self, vec_path="pretrain/hubert-soft.onnx", device=None):
        super().__init__()
        print("load model(s) from {}".format(vec_path))
        self.hidden_dim = 256
        if device is None:
            self.dev = torch.device("cpu")
        else:
            self.dev = torch.device(device)

        if device == 'cuda' or device == torch.device("cuda"):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
            
        self.model = onnxruntime.InferenceSession(vec_path, providers=providers)

    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        feats = feats.unsqueeze(0).cpu().detach().numpy()
        onnx_input = {self.model.get_inputs()[0].name: feats}
        logits = self.model.run(None, onnx_input)
        return torch.tensor(logits[0]).transpose(1, 2).to(self.dev)

```

# `vencoder/WavLMBasePlus.py`

这段代码是一个基于PyTorch实现的WavLM语音编码器。它的主要作用是将输入的Wav波形数据转化为语音信号，使得输入的语音信号可以被转化为文本、语音识别或其他语音应用需要的格式。

具体来说，这段代码的实现包括以下几个步骤：

1. 加载WavLM模型的预训练权重，这个权重是在训练之前通过训练模型下载的。

2. 如果用户指定了使用GPU设备，则加载GPU上的WavLM模型，否则加载CPU上的WavLM模型。

3. 在WavLM模型的基础上，实现了一个SpeechEncoder类，这个类继承自PyTorch中的SpeechEncoder类，实现了WavLM模型的语音编码器功能。

4. 在SpeechEncoder类中，实现了一个encoder函数，这个函数接受一个Wav波形作为输入，并返回一个编码后的语音信号。

5. 在encoder函数中，首先将输入的Wav波形数据经过一系列的处理，例如降采样、量化、预处理等，使得输入的Wav数据符合WavLM模型的输入要求。

6. 加载WavLM模型的配置文件，这个配置文件包含了WavLM模型的各种参数和设置，例如嵌入维度、隐藏维度、训练设置等。

7. 加载WavLM模型，这个模型是使用WaxLM配置文件来定义的。

8. 将WaxLM模型加载到WavLMBasePlus类中，并实现了SpeechEncoder类中的encoder函数。

9. 由于WavLM模型是用于语音编码的，因此它的输入需要是一个包含多个时间步的Wav数据，每个时间步包含了一个样本的语音信号。在SpeechEncoder类中，通过实现了一个Tokenizer类，用来将Wav数据中的每个样本转换为一个文本字符串，并使用这些字符串来启动WavLM模型，实现了一个从文本到语音的语音编码。


```py
import torch

from vencoder.encoder import SpeechEncoder
from vencoder.wavlm.WavLM import WavLM, WavLMConfig


class WavLMBasePlus(SpeechEncoder):
    def __init__(self, vec_path="pretrain/WavLM-Base+.pt", device=None):
        super().__init__()
        print("load model(s) from {}".format(vec_path))
        checkpoint = torch.load(vec_path)
        self.cfg = WavLMConfig(checkpoint['cfg'])
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        self.hidden_dim = self.cfg.encoder_embed_dim
        self.model = WavLM(self.cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.dev).eval()

    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        if self.cfg.normalize:
            feats = torch.nn.functional.layer_norm(feats, feats.shape)
        with torch.no_grad():
            with torch.inference_mode():
                units = self.model.extract_features(feats[None, :])[0]
                return units.transpose(1, 2)

```

# `vencoder/WhisperPPG.py`

这段代码是一个名为WhisperPPG的类，它继承自SpeechEncoder类，用于实现将音频转换为语音密钥的过程。以下是代码的作用解释：

1. 引入了PyTorch库，以及SpeechEncoder和Whisper模型的类。
2. 定义了一个WhisperPPG类，它继承自SpeechEncoder类，并在其中重写了__init__方法。
3. 在WhisperPPG类中，初始化了一个SpeechEncoder类中没有的方法，即设备、模型和隐藏维度。
4. 重写了SpeechEncoder类中的encoder方法，将输入的音频波形（wav）转换为语音密钥。
5. 在WhisperPPG类中，还实现了两个与SpeechEncoder类中方法相同的辅助方法，pad_or_trim和log_mel_spectrogram。
6. 最后，在训练时加载了一个已经训练好的Whisper模型，并将它与WhisperPPG类的模型中保存的模型状态共享，以便在预测时使用训练好的模型。


```py
import torch

from vencoder.encoder import SpeechEncoder
from vencoder.whisper.audio import log_mel_spectrogram, pad_or_trim
from vencoder.whisper.model import ModelDimensions, Whisper


class WhisperPPG(SpeechEncoder):
    def __init__(self, vec_path="pretrain/medium.pt", device=None):
        super().__init__()
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        checkpoint = torch.load(vec_path, map_location=device)
        dims = ModelDimensions(**checkpoint["dims"])
        model = Whisper(dims)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.hidden_dim = dims
        self.model = model.to(self.dev)

    def encoder(self, wav):
        audio = wav
        audln = audio.shape[0]
        ppgln = audln // 320
        audio = pad_or_trim(audio)
        mel = log_mel_spectrogram(audio).to(self.dev)
        with torch.no_grad():
            ppg = self.model.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
            ppg = torch.FloatTensor(ppg[:ppgln, ]).to(self.dev)
            return ppg[None, :, :].transpose(1, 2)

```

# `vencoder/WhisperPPGLarge.py`

这段代码是一个基于PyTorch实现的语音编码器，主要用于将音频信号转换为在高通情况下更易于处理的信号。

具体来说，它包括以下步骤：

1. 加载预训练的Whisper模型，该模型具有多个隐藏层和一些额外的成员变量，如词向量等。
2. 如果使用的是GPU，则加载音频数据，否则使用CPU加载。
3. 对音频信号进行填充或截取，使其具有至少320个时钟周期，以便在训练时能够准确地重构Whisper模型。
4. 将预处理后的音频信号转换为在高通情况下更易处理的信号。
5. 在编码过程中，将每2个时钟周期（或更少的周期）的音频信号输入到模型中，以获得低频部分。
6. 将低频部分乘以一个系数，以增加编码器的速度。
7. 对输入的音频信号进行逆运算，以恢复原始音频数据。
8. 将生成的音频数据存储到内存中，以便后续的分析和处理。


```py
import torch

from vencoder.encoder import SpeechEncoder
from vencoder.whisper.audio import log_mel_spectrogram, pad_or_trim
from vencoder.whisper.model import ModelDimensions, Whisper


class WhisperPPGLarge(SpeechEncoder):
    def __init__(self, vec_path="pretrain/large-v2.pt", device=None):
        super().__init__()
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        checkpoint = torch.load(vec_path, map_location=device)
        dims = ModelDimensions(**checkpoint["dims"])
        model = Whisper(dims)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.hidden_dim = dims
        self.model = model.to(self.dev)

    def encoder(self, wav):
        audio = wav
        audln = audio.shape[0]
        ppgln = audln // 320
        audio = pad_or_trim(audio)
        mel = log_mel_spectrogram(audio).to(self.dev)
        with torch.no_grad():
            ppg = self.model.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
            ppg = torch.FloatTensor(ppg[:ppgln, ]).to(self.dev)
            return ppg[None, :, :].transpose(1, 2)

```

# `vencoder/__init__.py`

我需要更具体的上下文来回答你的问题。可以请你提供一下代码或者提供一些关于代码的上下文信息吗？这样我才能够更好地解释代码的作用。


```py

```

# `vencoder/dphubert/components.py`

这段代码定义了一个名为“Building blocks for speech SSL models supporting pruning”的类。从代码中可以看出，它主要用于在PyTorch中构建支持剪枝的语音SSL模型。

具体来说，这段代码包括以下几个主要部分：

1. 定义了一个名为“PruneModel”的类，用于实现语音SSL模型的构建和训练。
2. 定义了一个名为“PruneModel”的类，用于实现语音SSL模型的构建和训练。
3. 定义了一个名为“Module”的类，用于实现PyTorch中的一个模型。
4. 定义了一个名为“Wav2Vec2Module”的类，用于实现语音SSL中的Wav2Vec2模型的构建和训练。
5. 定义了一个名为“Linear1激活函数”的函数，用于实现PyTorch中的一个激活函数。
6. 定义了一个名为“Prune”的函数，用于实现对PyTorch中的一个张量进行剪枝操作。
7. 定义了一个名为“Gather激活函数”的函数，用于实现PyTorch中的一个激活函数。
8. 定义了一个名为“ prune_seq”的函数，用于实现对一个序列进行剪枝操作。
9. 定义了一个名为“ build_superset”的函数，用于实现语音SSL模型的构建。
10. 定义了一个名为“train_loop”的函数，用于实现语音SSL模型的训练。

这段代码的作用是提供一个用于构建和支持剪枝的语音SSL模型的类和函数。这个类和函数可以在语音SSL模型的训练和构建中使用，从而提高模型的训练效率和性能。


```py
"""Building blocks for speech SSL models supporting pruning.

Originally from:
https://github.com/pytorch/audio/blob/main/torchaudio/models/wav2vec2/components.py

"""

import math
from collections import defaultdict
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import Module

```

这段代码定义了一个函数 `_init_transformer_params`，它接收一个 `module` 对象作为参数。

在函数内部，首先检查 `module` 是否为 `nn.Linear`，如果是，则执行该层的权重归一化和初始化。接下来，如果是 `nn.Embedding`，则归一化嵌入层的权重，并检查是否有可用的缓冲区大小 `padding_idx`。最后，根据 `module` 的类型对参数进行归一化或初始化。

如果 `module` 是 `nn.Functional`，则根据定义的 `forward` 方法对 `module` 进行参数归一化和初始化。


```py
from .hardconcrete import HardConcrete
from .pruning_utils import (
    prune_conv1d_layer,
    prune_layer_norm,
    prune_linear_layer,
)


def _init_transformer_params(module):
    """
    Initialize the weights of Transformer module in Wav2Vec2/HuBERT.

    If the module is ``nn.Linear``, normalize the weight with mean 0 and standard deviation 0.02.
    If ``bias`` is set to ``True`` in the module, set ``bias`` to 0.

    If the module is ``nn.Embedding``, normalize the weight with mean 0 and standard deviation 0.02.
    If ``padding_idx`` is not None, set the weight of padding to 0.

    Note:
        Ths method corresponds to
        `init_bert_params
        <https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/transformer_sentence_encoder.py#L21>`__
        in the original ``fairseq`` implementation.
    """

    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


```

This is a PyTorch implementation of a 2D convolutional neural network (CNN) model. It uses a combination of hard-construed representations and learnable representations. The model has an input layer, a variable number of convolutional layers with differentkernel_size, a normalization layer, and a linear layer. The convolutional layers use the gelu activation function, which is a softmax function that outputs a value between 0 and 1. The hard-construed representations are stored in the variables self.hard_concrete and self.conv. The hard-construed representations are used for the convolutional filters, and the convolutional filters are updated during training. The learnable representations are stored in the variables self.conv.


```py
class LayerNorm(nn.LayerNorm):
    """Layer norm with transpose"""

    def forward(self, input: Tensor) -> Tensor:
        x = input.transpose(-2, -1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.transpose(-2, -1)
        return x


class ConvLayerBlock(Module):
    """Convolution unit of FeatureExtractor"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool,
        layer_norm: Optional[Module],
        prune_conv_channels: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer_norm = layer_norm
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )

        if prune_conv_channels:
            self.hard_concrete = HardConcrete(n_in=out_channels, init_mean=0.01)
        else:
            self.hard_concrete = None

    def forward(
        self,
        x: Tensor,
        length: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): Shape: ``[batch, in_channels, in_frame]``.
            length (Tensor or None, optional): Shape ``[batch, ]``.
        Returns:
            Tensor: Shape ``[batch, out_channels, out_frames]``.
            Optional[Tensor]: Shape ``[batch, ]``.
        """
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        x = nn.functional.gelu(x)

        if self.hard_concrete is not None:
            channel_mask = self.hard_concrete()  # hard concrete mask, (out_channels,)
            x = x * channel_mask.unsqueeze(-1)

        if length is not None:
            length = torch.div(length - self.kernel_size, self.stride, rounding_mode="floor") + 1
            # When input length is 0, the resulting length can be negative. So fix it here.
            length = torch.max(torch.zeros_like(length), length)
        return x, length
    
    def get_num_params_and_out_channels(self, in_channels):
        if self.hard_concrete is not None:
            out_channels = self.hard_concrete.l0_norm()
        else:
            out_channels = self.conv.out_channels
        
        num_params = in_channels * out_channels * self.kernel_size
        if self.conv.bias is not None:
            num_params += out_channels
        if self.layer_norm is not None:
            num_params += out_channels * 2
        
        return num_params, out_channels


```

This class seems to be a custom implementation of the convolutional neural network (CNN)pruning algorithm provided by the PyTorch documentation.

It contains a method `prune()` which, based on the hardconcrete parameters, prunes the convolutional layers and dummy weights, and optionally, the layer norm. The hardconcrete parameters are used to define the output of the convolutional layer as a tensor, and this tensor is passed through the pruning algorithm.

The method takes an inplace operation as its argument and returns a tuple consisting of two elements: a list of new configurations and an array of indexes indicating which convolutional layers have been pruned.

The `prune()` method first loops through the convolutional layers and checks if the layer has the hardconcrete parameters set. If it does, the method creates a new configuration by copying the current layer configuration and appending it to the list of new configurations. Then it prunes the current layer by converting the index of the channels to one-dimensional and multiplying it by the batch size. Finally, it updates the hardconcrete parameters and removes the dummy weight.

If the layer does not have the hardconcrete parameters set, the method creates a new configuration by setting the output channels of the convolutional layer to the same value as the input channels and appending it to the list of new configurations. Then it appends the configuration to the list.

The method returns a tuple `(new_config, index)` where `new_config` is the list of new configurations and `index` is an array of indexes indicating which convolutional layers have been pruned.


```py
class FeatureExtractor(Module):
    """Extract features from audio

    Args:
        conv_layers (nn.ModuleList):
            convolution layers
    """

    def __init__(
        self,
        conv_layers: nn.ModuleList,
    ):
        super().__init__()
        self.conv_layers = conv_layers

        # NOTE: a dummy weight used to save the soft mask of the last conv layer
        self.dummy_weight = nn.Parameter(
            torch.ones(conv_layers[-1].conv.out_channels, dtype=torch.float32),
            requires_grad=False
        )

    def forward(
        self,
        x: Tensor,
        length: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor):
                Input Tensor representing a batch of audio,
                shape: ``[batch, time]``.
            length (Tensor or None, optional):
                Valid length of each input sample. shape: ``[batch, ]``.

        Returns:
            Tensor:
                The resulting feature, shape: ``[batch, frame, feature]``
            Optional[Tensor]:
                Valid length of each output sample. shape: ``[batch, ]``.
        """
        if x.ndim != 2:
            raise ValueError("Expected the input Tensor to be 2D (batch, time), " "but received {list(x.shape)}")

        x = x.unsqueeze(1)  # (batch, channel==1, frame)
        for layer in self.conv_layers:
            x, length = layer(x, length)  # (batch, feature, frame)
        x = x.transpose(1, 2)  # (batch, frame, feature)
        x = x * self.dummy_weight
        return x, length

    def get_num_params_and_final_out_channels(self):
        in_channels = 1
        num_params = 0
        for layer in self.conv_layers:
            layer_params, in_channels = layer.get_num_params_and_out_channels(in_channels)
            num_params += layer_params

        num_params += in_channels   # dummy weight
        
        return num_params, in_channels
    
    def prune(self):
        """"Prune conv layers and dummy weight based on hardconcrete parameters.
        This is an in-place operation.
        """
        new_config = []     # [(output_channel, kernel_size, stride), ...]
        for idx, layer in enumerate(self.conv_layers):
            if layer.hard_concrete is not None:
                assert not layer.hard_concrete.training
                mask = layer.hard_concrete()    # (out_features,)
                index = mask.nonzero().squeeze(-1)    # 2D -> 1D
                assert len(index) > 0, f"Conv channels pruned to zero at index {idx}"
                new_config.append(
                    (len(index), layer.kernel_size, layer.stride)
                )

                # prune the current layer
                prune_conv1d_layer(layer.conv, index, "output")
                if layer.layer_norm is not None:
                    prune_layer_norm(layer.layer_norm, index)

                # prune the next layer
                if idx == len(self.conv_layers) - 1:
                    self.dummy_weight.data *= mask
                    self.dummy_weight = nn.Parameter(
                        self.dummy_weight.index_select(0, index).clone().detach(), requires_grad=False
                    )
                else:
                    self.conv_layers[idx+1].conv.weight.data *= mask.unsqueeze(-1)
                    prune_conv1d_layer(self.conv_layers[idx+1].conv, index, dim="input")

                layer.hard_concrete = None
            else:
                new_config.append(
                    (layer.conv.out_channels, layer.kernel_size, layer.stride)
                )
                index = torch.arange(layer.conv.out_channels, dtype=torch.long)

        return new_config, index


```

这段代码定义了一个名为 FeatureProjection 的类，它是一个在 FeatureExtractor 和 Encoder 之间进行连接的层。这个层的作用是将输入的特征（具有 `in_features` 参数）连接到输出层（具有 `out_features` 参数），通过 `dropout` 来防止过拟合。

FeatureProjection 类包含三个方法：

1. `__init__` 方法：用于初始化包含输入层、输出层以及丢弃概率的参数。
2. `forward` 方法：用于前向传递输入，对输入进行层级归一化，并计算输出。
3. `get_num_params` 方法：用于计算输入层特征数目的参数。

由于 `FeatureProjection` 层主要起到连接作用，所以它的实现相对简单。


```py
class FeatureProjection(Module):
    """Layer that connects FeatureExtractor and Encoder

    Projects features to encoder dimension.

    Args:
        in_features (int): Input feature dim.
        out_features (int): Output feature dim.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.projection = nn.Linear(
            in_features,
            out_features,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor):
                Feature Tensor. shape: ``[batch, frame, in_feature]``
        Returns:
            Tensor: Projected features. ``[batch, frame, out_feature]``.
        """
        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x
    
    def get_num_params(self, in_features):
        return in_features * 2 + (in_features + 1) * self.projection.out_features


```

This is a PyTorch implementation of a 1-dimensional convolutional neural network (CNN) that uses a pre-trained VGG16 model as the backbone and adds a depthwise separable convolution to perform depthwise separable training. The network takes an input of shape `[batch, frame, feature]` and returns the output of shape `[batch, frame, feature]`.

The network has been modified to support depthwise separable training by adding a depthwise separable convolution in the feature map. This is done by first adding a batch normalization layer, followed by a depthwise separable convolution, followed by another batch normalization layer. This allows the network to learn more complex patterns at different depths.

The `__prepare_scriptable__` method has been added to remove an instance of the `torch.nn.utils.weight_norm` class, which is not needed in this case because of the depthwise separable convolution.

The `forward` method has been modified to support the new input shape of `x.transpose(-2, -1)` and to return the output of shape `[batch, frame, feature]`.


```py
class ConvolutionalPositionalEmbedding(Module):
    """Positional embedding which is placed at the beginning of Transformer.

    Args:
        embed_dim (int): Feature dimension of the input Tensor.
        kernel_size (int): The number of frames to be use.
        groups (int): The number of groups in feature dimensions.
    """

    def __init__(
        self,
        embed_dim: int,
        kernel_size: int,
        groups: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )

        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
        self.num_remove: int = 1 if kernel_size % 2 == 0 else 0

    def __prepare_scriptable__(self):
        for hook in self.conv._forward_pre_hooks.values():
            # The hook we want to remove is an instance of WeightNorm class, so
            # normally we would do `if isinstance(...)` but this class is not accessible
            # because of shadowing, so we check the module name directly.
            # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
            if hook.__module__ == "torch.nn.utils.weight_norm" and hook.__class__.__name__ == "WeightNorm":
                torch.nn.utils.remove_weight_norm(self.conv)
        return self

    def forward(self, x):
        """
        Args:
            x (Tensor): shape ``[batch, frame, feature]``.

        Returns:
            Tensor: The resulting feature. Shape ``[batch, frame, feature]``.
        """
        x = x.transpose(-2, -1)
        x = self.conv(x)
        if self.num_remove > 0:
            x = x[..., : -self.num_remove]
        x = torch.nn.functional.gelu(x)
        x = x.transpose(-2, -1)
        return x


```



This is a class that implements a multi-layer self-attention model. The model has a parameter called `self.hard_concrete_for_layer`, which is a tensor containing the hard-constraints for each layer of the model. If this tensor is not `None`, it is multiplied by `self.hard_concrete_for_layer.l0_norm()` to update the parameter with the minimum arithmetic operations.

The model also has a function called `prune()`, which removes some parameters from the model. This is done by first checking if the `self.hard_concrete_for_layer` is `None`. If it is not `None`, it creates a new configuration with `use_attention=True` and `num_heads=self.num_heads`, and then multiplies the weight and bias by the layer's mask to distribute them among the layers. Finally, it sets the `self.hard_concrete_for_heads` to `None`.

If `self.hard_concrete_for_layer` is `None` and `self.num_heads` is set, the model will create a new configuration with `use_attention=False` and `num_heads=0`, and then sets the `self.out_proj.weight` and `self.out_proj.bias` to the same as the layer's mask.

Note that the `prune()` function only modifies the parameters of the model and does not modify the computation of the model's output.


```py
class SelfAttention(Module):
    """Multihead Self Attention module

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): The number of heads.
        dropout (float, optional):
            Dropout probability on attn_output_weights. Default: ``0.0``
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        prune_heads: bool = False,  # whether to prune attention heads
        prune_layer: bool = False,  # whether to prune entire attention layers
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = torch.nn.Dropout(dropout)

        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=True)

        if prune_heads:
            self.hard_concrete_for_heads = HardConcrete(n_in=num_heads, init_mean=0.01)
        else:
            self.hard_concrete_for_heads = None

        if prune_layer:
            self.hard_concrete_for_layer = HardConcrete(n_in=1, init_mean=0.01)
        else:
            self.hard_concrete_for_layer = None

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): shape: ``[batch_size, sequence_length, embed_dim]``.
            attention_mask (Tensor or ``None``, optional):
                shape: ``[batch_size, 1, sequence_length, sequence_length]``
            position_bias: Not used. Only for the compatibility with :py:class:`WavLMSelfAttention`.
            key_padding_mask (Tensor or ``None``): Not used. Only for the compatibility with
                :py:class:`WavLMSelfAttention`.
        Returns:
            (Tensor, ``None``): The resulting attention output and ``None`` (necessary for compatibility
                with :py:class:`WavLMSelAttention`).
                Attention output shape: ``[batch, sequence_length, embed_dim]``.
        """
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(
                f"The expected input shape is (batch, sequence, embed_dim=={self.embed_dim}). " f"Found {x.shape}."
            )
        batch_size, length, embed_dim = x.size()
        
        shape = (batch_size, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        k = self.k_proj(x).view(*shape).permute(0, 2, 3, 1)  # B, nH, Hd, L
        v = self.v_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd

        # scale down q to avoid value overflow.
        weights = (self.scaling * q) @ k  # B, nH, L, L
        if attention_mask is not None:
            weights += attention_mask
        # subtracting a constant value from the tensor won't change the output of softmax.
        # apply the subtraction to avoid value overflow in torch.nn.functional.softmax.
        # for more details, please see Equation 7 in https://arxiv.org/abs/2112.08778
        weights = weights - weights.max(dim=-1, keepdim=True)[0]

        weights = torch.nn.functional.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        output = weights @ v  # B, nH, L, Hd

        if self.hard_concrete_for_heads is not None:
            head_mask = self.hard_concrete_for_heads()  # (nH,)
            output = output * head_mask.unsqueeze(-1).unsqueeze(-1)

        output = output.transpose(2, 1).reshape(batch_size, length, self.num_heads * self.head_dim)

        output = self.out_proj(output)

        if self.hard_concrete_for_layer is not None:
            layer_mask = self.hard_concrete_for_layer() # (1,)
            output = output * layer_mask

        return output, None  # Necessary for compatibility with WavLMSelAttention

    def get_num_params(self):
        if self.hard_concrete_for_heads is not None:
            num_heads = self.hard_concrete_for_heads.l0_norm()
        else:
            num_heads = self.num_heads
        num_params = (self.embed_dim + 1) * num_heads * self.head_dim * 3 \
            + (num_heads * self.head_dim + 1) * self.embed_dim

        if self.hard_concrete_for_layer is not None:
            num_params *= self.hard_concrete_for_layer.l0_norm()
        
        return num_params

    def prune(self):
        new_config = {
            "use_attention": True,
            "num_heads": self.num_heads,
        }
        if self.hard_concrete_for_layer is not None:
            assert not self.hard_concrete_for_layer.training
            layer_mask = self.hard_concrete_for_layer() # (1,)
            self.out_proj.weight.data *= layer_mask
            self.out_proj.bias.data *= layer_mask
            if layer_mask == 0:
                new_config["use_attention"] = False
            self.hard_concrete_for_layer = None

        if self.hard_concrete_for_heads is not None:
            assert not self.hard_concrete_for_heads.training
            head_mask = self.hard_concrete_for_heads()  # (num_heads,)
            new_config["num_heads"] = len(head_mask.nonzero())
            if new_config["num_heads"] == 0:
                new_config["use_attention"] = False
            else:
                full_mask = head_mask.repeat_interleave(self.head_dim)
                full_index = full_mask.nonzero().squeeze(-1)  # 1D

                prune_linear_layer(self.k_proj, full_index, "output")
                prune_linear_layer(self.v_proj, full_index, "output")
                prune_linear_layer(self.q_proj, full_index, "output")

                self.out_proj.weight.data *= full_mask
                prune_linear_layer(self.out_proj, full_index, "input")
            self.hard_concrete_for_heads = None

        return new_config


```

This is a class that implements the forward function for a pre-trained language model. The forward function takes in an input sequence and an optional attention mask, and returns the attention output and the position-wise biases of the model.

The attention mechanism is implemented using the `Attention` layer from the `Transformers` library. This layer computes the attention weights for each element in the input sequence by first projecting the input to a fixed-size feature dimension through a linear layer and then taking a softmax function. The attention weights are then used to compute the attention output, which is a weighted sum of the input elements.

The position-wise biases are implemented as additional parameters in the `Attention` layer. These biases are computed as the negative log-likelihood of the input sequence with the attention weights, normalized by the softmax function. This allows the model to focus on the most relevant parts of the input sequence when making predictions.

The class also has a `prune` method that removes some of the `Attention` layers from the model to reduce its size and increase its interpretability. This is done by setting all the weights and biases to zero for the corresponding layers, and only keeping the layers that have already been trained.


```py
class WavLMSelfAttention(SelfAttention):
    """Multi-headed self-attention for WavLM model :cite:`chen2022wavlm`.

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): The number of heads.
        dropout (float, optional): Dropout probability on attn_output_weights. (Default: to ``0.0``)
        bias (bool, optional): If ``True``, add bias to input / output projection layers. (Default: ``True``)
        has_relative_attention_bias (bool, optional): If ``True``, apply relative position embedding.
            Necessary in the first encoder layer, but not in the subsequent ones. (Default: ``False``)
        num_buckets (int, optional): Number of buckets for relative position embedding. (Default: ``32``)
        max_distance (int, optional): Naximum distance for relative position embedding. (Default: ``128``)
        gru_rel_pos (bool, optional): If ``True``, apply gated relative position embedding. (Default: ``False``)
    """

    def __init__(
        self,
        embed_dim: int,
        total_num_heads: int,
        remaining_heads: Optional[List[int]] = None,
        dropout: float = 0.0,
        bias: bool = True,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 32,
        max_distance: int = 128,
        gru_rel_pos: bool = True,
        prune_heads: bool = False,
        prune_layer: bool = False,
    ):
        self.total_num_heads = total_num_heads
        if remaining_heads is None:
            self.remaining_heads = list(range(total_num_heads))
        else:
            self.remaining_heads = remaining_heads  # list of indices
        
        self.head_dim = embed_dim // total_num_heads

        super().__init__(embed_dim, len(self.remaining_heads), self.head_dim, dropout, prune_heads, prune_layer)

        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        if has_relative_attention_bias:
            self.rel_attn_embed = nn.Embedding(num_buckets, total_num_heads)
        else:
            self.rel_attn_embed = None

        # override linear layers to customize bias
        self.k_proj = nn.Linear(embed_dim, len(self.remaining_heads) * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, len(self.remaining_heads) * self.head_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, len(self.remaining_heads) * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(len(self.remaining_heads) * self.head_dim, embed_dim, bias=bias)

        self.gru_rel_pos = gru_rel_pos
        if self.gru_rel_pos:
            self.gru_rel_pos_linear = nn.Linear(self.head_dim, 8)
            self.gru_rel_pos_const = nn.Parameter(torch.ones(1, total_num_heads, 1, 1))
        self.has_position_bias = True

    def compute_bias(self, query_length: int, key_length: int) -> Tensor:
        """Compute relative position embeddings for WavLM model.
        Args:
            query_length (int): Query position can take values between 0 and ``query_length - 1``.
            key_length (int): Key position can take values between 0 and ``key_length - 1``.
        Returns:
            Tensor of shape `(num_heads, query_length, key_length)`, relative positions embeddings
        """
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # Shape (query_length, key_length)
        relative_position_bucket = self._relative_positions_bucket(relative_position, bidirectional=True)
        relative_position_bucket = relative_position_bucket.to(self.rel_attn_embed.weight.device)
        values = self.rel_attn_embed(relative_position_bucket)  # Shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1])
        return values

    def _relative_positions_bucket(self, relative_positions: Tensor, bidirectional: bool = True):
        """Compute relative position buckets for WavLM model. Computation similar to formula (5) in WavLM
           paper :cite:`chen2022wavlm`.
        Args:
            relative_positions (Tensor): Relative offsets between query and key positions,
                of shape ``(query_length, key_length)``.
            bidirectional (bool): If ``True``, values will be filled both above and below the diagonal in the resulting
                matrix. If ``False``, the elements above the diagonal (i.e. with negative relative offsets) will be set
                to zero. (Default ``True``)
        Returns:
            Tensor of shape ``(query_length, key_length)`` filled bucketed values of with relative positions.
        """
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        # Shape (query_length, key_length)
        relative_buckets = torch.zeros_like(relative_positions, dtype=torch.long)

        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_positions > 0).to(torch.long) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = -torch.min(relative_positions, torch.zeros_like(relative_positions))

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_postion_if_large = max_exact + (
            torch.log(relative_positions.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_positions, relative_postion_if_large)
        return relative_buckets

    def forward(
        self,
        query: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            query (Tensor): Input of shape ``(batch_size, src_len, embed_dim)``.
            key_padding_mask (Tensor or None, optional): Mask to exclude keys that are pads, of shape
                `(batch, src_len)`, where padding elements are indicated by 1s. (Default: ``None``)
            attn_mask: Needs to be ``None``. The argument exists for compatibility with
                ``EncoderLayer``. (Default: ``None``)
            position_bias (Tensor or None, optional): Position bias of shape
                ``(batch_size * num_heads, src_len, src_len)``. When used inside WavLM model encoder, will be
                generated in the first layer and then passed from each encoder layer to the next one.
                (Default: ``None``)
        Returns:
            attn_output (Tensor): Attention output of shape ``(batch_size, src_len, embed_dim)``.
            position_bias (Tensor or None): Position bias of shape ``(batch_size * num_heads, src_len, src_len)``.
        """
        bsz, seq_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert key_padding_mask is None

        # only for the first layer
        if self.rel_attn_embed is not None and position_bias is None:
            position_bias = self.compute_bias(seq_len, seq_len)
            position_bias = position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.total_num_heads, seq_len, seq_len)

        attn_mask_rel_pos: Optional[Tensor] = None
        if position_bias is not None:
            attn_mask_rel_pos = position_bias
            if self.gru_rel_pos:  # Apply gating on relative position bias
                query_layer = query.view(bsz, seq_len, self.total_num_heads, -1)
                query_layer = query_layer.permute(0, 2, 1, 3)

                gate_a, gate_b = torch.sigmoid(
                    self.gru_rel_pos_linear(query_layer).view(bsz, self.total_num_heads, seq_len, 2, 4).sum(-1, keepdim=False)
                ).chunk(2, dim=-1)
                gate_a_1 = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0
                attn_mask_rel_pos = gate_a_1.view(bsz * self.total_num_heads, -1, 1) * position_bias

            attn_mask_rel_pos = attn_mask_rel_pos.view((-1, seq_len, seq_len))
            attn_mask_rel_pos = attn_mask_rel_pos.reshape(bsz, self.total_num_heads, seq_len, seq_len)[:, self.remaining_heads, :, :]

        attn_mask = attn_mask_rel_pos
        if attention_mask is not None:
            attn_mask = attn_mask + attention_mask
        if key_padding_mask is not None:
            attn_mask = attn_mask.masked_fill(
                key_padding_mask.reshape(bsz, 1, 1, seq_len),
                float("-inf")
            )
        attn_output, _ = super().forward(query, attention_mask=attn_mask)

        return attn_output, position_bias

    def prune(self):
        new_config = {
            "use_attention": True,
            "remaining_heads": self.remaining_heads,
        }
        if self.hard_concrete_for_layer is not None:
            assert not self.hard_concrete_for_layer.training
            layer_mask = self.hard_concrete_for_layer() # (1,)
            self.out_proj.weight.data *= layer_mask
            self.out_proj.bias.data *= layer_mask
            if layer_mask == 0:
                new_config["use_attention"] = False
            self.hard_concrete_for_layer = None

        if self.hard_concrete_for_heads is not None:
            assert not self.hard_concrete_for_heads.training
            head_mask = self.hard_concrete_for_heads()  # (num_heads,)
            new_config["remaining_heads"] = head_mask.nonzero().squeeze(-1).tolist()
            if len(new_config["remaining_heads"]) == 0:
                new_config["use_attention"] = False
            else:
                full_mask = head_mask.repeat_interleave(self.head_dim)
                full_index = full_mask.nonzero().squeeze(-1)  # 1D

                prune_linear_layer(self.k_proj, full_index, "output")
                prune_linear_layer(self.v_proj, full_index, "output")
                prune_linear_layer(self.q_proj, full_index, "output")

                self.out_proj.weight.data *= full_mask
                prune_linear_layer(self.out_proj, full_index, "input")
            self.hard_concrete_for_heads = None

        return new_config


```

This is a class that defines a neural network model and performs the task of training and pruning it.

The model has two features:

1. An intermediate feature layer, which is a dense layer with intermediate density and non-zero values. This layer is used for storing the intermediate representations from the input data and for computing the output.

2. A feed-forward layer, which is a dense layer with output features and uses the intermediate representations computed in step 1 to compute the output.

The model also has two options:

1. The `hard_concrete_for_layer` property, which is a property of the `IntermediateDense` layer that controls whether it should compute the output in `hard_state` or `soft_state`. If `hard_concrete_for_layer` is `True`, the model will compute the output in `hard_state`.

2. The `hard_concrete_for_intermediate` property, which is a property of the `output_dense` layer that controls whether it should compute the output in `hard_state` or `soft_state`. If `hard_concrete_for_intermediate` is `True`, the model will compute the output in `hard_state`.

The `prune` method of the model is used for removing the parameters that are below zero or that have negative values.

The `out_features` property of the `IntermediateDense` layer is used to compute the number of output features.

The `num_params` property of the model is computed as `(io_features + 1) * intermediate_features + (intermediate_features + 1) * io_features`.

The `IntermediateDense` layer has two options:

1. The `use_feed_forward` property, which is a boolean value that indicates whether the `ff_interm_features` property should be computed. If `use_feed_forward` is `True`, the `ff_interm_features` property will be computed.

2. The `ff_interm_features` property, which is an integer that specifies the number of input features that will be passed through the `ff_interm_features` function.

The `IntermediateDense` layer has two methods:

1. `ff_interm_features`: This method computes the `ff_interm_features` property of the layer by breaking down the input features based on the `ff_interm_features` property. It uses the `mask_` method to compute the output of the layer, where `mask_` is a boolean that specifies the mask to use. This method is useful for training models that use multiple intermediate representations.

2. `output`: This method is used for computing the output of the `ff_interm_features` function. It is computed by passing the input features through the `ff_interm_features` function and then through the feed-forward dense layer.

The `train` method of the model is used to train the model. It takes as input the training data and the model configuration.

The `predict` method of the model is used to predict the output of the model. It takes as input the input data and the model configuration.

The `compile` method of the model is used for computing the loss function and the optimizer.

The `update` method of the model is used for updating the parameters of the model.


```py
class FeedForward(Module):
    """Layer that follows attention layer in encoder layer."""

    def __init__(
        self,
        io_features: int,
        intermediate_features: int,
        intermediate_dropout: float,
        output_dropout: float,
        prune_intermediate: bool = False,
        prune_layer: bool = False,
    ):
        super().__init__()
        self.intermediate_dense = nn.Linear(io_features, intermediate_features)
        self.intermediate_dropout = nn.Dropout(intermediate_dropout)
        self.output_dense = nn.Linear(intermediate_features, io_features)
        self.output_dropout = nn.Dropout(output_dropout)

        if prune_intermediate:
            self.hard_concrete_for_intermediate = HardConcrete(
                n_in=intermediate_features, init_mean=0.5
            )
        else:
            self.hard_concrete_for_intermediate = None
        
        if prune_layer:
            self.hard_concrete_for_layer = HardConcrete(n_in=1, init_mean=0.01)
        else:
            self.hard_concrete_for_layer = None

    def forward(self, x):
        """
        Args:
            x (Tensor): shape: `(batch, sequence_length, io_features)`
        Returns:
            x (Tensor): shape: `(batch, sequence_length, io_features)`
        """
        x = self.intermediate_dense(x)
        x = torch.nn.functional.gelu(x)
        x = self.intermediate_dropout(x)

        if self.hard_concrete_for_intermediate is not None:
            intermediate_mask = self.hard_concrete_for_intermediate()   # (intermediate_features,)
            x = x * intermediate_mask

        x = self.output_dense(x)
        x = self.output_dropout(x)

        if self.hard_concrete_for_layer is not None:
            layer_mask = self.hard_concrete_for_layer()     # (1,)
            x = x * layer_mask

        return x
    
    def get_num_params(self):
        io_features = self.intermediate_dense.in_features
        if self.hard_concrete_for_intermediate is not None:
            intermediate_features = self.hard_concrete_for_intermediate.l0_norm()
        else:
            intermediate_features = self.intermediate_dense.out_features
        num_params = (io_features + 1) * intermediate_features + (intermediate_features + 1) * io_features

        if self.hard_concrete_for_layer is not None:
            num_params *= self.hard_concrete_for_layer.l0_norm()
        
        return num_params
    
    def prune(self):
        new_config = {
            "use_feed_forward": True,
            "ff_interm_features": self.intermediate_dense.out_features
        }
        if self.hard_concrete_for_layer is not None:
            assert not self.hard_concrete_for_layer.training
            layer_mask = self.hard_concrete_for_layer()
            self.output_dense.weight.data *= layer_mask
            self.output_dense.bias.data *= layer_mask
            if layer_mask == 0:
                new_config["use_feed_forward"] = False
            self.hard_concrete_for_layer = None

        if self.hard_concrete_for_intermediate is not None:
            assert not self.hard_concrete_for_intermediate.training
            interm_mask = self.hard_concrete_for_intermediate()
            interm_index = interm_mask.nonzero().squeeze(-1)    # NOTE: must specify dim=-1
            new_config["ff_interm_features"] = len(interm_index)
            if new_config["ff_interm_features"] == 0:
                new_config["use_feed_forward"] = False
            else:
                prune_linear_layer(self.intermediate_dense, interm_index, "output")

                self.output_dense.weight.data *= interm_mask
                prune_linear_layer(self.output_dense, interm_index, "input")
            self.hard_concrete_for_intermediate = None

        return new_config


```

This is a class that wraps the functionality of a neural network model, specifically the Transformer architecture.  It is a subclass of the `TensorFlow.keras.layers.Dense` class and is meant to be used as a custom layer for the Transformer architecture.

This class takes a number of parameters, including the shape of the input tensor, the position of the bias, and the mask for the key in the routing mechanism.

The `forward` method defines the forward pass of the model. It takes the input tensor and applies the model's activation function, followed by any additional operations such as adding a position bias.

The `get_num_params` method returns the total number of parameters in the model.

It is intended to be used as a custom layer for the Transformer architecture and can be used in the following ways:

1. As a regular layer: You can use the `Dense` class to create a new layer and call it on a given input tensor, passing in the desired shape and activation function. For example:
```py
new_layer = MyCustomTransformerLayer(dtype=tf.float32)
```
2. As a custom layer: You can define your own custom layer by implementing the `forward` method and passing it in to the `Dense` class. For example:
```py
class MyCustomTransformerCustomLayer(tf.keras.layers.Dense):
   def __init__(self, dtype=tf.float32, num_params=1024):
       super().__init__(dtype=dtype, name='custom_transformer_custom_layer',
                            activation='tanh', num_actions=42)
       self.params = tf.keras.layers.MovingAverage(num_params)

   def forward(self, input_tensor):
       return self.self_apply_gradient(input_tensor, num_params) + input_tensor
```

Please note that this is a simplified example and this class may not perform optimizations like others.


```py
class EncoderLayer(Module):
    """A layer unit in encoder. Combines multihead self attention and feed forward."""

    def __init__(
        self,
        attention: Optional[Module],    # can be None if the entire layer is pruned
        dropout: float,
        layer_norm_first: bool,
        feed_forward: Optional[Module], # can be None if the entire layer is pruned
        embed_dim: int,
    ):
        super().__init__()
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.layer_norm_first = layer_norm_first
        self.feed_forward = feed_forward
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): Input of shape ``(batch, sequence_length, embed_dim)``.
            attention_mask (Tensor or ``None``, optional): attention mask
                of shape ``(batch, 1, sequence_length, sequence_length)``. (Default: ``None``)
            position_bias (Tensor or ``None``, optional): position bias of shape
                ``(batch_size * num_heads, src_len, src_len)``.
                Only necessary for WavLM model, ``None`` otherwise. (Default: ``None``)
            key_padding_mask (Tensor or ``None``, optional): key padding mask of shape ``(batch_size, src_len)``.
                Only used for WavLM model, ignored otherwise. (Default: ``None``)
        Returns:
            (x, position_bias): Shapes are the same as in the input. Position bias is only relevant for WaLM model,
                ``None`` otherwise.
        """
        if self.attention is not None:
            residual = x

            if self.layer_norm_first:
                x = self.layer_norm(x)

            x, position_bias = self.attention(
                x, attention_mask=attention_mask, position_bias=position_bias, key_padding_mask=key_padding_mask
            )

            x = self.dropout(x)
            x = residual + x

        if self.layer_norm_first:
            if self.feed_forward is not None:
                x = x + self.feed_forward(self.final_layer_norm(x))
        else:
            # NOTE: for post norm, the layer norms should always be applied even if the layers are pruned.
            x = self.layer_norm(x)
            if self.feed_forward is not None:
                x = x + self.feed_forward(x)
            x = self.final_layer_norm(x)
        return x, position_bias

    def get_num_params(self):
        num_params = self.embed_dim * 2 * 2     # two layer norms
        if self.attention is not None:
            num_params += self.attention.get_num_params()
        if self.feed_forward is not None:
            num_params += self.feed_forward.get_num_params()
        return num_params


```

This is a PyTorch implementation of a Multi-Head self-attention model. It takes in a list of tensors `x` and a list of integer layers `layers`. The layers are expected to be either fully connected or convolutional neural networks (CNNs), and should have a `forward` method that takes in `x` and returns an output tensor.

The `MultiHeadAttention` class inherits from the `torch.nn.Multihead self-attention` class and extends it to handle multiple layers. The `__init__` method sets the number of layers to `num_layers` and the use of attention mechanisms in each layer.

The `forward` method of each layer is implemented to handle the computation of the forward pass of each layer. It first performs position-wise normalization of the input tensor `x`, and then applies the multi-head self-attention mechanism to the input tensor.

The `get_num_params` method returns the total number of parameters of the model, which is the sum of all the parameters of each layer.

The `prune` method removes the self-attention mechanism from all layers and reduces the number of parameters by half.

Note that this implementation is just one possible way to implement a multi-head self-attention model, and it may not be the only way to achieve good performance.


```py
class Transformer(Module):
    def __init__(
        self,
        pos_conv_embed: Module,
        dropout: float,
        layers: Module,
        layer_norm_first: bool,
        layer_drop: float,
    ):
        super().__init__()
        self.pos_conv_embed = pos_conv_embed
        self.layer_norm = nn.LayerNorm(pos_conv_embed.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.layer_drop = layer_drop
        self.dropout = nn.Dropout(dropout)
        self.layers = layers

    def _preprocess(self, x: Tensor):
        x = x + self.pos_conv_embed(x)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        x = self.dropout(x)
        return x

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
    ) -> Tensor:
        x = self._preprocess(x)
        for layer in self.layers:
            if not (self.training and torch.rand(1).item() <= self.layer_drop):
                x, position_bias = layer(x, attention_mask, position_bias=position_bias)

        if not self.layer_norm_first:
            x = self.layer_norm(x)
        return x

    def get_intermediate_outputs(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
        position_bias: Optional[Tensor] = None,
    ) -> List[Tensor]:
        if num_layers is not None:
            if not 0 < num_layers <= len(self.layers):
                raise ValueError(f"`num_layers` must be between [1, {len(self.layers)}]")

        ret: List[Tensor] = []
        x = self._preprocess(x)
        for layer in self.layers:
            x, position_bias = layer(x, attention_mask, position_bias=position_bias)
            ret.append(x)
            if num_layers is not None and len(ret) >= num_layers:
                return ret
        return ret
    
    def get_num_params(self):
        # pos_conv_embed and layer_norm
        num_params = sum(p.numel() for p in self.pos_conv_embed.parameters()) + self.pos_conv_embed.embed_dim * 2
        for layer in self.layers:
            num_params += layer.get_num_params()
        return num_params
    
    def prune(self):
        new_config = defaultdict(list)
        for layer in self.layers:
            attention_config = layer.attention.prune()
            new_config["use_attention"].append(attention_config["use_attention"])
            if "remaining_heads" in attention_config:
                new_config["remaining_heads"].append(attention_config["remaining_heads"])
            else:
                new_config["num_heads"].append(attention_config["num_heads"])

            if not attention_config["use_attention"]:
                layer.attention = None
            
            ff_config = layer.feed_forward.prune()
            new_config["use_feed_forward"].append(ff_config["use_feed_forward"])
            new_config["ff_interm_features"].append(ff_config["ff_interm_features"])
            if not ff_config["use_feed_forward"]:
                layer.feed_forward = None
        
        return new_config


```

This is a PyTorch implementation of a simple neural network model for classification
and translation tasks.

This model consists of a preprocessing step, a feature projection layer, a transformer layer, and a
"last" (or "final") layer with a linear activation function.
The preprocessing step includes padded elements and zero-out them.
The feature projection layer has a linear activation function for the attention input.
The transformer layer has multiple layers of self-attention and feed-forward neural networks
to extract the input features and apply the attention mechanism.
The `get_num_params` function is used to calculate the number of parameters for the model.
The `prune` function is used to remove sub-modules from the model.

This model can be run on a neural network device, such as a GPU, and can be trained
with various optimization algorithms, such as Stochastic Gradient Descent (SGD)
and Adam.


```py
class Encoder(Module):
    def __init__(
        self,
        feature_projection: Module,
        transformer: Module,
    ):
        super().__init__()
        self.feature_projection = feature_projection
        self.transformer = transformer

    def _preprocess(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x = self.feature_projection(features)

        mask: Optional[Tensor] = None
        if lengths is not None:
            batch_size, max_len, _ = x.shape
            # create mask for padded elements and zero-out them
            mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
            x[mask] = 0.0
            # extend the mask to attention shape and set weight
            mask = -10000.0 * mask[:, None, None, :].to(dtype=features.dtype)
            mask = mask.expand(batch_size, 1, max_len, max_len)
        return x, mask

    def forward(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        x, mask = self._preprocess(features, lengths)
        x = self.transformer(x, attention_mask=mask)
        return x

    def extract_features(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> List[Tensor]:
        x, masks = self._preprocess(features, lengths)
        interm = self.transformer.get_intermediate_outputs(x, attention_mask=masks, num_layers=num_layers)
        return [x] + interm
    
    def get_num_params(self, in_features):
        """Calculate the current model size."""
        feature_projection_size = self.feature_projection.get_num_params(in_features)
        transformer_size = self.transformer.get_num_params()
        return feature_projection_size + transformer_size
    
    def prune(self, conv_out_index):
        """In-place pruning of submodules."""
        prune_layer_norm(self.feature_projection.layer_norm, conv_out_index)
        prune_linear_layer(self.feature_projection.projection, conv_out_index, "input")
        transformer_config = self.transformer.prune()
        return transformer_config


```

This is a Python implementation of the Wav2Vec2 model, which is a pre-trained neural network model for speech synthesis. The model consists of a sequence of convolutional layers followed by a池化层和露丝层，用于提取输入数据中的特征。

The Wav2Vec2 model is trained to predict a fixed-length output，with a focus on the quality of the output，即使 the input is low-quality。This implementation Use美丽小这个库，对句柄到数组的转化是把各个卷积层在第一个参数中露丝，第二个参数为1。


```py
################################################################################
def _get_feature_extractor(
    norm_mode: str,
    shapes: List[Tuple[int, int, int]],
    bias: bool,
    prune_conv_channels: bool = False,
) -> FeatureExtractor:
    """
    Args:
        norm_mode (str):
            Either "group_norm" or "layer_norm".
            If "group_norm", then a single normalization is applied
            in the first convolution block. Otherwise, all the convolution
            blocks will have layer normalization.
            This option corresponds to "extractor_mode" from fairseq.
            Expected values are "group_norm" for Base arch, and
            "layer_norm" for Large arch.
        shapes (list of tuple of int):
            Configuration of convolution layers. List of convolution configuration,
            i.e. ``[(output_channel, kernel_size, stride), ...]``
            This option corresponds to "conv_feature_layers" from fairseq.
            Expected values are
            ``[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2``
            for all the architectures.
        bias (bool):
            Whether to include bias term to each convolution operation.
            This option corresponds to "conv_bias" from fairseq.
            Expected values are False for Base arch, and True for Large arch.

    See Also:
        * Original implementation
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L666-L733
        * "extractor_mode"
          - Def and base:
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L38-L45
          - Large:
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L52
        * "conv_feature_layers"
          - Def, base and large:
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L94-L100
        * "conv_bias"
          - Def and base:
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L101-L103
          - Large:
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L61
    """
    if norm_mode not in ["group_norm", "layer_norm"]:
        raise ValueError("Invalid norm mode")
    blocks = []
    in_channels = 1
    for i, (out_channels, kernel_size, stride) in enumerate(shapes):
        normalization = None
        if norm_mode == "group_norm" and i == 0:
            normalization = nn.GroupNorm(
                num_groups=out_channels,
                num_channels=out_channels,
                affine=True,
            )
        elif norm_mode == "layer_norm":
            normalization = LayerNorm(
                normalized_shape=out_channels,
                elementwise_affine=True,
            )
        blocks.append(
            ConvLayerBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                layer_norm=normalization,
                prune_conv_channels=prune_conv_channels,
            )
        )
        in_channels = out_channels
    return FeatureExtractor(nn.ModuleList(blocks))


```

This is a PyTorch implementation of a neural network model for speech synthesis tasks. The model takes an input audio signal and synthesizes a corresponding text. The text is generated by a pre-trained transformer model, which has been fine-tuned on a large corpus of text-audio pairs.

The model has several components: the input encoder, the attention mechanism, and the transformer model.

The input encoder receives the input audio signal and adds a positional encoding. The positional encoding is a learnable vector that adds context to the input audio.

The attention mechanism is used to determine which parts of the input audio to pay attention to when generating the text. This is done by a self-attention mechanism that takes the input audio as input and outputs a set of attention weights. These weights are then used to compute a weighted sum of the input audio, which is then passed through a feed-forward neural network to generate the text.

The transformer model is the core of the network and takes the output of the input encoder as input. It has multiple layers of encoder and decoder blocks with different functionalities.

The layers of the transformer model are:

1. The encoder-decoder connection, which computes the context by taking the output of the input encoder and concatenating it with the attention weights.
2. Multi-head self-attention module, which computes the attention weights using the input audio.
3. Dropout and normalization layers.
4. The feed-forward neural network, which takes the output of the self-attention module and generates the output.
5.层归一化层， which normalize the output of the feed-forward neural network.
6.具有微调，用于在不同的数据集上训练.

这个模型在某些方面使用了 fairseq 库，该库提供了方便的预训练模型。此外，它还使用了注意力机制和位置编码机制，以便在生成文本时考虑音频的上下文信息。


```py
def _get_encoder(
    in_features: int,
    embed_dim: int,
    dropout_input: float,
    pos_conv_kernel: int,
    pos_conv_groups: int,
    num_layers: int,
    use_attention: List[bool],
    use_feed_forward: List[bool],
    num_heads: List[int],
    head_dim: int,
    attention_dropout: float,
    ff_interm_features: List[int],
    ff_interm_dropout: float,
    dropout: float,
    layer_norm_first: bool,
    layer_drop: float,
    prune_attention_heads: bool = False,
    prune_attention_layer: bool = False,
    prune_feed_forward_intermediate: bool = False,
    prune_feed_forward_layer: bool = False,
) -> Encoder:
    """
    Args:
        in_features (int): The number of input features.
        embed_dim (int):
            The dimension of embedding.
            This option corresponds to "encoder_embed_dim" from fairseq.
            Expected values are 768 for Base arch, and 1024 for Large arch.
        dropout_input (float):
            The dropout probability applied after the input feature is projected
            to ``embed_dim``.
            This option corresponds to "dropout_input" from fairseq.
            Expected values are 0.1 for both Base and Large arch.
        pos_conv_kernel (int):
            The kernel size of convolutional positional embeddings.
            This option corresponds to "conv_pos" from fairseq.
            Expected values are 128 for both Base and Large arch.
        pos_conv_groups (int):
            The number of groups of convolutional positional embeddings.
            This option corresponds to "conv_pos_groups" from fairseq.
            Expected values are 16 for both Base and Large arch.
        num_layers (int):
            The number of self attention layers in transformer block.
            This option corresponds to "encoder_layers" from fairseq.
            Expected values are 12 for Base and 24 for Large arch.
        num_heads (int):
            The number of heads in self attention layers.
            This option corresponds to "encoder_attention_heads" from fairseq.
            Expected values are 12 for Base and 16 for Large arch.
        attention_dropout (float):
            The dropout probability applied after softmax in self-attention layer.
            This option corresponds to "attention_dropout" from fairseq.
            Expected values are 0.1 for Base and 0.0 for Large arch.
        ff_interm_features (int):
            The dimension of hidden features in feed forward layer.
            This option corresponds to "encoder_ffn_embed_dim" from fairseq.
            Expected values are 3072 for Base and 4096 for Large arch.
        ff_interm_dropout (float):
            The dropout probability applied in feedforward layer.
            This option correspinds to "activation_dropout" from fairseq.
            Expected values are 0.1 for both Base and Large arch.
        dropout (float):
            The dropout probability applied at the end of feed forward layer.
            This option corresponds to "dropout" from fairseq.
            Expected values are 0.1 for Base and 0.0 for Large arch.
        layer_norm_first (bool):
            Control the order of layer norm in transformer layer and each encoder layer.
            If True, in transformer layer, layer norm is applied before features are fed
            to encoder layers. In encoder layer, two layer norms are applied before and after
            self attention.
            If False, in transformer layer, layer norm is applied after features are fed
            to encoder layers. In encoder layer, two layer norms are applied after self
            attention, before and after feed forward.
            This option corresponds to "layer_norm_first" from fairseq.
            Expected values are False for Base and True for Large arch.
        layer_drop (float):
            Probability to drop each encoder layer during training.
            This option corresponds to "layerdrop" from fairseq.
            Expected values are 0.1 for both Base and Large arch.

    See Also:
        * "encoder_embed_dim"
          - Def and base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L49-L51
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L64
        * "dropout_input"
          - Def, base and large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L75-L78
        * "conv_pos"
          - Def, base and large
            NOTE: The description is wrong.
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L204-L207
          - Usage
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L756
        * "conv_pos_groups"
          - Def, base and large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L208-L211
        * "encoder_layers"
          - Def and base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L46-L48
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L63
        * "encoder_attention_heads"
          - Def and base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L55-L57
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L66
        * "attention_dropout"
          - Def and base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L66-L68
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L60
        * "encoder_ffn_embed_dim"
          - Def and base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L52-L54
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L65
        * "activation_dropout"
          - Def
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L69-L71
          - Base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/finetuning/base_960h.yaml#L55
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/finetuning/vox_960h.yaml#L55
        * "dropout"
          - Def and base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L63-L65
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L59
        * "layer_norm_first"
          - Def and base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L91-L93
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L53
        * "layerdrop"
          - Def
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L72-L74
          - Base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/finetuning/base_960h.yaml#L54
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/finetuning/vox_960h.yaml#L54
    """
    feature_projection = FeatureProjection(in_features, embed_dim, dropout_input)
    pos_conv = ConvolutionalPositionalEmbedding(embed_dim, pos_conv_kernel, pos_conv_groups)

    # Original impl
    # https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L768-L782
    encoder_layers = nn.ModuleList()
    for idx in range(num_layers):
        if use_attention[idx]:
            attention = SelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads[idx],
                head_dim=head_dim,
                dropout=attention_dropout,
                prune_heads=prune_attention_heads,
                prune_layer=prune_attention_layer,
            )
        else:
            attention = None
        if use_feed_forward[idx]:
            feed_forward = FeedForward(
                io_features=embed_dim,
                intermediate_features=ff_interm_features[idx],
                intermediate_dropout=ff_interm_dropout,
                output_dropout=dropout,
                prune_intermediate=prune_feed_forward_intermediate,
                prune_layer=prune_feed_forward_layer,
            )
        else:
            feed_forward = None
        encoder_layers.append(
            EncoderLayer(
                attention=attention,
                dropout=dropout,
                layer_norm_first=layer_norm_first,
                feed_forward=feed_forward,
                embed_dim=embed_dim,
            )
        )
    transformer = Transformer(
        pos_conv_embed=pos_conv,
        dropout=dropout,
        layers=encoder_layers,
        layer_norm_first=not layer_norm_first,
        layer_drop=layer_drop,
    )
    return Encoder(feature_projection, transformer)


```

This is a Python implementation of a Wav2Vec model based on the Transformer architecture. It uses a combination of self-attention and feed-forward mechanisms for encoding the input features and produces a pre-trained transcription representation. The model takes an input sequence of length `seq_len` and an encoder_key of length `key_dim`, and encodes it into a fixed-length output sequence of length `seq_len`.

The self-attention mechanism is applied multiple times in parallel, with each layer of the encoder having an attention dropout of 0.1 and a maximum distance of 0.2. The attention is calculated based on the input sequence and the encoder key using the WavLMSelfAttention class.

The feed-forward layers are applied after the self-attention mechanism. They have an intermediate feature size of `ff_interm_features` and an output dropout of 0.1. The layers are initialized with the input features and the encoder key, and the intermediate outputs from each self-attention layer are concatenated with the corresponding attention output to form the input to the feed-forward layer.

The Wav2Vec model is trained end-to-end by minimizing the cross-entropy loss between the input and output sequences. The loss function is defined as the negative log-likelihood ratio of the input to output, where the log-likelihood is calculated using the softmax activation of the output probabilities.

The output of the model is the pre-trained transcription representation, which can be used for further processing or as input to other layers.


```py
def _get_wavlm_encoder(
    in_features: int,
    embed_dim: int,
    dropout_input: float,
    pos_conv_kernel: int,
    pos_conv_groups: int,
    num_layers: int,
    use_attention: List[bool],
    use_feed_forward: List[bool],
    total_num_heads: List[int],
    remaining_heads: List[List[int]],
    num_buckets: int,
    max_distance: int,
    attention_dropout: float,
    ff_interm_features: List[int],
    ff_interm_dropout: float,
    dropout: float,
    layer_norm_first: bool,
    layer_drop: float,
    prune_attention_heads: bool = False,
    prune_attention_layer: bool = False,
    prune_feed_forward_intermediate: bool = False,
    prune_feed_forward_layer: bool = False,
) -> Encoder:
    """
    Construct encoder for WavLM model :cite:`chen2022wavlm`. The structure of the encoder and most of the argments are
    the same as in :py:func:`_get_encoder` so refer there for documentation. The only difference from Wav2Vec2 encoder
    is usage of `WavLMSelfAttention` instead of `SelfAttention` and two additional parameters: `num_buckets` and
    `max_distance`.
    Args:
        in_features (int): See :py:func:`_get_encoder`.
        embed_dim (int): See :py:func:`_get_encoder`.
        dropout_input (float): See :py:func:`_get_encoder`.
        pos_conv_kernel (int): See :py:func:`_get_encoder`.
        pos_conv_groups (int): See :py:func:`_get_encoder`.
        num_layers (int): See :py:func:`_get_encoder`.
        num_heads (int): See :py:func:`_get_encoder`.
        num_buckets (int): Number of buckets for relative position embedding.
        max_distance (int): Maximum distance for relative position embedding.
        attention_dropout (float): See :py:func:`_get_encoder`.
        ff_interm_features (int): See :py:func:`_get_encoder`.
        ff_interm_dropout (float): See :py:func:`_get_encoder`.
        dropout (float): See :py:func:`_get_encoder`.
        layer_norm_first (bool): See :py:func:`_get_encoder`.
        layer_drop (float): See :py:func:`_get_encoder`.

    """
    feature_projection = FeatureProjection(in_features, embed_dim, dropout_input)
    pos_conv = ConvolutionalPositionalEmbedding(embed_dim, pos_conv_kernel, pos_conv_groups)

    # Original impl
    # https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L768-L782
    encoder_layers = nn.ModuleList()
    for i in range(num_layers):
        if use_attention[i]:
            attention = WavLMSelfAttention(
                embed_dim=embed_dim,
                total_num_heads=total_num_heads[i],
                remaining_heads=remaining_heads[i],
                dropout=attention_dropout,
                has_relative_attention_bias=(i == 0),  # Position embedding is only necessary in the first layer.
                num_buckets=num_buckets,
                max_distance=max_distance,
                prune_heads=prune_attention_heads,
                prune_layer=prune_attention_layer,
            )
        else:
            attention = None
        if use_feed_forward[i]:
            feed_forward = FeedForward(
                io_features=embed_dim,
                intermediate_features=ff_interm_features[i],
                intermediate_dropout=ff_interm_dropout,
                output_dropout=dropout,
                prune_intermediate=prune_feed_forward_intermediate,
                prune_layer=prune_feed_forward_layer,
            )
        else:
            feed_forward = None
        encoder_layers.append(
            EncoderLayer(
                attention=attention,
                dropout=dropout,
                layer_norm_first=layer_norm_first,
                feed_forward=feed_forward,
                embed_dim=embed_dim,
            )
        )
    transformer = Transformer(
        pos_conv_embed=pos_conv,
        dropout=dropout,
        layers=encoder_layers,
        layer_norm_first=not layer_norm_first,
        layer_drop=layer_drop,
    )
    return Encoder(feature_projection, transformer)


```

这段代码定义了一个名为 `_get_padding_mask` 的函数，以及一个名为 `GradMultiply` 的类。

函数 `_get_padding_mask` 接收两个输入张量 `input` 和 `lengths`，并返回一个输出张量 `mask`。函数的作用是生成一个与输入张量相同大小，但只能在指定长度的子张量上为 1 的张量，用于对输入张量进行填充以符合特定的数据长度。

类 `GradMultiply` 是一个实现了 `torch.autograd.Function` 的类，它接收一个输入张量 `x` 和一个缩放因子 `scale`，并返回一个输出张量 `res`。函数的作用是在输入张量上执行一个名为 `forward` 的方法，该方法按照输入张量的放大因子对输入张量进行平滑，并返回输出张量 `res`。函数接收一个名为 `backward` 的方法，该方法按照输入张量的放大因子对输入张量的梯度进行计算，并返回输出张量 `grad`。


```py
def _get_padding_mask(input: Tensor, lengths: Tensor) -> Tensor:
    """Generate the padding mask given the padded input and the lengths Tensors.
    Args:
        input (Tensor): The padded Tensor of dimension `[batch, max_len, frequency]`.
        lengths (Tensor): The lengths Tensor of dimension `[batch,]`.

    Returns:
        (Tensor): The padding mask.
    """
    batch_size, max_len, _ = input.shape
    mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
    return mask


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

```