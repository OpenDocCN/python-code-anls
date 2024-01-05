# `d:/src/tocomm/Bert-VITS2\data_utils.py`

```
# 导入所需的模块
import os  # 提供与操作系统相关的功能
import random  # 提供生成随机数的功能
import torch  # 提供深度学习框架的功能
import torch.utils.data  # 提供用于加载数据的工具
from tqdm import tqdm  # 提供进度条功能
from tools.log import logger  # 导入自定义的日志模块
import commons  # 导入自定义的通用函数模块
from mel_processing import spectrogram_torch, mel_spectrogram_torch  # 导入自定义的声谱图处理模块
from utils import load_wav_to_torch, load_filepaths_and_text  # 导入自定义的工具函数
from text import cleaned_text_to_sequence  # 导入自定义的文本处理模块
from config import config  # 导入自定义的配置文件

"""Multi speaker version"""


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """
    # 这个类是一个数据集类，用于加载音频、说话人ID和文本对
    # 它的功能包括：加载音频、说话人ID和文本对，将文本标准化并转换为整数序列，从音频文件计算声谱图。
    """

    def __init__(self, audiopaths_sid_text, hparams):
        # 将音频路径、说话人ID和文本信息加载为一个列表
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        # 设置音频的最大值
        self.max_wav_value = hparams.max_wav_value
        # 设置采样率
        self.sampling_rate = hparams.sampling_rate
        # 设置滤波器长度
        self.filter_length = hparams.filter_length
        # 设置跳跃长度
        self.hop_length = hparams.hop_length
        # 设置窗口长度
        self.win_length = hparams.win_length
        # 设置采样率
        self.sampling_rate = hparams.sampling_rate
        # 设置说话人映射
        self.spk_map = hparams.spk2id
        # 设置超参数
        self.hparams = hparams

        # 检查是否使用后验梅尔频谱编码器
        self.use_mel_spec_posterior = getattr(
            hparams, "use_mel_posterior_encoder", False
        )
        # 如果使用后验梅尔频谱编码器，则设置梅尔频道数
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)

        # 检查是否清理文本
        self.cleaned_text = getattr(hparams, "cleaned_text", False)
        self.add_blank = hparams.add_blank
```
这行代码将hparams中的add_blank属性赋值给self对象的add_blank属性。

```
        self.min_text_len = getattr(hparams, "min_text_len", 1)
```
这行代码将hparams中的min_text_len属性赋值给self对象的min_text_len属性，如果hparams中没有min_text_len属性，则默认值为1。

```
        self.max_text_len = getattr(hparams, "max_text_len", 384)
```
这行代码将hparams中的max_text_len属性赋值给self对象的max_text_len属性，如果hparams中没有max_text_len属性，则默认值为384。

```
        random.seed(1234)
```
这行代码设置随机数种子为1234，用于生成随机数。

```
        random.shuffle(self.audiopaths_sid_text)
```
这行代码对self对象的audiopaths_sid_text属性进行随机打乱。

```
        self._filter()
```
这行代码调用self对象的_filter方法。

```
    def _filter(self):
```
这行代码定义了一个名为_filter的方法，该方法属于self对象。

```
        """
        Filter text & store spec lengths
        """
```
这是_filter方法的文档字符串，用于解释方法的作用。

```
        audiopaths_sid_text_new = []
        lengths = []
        skipped = 0
```
这几行代码定义了三个变量，分别是audiopaths_sid_text_new、lengths和skipped，并初始化为空列表和0。
        logger.info("Init dataset...")
```
这行代码用于记录日志，输出"Init dataset..."。

```
        for _id, spk, language, text, phones, tone, word2ph in tqdm(
            self.audiopaths_sid_text
        ):
```
这是一个for循环，用于遍历`self.audiopaths_sid_text`中的元素，并将每个元素的值分别赋给`_id, spk, language, text, phones, tone, word2ph`这些变量。

```
            audiopath = f"{_id}"
```
这行代码将`_id`转换为字符串，并将结果赋给`audiopath`变量。

```
            if self.min_text_len <= len(phones) and len(phones) <= self.max_text_len:
```
这是一个条件语句，判断`phones`的长度是否在`self.min_text_len`和`self.max_text_len`之间。

```
                phones = phones.split(" ")
                tone = [int(i) for i in tone.split(" ")]
                word2ph = [int(i) for i in word2ph.split(" ")]
```
这几行代码将`phones`、`tone`和`word2ph`分别按空格进行分割，并将结果转换为列表。

```
                audiopaths_sid_text_new.append(
                    [audiopath, spk, language, text, phones, tone, word2ph]
                )
```
这行代码将`audiopath, spk, language, text, phones, tone, word2ph`这些变量组成一个列表，并将该列表添加到`audiopaths_sid_text_new`列表中。

```
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
```
这行代码获取`audiopath`文件的大小，并将其除以`(2 * self.hop_length)`的结果添加到`lengths`列表中。

```
            else:
                skipped += 1
```
如果条件`self.min_text_len <= len(phones) and len(phones) <= self.max_text_len`不满足，则将`skipped`变量加1。

```
        logger.info(
            "skipped: "
            + str(skipped)
            + ", total: "
            + str(len(self.audiopaths_sid_text))
```
这行代码用于记录日志，输出"skipped: "、`skipped`的值、", total: "和`self.audiopaths_sid_text`的长度。
def get_audio_text_speaker_pair(self, audiopath_sid_text):
    # separate filename, speaker_id and text
    audiopath, sid, language, text, phones, tone, word2ph = audiopath_sid_text
    # 调用get_text函数，获取文本的特征向量
    bert, ja_bert, en_bert, phones, tone, language = self.get_text(
        text, word2ph, phones, tone, language, audiopath
    )
    # 调用get_audio函数，获取音频的特征向量和波形数据
    spec, wav = self.get_audio(audiopath)
    # 将speaker_id转换为LongTensor类型
    sid = torch.LongTensor([int(self.spk_map[sid])])
    # 返回音频、文本和说话人的特征向量以及波形数据
    return (phones, spec, wav, sid, tone, language, bert, ja_bert, en_bert)

def get_audio(self, filename):
    # 调用load_wav_to_torch函数，加载音频文件并转换为torch.Tensor类型
    audio, sampling_rate = load_wav_to_torch(filename)
    # 如果音频的采样率与预设的采样率不一致
    if sampling_rate != self.sampling_rate:
# 如果文件名的采样率与目标采样率不匹配，则抛出 ValueError 异常
raise ValueError(
    "{} {} SR doesn't match target {} SR".format(
        filename, sampling_rate, self.sampling_rate
    )
)

# 将音频数据归一化到 [-1, 1] 的范围
audio_norm = audio / self.max_wav_value

# 将音频数据转换为一维张量
audio_norm = audio_norm.unsqueeze(0)

# 将文件名中的 ".wav" 替换为 ".spec.pt"，得到对应的频谱文件名
spec_filename = filename.replace(".wav", ".spec.pt")

# 如果使用梅尔频谱后验，则将文件名中的 ".spec.pt" 替换为 ".mel.pt"
if self.use_mel_spec_posterior:
    spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")

# 尝试从文件中加载频谱数据
try:
    spec = torch.load(spec_filename)
# 如果加载失败，则根据音频数据计算频谱
except:
    if self.use_mel_spec_posterior:
        spec = mel_spectrogram_torch(
            audio_norm,
            self.filter_length,
            self.n_mel_channels,
            self.sampling_rate,
            self.hop_length,
            ...
def get_text(self, text, word2ph, phone, tone, language_str, wav_path):
    # 获取文本信息、单词到音素的映射、音素信息、音调信息、语言信息和音频文件路径
    # 这个函数用于获取文本相关的参数，以便后续处理
phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)
```
将`phone`、`tone`和`language_str`作为参数传递给`cleaned_text_to_sequence`函数，并将返回的结果分别赋值给`phone`、`tone`和`language`。

```
if self.add_blank:
    phone = commons.intersperse(phone, 0)
    tone = commons.intersperse(tone, 0)
    language = commons.intersperse(language, 0)
    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1
```
如果`self.add_blank`为真，则执行以下操作：
- 使用`commons.intersperse`函数在`phone`中插入0，将结果重新赋值给`phone`。
- 使用`commons.intersperse`函数在`tone`中插入0，将结果重新赋值给`tone`。
- 使用`commons.intersperse`函数在`language`中插入0，将结果重新赋值给`language`。
- 遍历`word2ph`的索引，将每个元素乘以2。
- 将`word2ph`的第一个元素加1。

```
bert_path = wav_path.replace(".wav", ".bert.pt")
```
将`wav_path`中的".wav"替换为".bert.pt"，并将结果赋值给`bert_path`。

```
try:
    bert_ori = torch.load(bert_path)
    assert bert_ori.shape[-1] == len(phone)
except Exception as e:
    logger.warning("Bert load Failed")
    logger.warning(e)
```
尝试加载`bert_path`指定的文件，并将结果赋值给`bert_ori`。如果加载失败或`bert_ori`的最后一个维度的长度不等于`phone`的长度，则抛出异常并记录警告信息。

```
if language_str == "ZH":
    bert = bert_ori
    ja_bert = torch.randn(1024, len(phone))
    en_bert = torch.randn(1024, len(phone))
```
如果`language_str`等于"ZH"，则执行以下操作：
- 将`bert_ori`赋值给`bert`。
- 使用`torch.randn`函数生成一个大小为(1024, len(phone))的张量，并将结果赋值给`ja_bert`。
- 使用`torch.randn`函数生成一个大小为(1024, len(phone))的张量，并将结果赋值给`en_bert`。
        elif language_str == "JP":
            bert = torch.randn(1024, len(phone))
            ja_bert = bert_ori
            en_bert = torch.randn(1024, len(phone))
```
如果`language_str`等于"JP"，则执行以下操作：
- 使用`torch.randn(1024, len(phone))`生成一个大小为1024x`phone`长度的随机张量，并将其赋值给变量`bert`。
- 将变量`bert_ori`的值赋给变量`ja_bert`。
- 使用`torch.randn(1024, len(phone))`生成一个大小为1024x`phone`长度的随机张量，并将其赋值给变量`en_bert`。

```
        elif language_str == "EN":
            bert = torch.randn(1024, len(phone))
            ja_bert = torch.randn(1024, len(phone))
            en_bert = bert_ori
```
如果`language_str`等于"EN"，则执行以下操作：
- 使用`torch.randn(1024, len(phone))`生成一个大小为1024x`phone`长度的随机张量，并将其赋值给变量`bert`。
- 使用`torch.randn(1024, len(phone))`生成一个大小为1024x`phone`长度的随机张量，并将其赋值给变量`ja_bert`。
- 将变量`bert_ori`的值赋给变量`en_bert`。

```
        phone = torch.LongTensor(phone)
        tone = torch.LongTensor(tone)
        language = torch.LongTensor(language)
```
将`phone`、`tone`和`language`转换为`torch.LongTensor`类型，并分别赋值给变量`phone`、`tone`和`language`。

```
        return bert, ja_bert, en_bert, phone, tone, language
```
返回变量`bert`、`ja_bert`、`en_bert`、`phone`、`tone`和`language`。

```
    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid
```
定义一个名为`get_sid`的方法，该方法接受一个参数`sid`。将`sid`转换为`torch.LongTensor`类型，并返回。

```
    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])
```
定义一个名为`__getitem__`的方法，该方法接受一个参数`index`。通过索引`index`获取`audiopaths_sid_text`列表中的元素，并将其作为参数传递给`get_audio_text_speaker_pair`方法，并返回其结果。
    def __len__(self):
        return len(self.audiopaths_sid_text)
```
这是一个类方法，用于返回`self.audiopaths_sid_text`的长度。

```
class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids
```
这是一个类`TextAudioSpeakerCollate`的构造函数，用于初始化类的实例。`return_ids`是一个可选参数，默认为`False`。

```
    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        )
```
这是一个类方法，用于将训练批次`batch`中的文本、音频和说话人身份进行整理。`batch`是一个包含`text_normalized`、`spec_normalized`、`wav_normalized`和`sid`的列表。在这个方法中，将对文本序列进行右侧零填充，使其长度与最大输入长度相同。然后，使用`torch.sort`函数对填充后的文本序列长度进行降序排序，并返回排序后的结果。
# 计算批次中文本的最大长度
max_text_len = max([len(x[0]) for x in batch])
# 计算批次中频谱的最大长度
max_spec_len = max([x[1].size(1) for x in batch])
# 计算批次中音频的最大长度
max_wav_len = max([x[2].size(1) for x in batch])

# 创建存储文本长度的张量
text_lengths = torch.LongTensor(len(batch))
# 创建存储频谱长度的张量
spec_lengths = torch.LongTensor(len(batch))
# 创建存储音频长度的张量
wav_lengths = torch.LongTensor(len(batch))
# 创建存储样本ID的张量
sid = torch.LongTensor(len(batch))

# 创建存储填充后文本的张量
text_padded = torch.LongTensor(len(batch), max_text_len)
# 创建存储填充后音调的张量
tone_padded = torch.LongTensor(len(batch), max_text_len)
# 创建存储填充后语言的张量
language_padded = torch.LongTensor(len(batch), max_text_len)
# 创建存储填充后BERT特征的张量
bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
# 创建存储填充后日语BERT特征的张量
ja_bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
# 创建存储填充后英语BERT特征的张量
en_bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)

# 创建存储填充后频谱的张量
spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
# 创建存储填充后音频的张量
wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
# 将填充后的文本张量置零
text_padded.zero_()
        tone_padded.zero_()
        # 将 tone_padded 张量的所有元素置零，用于存储音调信息

        language_padded.zero_()
        # 将 language_padded 张量的所有元素置零，用于存储语言信息

        spec_padded.zero_()
        # 将 spec_padded 张量的所有元素置零，用于存储音频频谱信息

        wav_padded.zero_()
        # 将 wav_padded 张量的所有元素置零，用于存储音频数据

        bert_padded.zero_()
        # 将 bert_padded 张量的所有元素置零，用于存储 BERT 编码信息

        ja_bert_padded.zero_()
        # 将 ja_bert_padded 张量的所有元素置零，用于存储日语 BERT 编码信息

        en_bert_padded.zero_()
        # 将 en_bert_padded 张量的所有元素置零，用于存储英语 BERT 编码信息

        for i in range(len(ids_sorted_decreasing)):
            # 遍历 ids_sorted_decreasing 列表的索引

            row = batch[ids_sorted_decreasing[i]]
            # 获取 batch 列表中按照 ids_sorted_decreasing 排序的第 i 个元素

            text = row[0]
            # 获取 row 列表的第一个元素，即文本数据
            text_padded[i, : text.size(0)] = text
            # 将文本数据按照长度进行填充，并存储到 text_padded 张量中
            text_lengths[i] = text.size(0)
            # 记录文本数据的长度

            spec = row[1]
            # 获取 row 列表的第二个元素，即音频频谱数据
            spec_padded[i, :, : spec.size(1)] = spec
            # 将音频频谱数据按照长度进行填充，并存储到 spec_padded 张量中
            spec_lengths[i] = spec.size(1)
            # 记录音频频谱数据的长度

            wav = row[2]
            # 获取 row 列表的第三个元素，即音频数据
# 将wav数据填充到wav_padded数组的第i行，第二维度的长度为wav的第二维度长度
wav_padded[i, :, : wav.size(1)] = wav

# 将wav的第二维度长度赋值给wav_lengths数组的第i个元素
wav_lengths[i] = wav.size(1)

# 将row的第3个元素赋值给sid数组的第i个元素
sid[i] = row[3]

# 将row的第4个元素赋值给tone_padded数组的第i行
tone = row[4]
tone_padded[i, : tone.size(0)] = tone

# 将row的第5个元素赋值给language_padded数组的第i行
language = row[5]
language_padded[i, : language.size(0)] = language

# 将row的第6个元素赋值给bert_padded数组的第i行，第二维度的长度为bert的第二维度长度
bert = row[6]
bert_padded[i, :, : bert.size(1)] = bert

# 将row的第7个元素赋值给ja_bert_padded数组的第i行，第二维度的长度为ja_bert的第二维度长度
ja_bert = row[7]
ja_bert_padded[i, :, : ja_bert.size(1)] = ja_bert

# 将row的第8个元素赋值给en_bert_padded数组的第i行，第二维度的长度为en_bert的第二维度长度
en_bert = row[8]
en_bert_padded[i, :, : en_bert.size(1)] = en_bert
        return (
            text_padded,  # 返回填充后的文本数据
            text_lengths,  # 返回文本数据的长度
            spec_padded,  # 返回填充后的音频数据
            spec_lengths,  # 返回音频数据的长度
            wav_padded,  # 返回填充后的波形数据
            wav_lengths,  # 返回波形数据的长度
            sid,  # 返回说话人ID
            tone_padded,  # 返回填充后的音调数据
            language_padded,  # 返回填充后的语言数据
            bert_padded,  # 返回填充后的BERT数据
            ja_bert_padded,  # 返回填充后的日语BERT数据
            en_bert_padded,  # 返回填充后的英语BERT数据
        )
```
这段代码是一个函数的返回语句，返回了一个元组，包含了多个变量。每个变量都有特定的作用和含义，注释对每个变量进行了解释。

```
class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
```
这段代码定义了一个类`DistributedBucketSampler`，继承自`torch.utils.data.distributed.DistributedSampler`。类的作用是维护一个批次中输入长度相似的样本，长度分组由边界值指定。
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
```
这部分代码是对边界的说明。boundaries是一个列表，其中包含了三个边界值b1、b2和b3。这段注释解释了边界的含义，即任何一个批次的长度要么在b1和b2之间，要么在b2和b3之间。它还解释了这个类的作用，即删除不在边界范围内的样本。

```
    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
```
这是类的构造函数。它接受以下参数：
- dataset: 数据集对象
- batch_size: 批次大小
- boundaries: 边界列表
- num_replicas: 复制数（可选）
- rank: 排名（可选）
- shuffle: 是否打乱数据集顺序（默认为True）

```
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
```
调用父类的构造函数，传递dataset、num_replicas、rank和shuffle参数。

```
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
```
将dataset对象的lengths属性赋值给self.lengths，将batch_size赋值给self.batch_size，将boundaries赋值给self.boundaries。
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
```
创建两个变量`self.buckets`和`self.num_samples_per_bucket`，并将它们的值设为调用`_create_buckets()`方法的返回值。

```
        self.total_size = sum(self.num_samples_per_bucket)
```
计算`self.num_samples_per_bucket`列表中所有元素的和，并将结果赋值给`self.total_size`变量。

```
        self.num_samples = self.total_size // self.num_replicas
```
将`self.total_size`除以`self.num_replicas`的整数部分赋值给`self.num_samples`变量。

```
    def _create_buckets(self):
```
定义一个名为`_create_buckets`的方法。

```
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
```
创建一个列表`buckets`，其中包含`len(self.boundaries) - 1`个空列表。

```
        for i in range(len(self.lengths)):
```
对于`self.lengths`列表中的每个元素，执行以下操作：

```
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
```
将`self.lengths[i]`的值赋给`length`变量，并将调用`_bisect(length)`方法的返回值赋给`idx_bucket`变量。

```
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
```
如果`idx_bucket`不等于-1，则将`i`添加到`buckets[idx_bucket]`列表中。

```
        try:
            for i in range(len(buckets) - 1, 0, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)
            assert all(len(bucket) > 0 for bucket in buckets)
        # When one bucket is not traversed
        except Exception as e:
```
尝试执行以下操作：

- 从`len(buckets) - 1`开始，递减1，直到0为止，对于每个索引`i`：
  - 如果`buckets[i]`列表的长度为0，则将`buckets[i]`和`self.boundaries[i + 1]`从列表中移除。
- 断言`buckets`列表中的每个子列表的长度都大于0。
- 如果出现异常，则将异常对象赋给`e`变量。
            print("Bucket warning ", e)
```
这行代码用于打印一个警告信息，警告信息的内容是 "Bucket warning " 加上变量 e 的值。

```
            for i in range(len(buckets) - 1, -1, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)
```
这段代码用于遍历列表 buckets 中的元素。如果某个元素的长度为 0，则将其从列表中移除，并且将 self.boundaries 列表中对应位置的元素也移除。

```
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
```
这段代码用于计算每个 bucket 中样本的数量，并将结果存储在 num_samples_per_bucket 列表中。首先，获取当前 bucket 的长度 len_bucket。然后，计算总批次大小 total_batch_size，即 self.num_replicas 乘以 self.batch_size。接下来，计算余数 rem，其计算方式为 total_batch_size 减去 len_bucket 对 total_batch_size 取模的结果，再对 total_batch_size 取模。最后，将 len_bucket 和 rem 相加，并将结果添加到 num_samples_per_bucket 列表中。

```
        return buckets, num_samples_per_bucket
```
这行代码用于返回两个值，分别是列表 buckets 和列表 num_samples_per_bucket。

```
    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
```
这段代码定义了一个迭代器的方法 __iter__()。首先，创建了一个 torch.Generator 对象 g。然后，使用 self.epoch 的值作为种子，通过调用 g.manual_seed() 方法设置生成器的种子。这样可以根据 epoch 的值确定性地进行洗牌操作。
        indices = []
        # 如果需要打乱数据，则对每个 bucket 进行随机排列
        if self.shuffle:
            for bucket in self.buckets:
                # 生成一个随机排列的索引列表，并将其添加到 indices 列表中
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                # 生成一个按顺序排列的索引列表，并将其添加到 indices 列表中
                indices.append(list(range(len(bucket))))

        batches = []
        # 遍历每个 bucket
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            # 如果 bucket 为空，则跳过
            if len_bucket == 0:
                continue
            # 获取当前 bucket 对应的索引列表
            ids_bucket = indices[i]
            # 获取当前 bucket 需要采样的样本数量
            num_samples_bucket = self.num_samples_per_bucket[i]

            # 添加额外的样本使其能够被均匀地整除
            rem = num_samples_bucket - len_bucket
```

注释解释了代码的作用，包括对数据进行打乱或按顺序排列，以及添加额外的样本使其能够被均匀地整除。
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )
```
这段代码是将`ids_bucket`进行扩展，以便在进行子采样和分批处理时使用。

```
            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]
```
这段代码是对`ids_bucket`进行子采样，根据`self.rank`和`self.num_replicas`的值来选择子采样的元素。

```
            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)
```
这段代码是将`ids_bucket`按照`self.batch_size`进行分批处理，每个批次的元素存储在`batch`列表中，然后将每个批次添加到`batches`列表中。

```
        if self.shuffle:
```
这段代码是判断是否需要对`batches`列表进行洗牌操作，根据`self.shuffle`的值来决定是否执行洗牌操作。
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
```
这行代码使用`torch.randperm`函数生成一个随机的排列索引，用于对`batches`列表进行洗牌操作。`len(batches)`表示`batches`列表的长度，`generator=g`表示使用指定的随机数生成器`g`。`.tolist()`将生成的随机索引转换为列表形式。

```
            batches = [batches[i] for i in batch_ids]
```
这行代码根据生成的随机索引`batch_ids`重新排列`batches`列表中的元素，以实现洗牌操作。

```
        self.batches = batches
```
这行代码将洗牌后的`batches`列表赋值给类的属性`self.batches`。

```
        assert len(self.batches) * self.batch_size == self.num_samples
```
这行代码使用断言语句来确保洗牌后的`batches`列表中的样本数量与总样本数量`self.num_samples`相匹配。如果不匹配，将会触发一个断言错误。

```
        return iter(self.batches)
```
这行代码将洗牌后的`batches`列表转换为一个迭代器，并返回该迭代器。

```
    def _bisect(self, x, lo=0, hi=None):
```
这是一个私有方法`_bisect`的定义，该方法用于执行二分查找。

```
        if hi is None:
            hi = len(self.boundaries) - 1
```
这行代码用于设置二分查找的上界`hi`。如果未指定上界，则将其设置为`self.boundaries`列表的长度减1。

```
        if hi > lo:
```
这行代码判断是否需要继续进行二分查找。如果上界`hi`大于下界`lo`，则继续进行二分查找。

```
            mid = (hi + lo) // 2
```
这行代码计算二分查找的中间位置`mid`，使用整数除法运算符`//`来确保结果为整数。

```
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
```
这行代码判断目标值`x`是否位于中间位置`mid`和`mid+1`之间的区间内。如果是，则返回中间位置`mid`。

```
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
```
这行代码判断目标值`x`是否小于等于中间位置`mid`的值。如果是，则递归调用`_bisect`方法，在下界`lo`和中间位置`mid`之间继续进行二分查找。

```
            else:
                return self._bisect(x, mid + 1, hi)
```
这行代码表示目标值`x`大于中间位置`mid`的值，因此递归调用`_bisect`方法，在中间位置`mid+1`和上界`hi`之间继续进行二分查找。

```
        else:
```
这行代码表示二分查找已经完成，上界`hi`等于下界`lo`。
# 返回-1，表示读取 ZIP 文件失败
return -1

# 返回数据集的长度，即样本数量除以批次大小
return self.num_samples // self.batch_size
```

这段代码是一个类的方法，包含了两个函数。第一个函数是`__getitem__`，用于获取指定索引位置的样本数据。第二个函数是`__len__`，用于返回数据集的长度。
```