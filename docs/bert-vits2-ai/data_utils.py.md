# `Bert-VITS2\data_utils.py`

```py
# 导入所需的库
import os
import random
import torch
import torch.utils.data
from tqdm import tqdm
from tools.log import logger
import commons
from mel_processing import spectrogram_torch, mel_spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import cleaned_text_to_sequence
from config import config

"""Multi speaker version"""

# 定义一个数据加载类，用于加载音频、说话者ID和文本对
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    # 初始化函数，接受音频路径、说话者ID和文本对以及超参数作为输入
    def __init__(self, audiopaths_sid_text, hparams):
        # 加载音频路径、说话者ID和文本对
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        # 设置最大音频值
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
        # 设置说话者映射
        self.spk_map = hparams.spk2id
        # 设置超参数
        self.hparams = hparams

        # 检查是否使用后验梅尔频谱编码器
        self.use_mel_spec_posterior = getattr(
            hparams, "use_mel_posterior_encoder", False
        )
        if self.use_mel_spec_posterior:
            # 设置梅尔频道数
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)

        # 检查是否清理文本
        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        # 添加空白
        self.add_blank = hparams.add_blank
        # 设置最小文本长度
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        # 设置最大文本长度
        self.max_text_len = getattr(hparams, "max_text_len", 384)

        # 设置随机种子并打乱数据
        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        # 过滤数据
        self._filter()
    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # 存储用于分桶的频谱长度
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        skipped = 0
        logger.info("Init dataset...")
        for _id, spk, language, text, phones, tone, word2ph in tqdm(
            self.audiopaths_sid_text
        ):
            audiopath = f"{_id}"
            if self.min_text_len <= len(phones) and len(phones) <= self.max_text_len:
                phones = phones.split(" ")
                tone = [int(i) for i in tone.split(" ")]
                word2ph = [int(i) for i in word2ph.split(" ")]
                audiopaths_sid_text_new.append(
                    [audiopath, spk, language, text, phones, tone, word2ph]
                )
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
            else:
                skipped += 1
        logger.info(
            "skipped: "
            + str(skipped)
            + ", total: "
            + str(len(self.audiopaths_sid_text))
        )
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # 分离文件名、说话者ID和文本
        audiopath, sid, language, text, phones, tone, word2ph = audiopath_sid_text

        bert, ja_bert, en_bert, phones, tone, language = self.get_text(
            text, word2ph, phones, tone, language, audiopath
        )

        spec, wav = self.get_audio(audiopath)
        sid = torch.LongTensor([int(self.spk_map[sid])])

        return (phones, spec, wav, sid, tone, language, bert, ja_bert, en_bert)
    # 从文件名获取音频数据和采样率
    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        # 如果采样率不匹配目标采样率，则抛出数值错误
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} {} SR doesn't match target {} SR".format(
                    filename, sampling_rate, self.sampling_rate
                )
            )
        # 对音频数据进行归一化处理
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        # 将文件名中的".wav"替换为".spec.pt"作为频谱文件名
        spec_filename = filename.replace(".wav", ".spec.pt")
        # 如果使用梅尔频谱后处理，则将文件名中的".spec.pt"替换为".mel.pt"
        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
        # 尝试加载频谱数据文件
        try:
            spec = torch.load(spec_filename)
        # 如果加载失败，则根据是否使用梅尔频谱后处理进行不同的处理
        except:
            if self.use_mel_spec_posterior:
                spec = mel_spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.n_mel_channels,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    self.hparams.mel_fmin,
                    self.hparams.mel_fmax,
                    center=False,
                )
            else:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
            # 去除频谱数据的批次维度
            spec = torch.squeeze(spec, 0)
            # 如果配置为训练时缓存频谱数据，则保存频谱数据到文件
            if config.train_ms_config.spec_cache:
                torch.save(spec, spec_filename)
        # 返回频谱数据和归一化后的音频数据
        return spec, audio_norm
    # 获取文本信息，将文本转换为序列
    def get_text(self, text, word2ph, phone, tone, language_str, wav_path):
        phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)  # 将文本转换为序列
        # 如果需要添加空白符
        if self.add_blank:
            phone = commons.intersperse(phone, 0)  # 在phone序列中插入0
            tone = commons.intersperse(tone, 0)  # 在tone序列中插入0
            language = commons.intersperse(language, 0)  # 在language序列中插入0
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2  # 将word2ph中的每个元素乘以2
            word2ph[0] += 1  # 将word2ph的第一个元素加1
        bert_path = wav_path.replace(".wav", ".bert.pt")  # 替换wav_path中的".wav"为".bert.pt"
        try:
            bert_ori = torch.load(bert_path)  # 加载bert模型
            assert bert_ori.shape[-1] == len(phone)  # 断言bert_ori的最后一个维度长度等于phone的长度
        except Exception as e:
            logger.warning("Bert load Failed")  # 记录警告信息
            logger.warning(e)  # 记录异常信息

        # 根据语言类型分配bert模型
        if language_str == "ZH":
            bert = bert_ori
            ja_bert = torch.randn(1024, len(phone))
            en_bert = torch.randn(1024, len(phone))
        elif language_str == "JP":
            bert = torch.randn(1024, len(phone))
            ja_bert = bert_ori
            en_bert = torch.randn(1024, len(phone))
        elif language_str == "EN":
            bert = torch.randn(1024, len(phone))
            ja_bert = torch.randn(1024, len(phone))
            en_bert = bert_ori
        phone = torch.LongTensor(phone)  # 将phone转换为LongTensor类型
        tone = torch.LongTensor(tone)  # 将tone转换为LongTensor类型
        language = torch.LongTensor(language)  # 将language转换为LongTensor类型
        return bert, ja_bert, en_bert, phone, tone, language  # 返回bert, ja_bert, en_bert, phone, tone, language

    # 获取sid信息
    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])  # 将sid转换为LongTensor类型
        return sid  # 返回sid

    # 获取指定索引的数据
    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])  # 返回指定索引的音频文本说话者对

    # 获取数据集的长度
    def __len__(self):
        return len(self.audiopaths_sid_text)  # 返回音频路径、sid和文本的数量
class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        # 初始化函数，用于指定是否返回ids
        self.return_ids = return_ids

class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        # 初始化函数，用于创建分布式的bucket采样器
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        # 获取数据集的长度
        self.lengths = dataset.lengths
        # 设置批量大小
        self.batch_size = batch_size
        # 设置边界
        self.boundaries = boundaries

        # 创建bucket和每个bucket的样本数量
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        # 计算总大小
        self.total_size = sum(self.num_samples_per_bucket)
        # 计算每个replica的样本数量
        self.num_samples = self.total_size // self.num_replicas
    # 创建一个空列表，列表长度为边界值列表长度减一
    buckets = [[] for _ in range(len(self.boundaries) - 1)]
    # 遍历长度列表
    for i in range(len(self.lengths)):
        # 获取当前长度
        length = self.lengths[i]
        # 通过二分查找获取长度所在的桶的索引
        idx_bucket = self._bisect(length)
        # 如果找到了对应的桶
        if idx_bucket != -1:
            # 将当前长度的索引添加到对应的桶中
            buckets[idx_bucket].append(i)

    # 尝试执行以下代码块，如果出现异常则执行 except 代码块
    try:
        # 从后往前遍历桶列表
        for i in range(len(buckets) - 1, 0, -1):
            # 如果当前桶为空
            if len(buckets[i]) == 0:
                # 移除当前桶
                buckets.pop(i)
                # 移除对应的边界值
                self.boundaries.pop(i + 1)
        # 断言所有桶都不为空
        assert all(len(bucket) > 0 for bucket in buckets)
    # 当有一个桶没有被遍历到时
    except Exception as e:
        # 打印警告信息
        print("Bucket warning ", e)
        # 从后往前遍历桶列表
        for i in range(len(buckets) - 1, -1, -1):
            # 如果当前桶为空
            if len(buckets[i]) == 0:
                # 移除当前桶
                buckets.pop(i)
                # 移除对应的边界值
                self.boundaries.pop(i + 1)

    # 计算每个桶中样本的数量
    num_samples_per_bucket = []
    for i in range(len(buckets)):
        # 获取当前桶的长度
        len_bucket = len(buckets[i])
        # 计算需要补齐的样本数量
        total_batch_size = self.num_replicas * self.batch_size
        rem = (
            total_batch_size - (len_bucket % total_batch_size)
        ) % total_batch_size
        # 将补齐后的样本数量添加到列表中
        num_samples_per_bucket.append(len_bucket + rem)
    # 返回桶列表和每个桶中样本的数量列表
    return buckets, num_samples_per_bucket
    # 定义一个迭代器方法，用于生成数据批次
    def __iter__(self):
        # 根据 epoch 确定性地进行洗牌
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        # 如果需要洗牌
        if self.shuffle:
            # 对每个数据桶进行洗牌
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            # 如果不需要洗牌，直接生成索引
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        # 遍历每个数据桶
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            # 如果数据桶为空，则跳过
            if len_bucket == 0:
                continue
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # 添加额外的样本使其能够均匀分割
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # 对数据进行子采样
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # 进行批处理
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        # 如果需要洗牌
        if self.shuffle:
            # 对生成的批次进行洗牌
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        # 将生成的批次保存到实例变量中
        self.batches = batches

        # 断言生成的批次数量乘以批次大小等于总样本数
        assert len(self.batches) * self.batch_size == self.num_samples
        # 返回批次的迭代器
        return iter(self.batches)
    # 二分查找算法，用于查找元素 x 在 boundaries 列表中的位置
    def _bisect(self, x, lo=0, hi=None):
        # 如果未指定 hi 参数，则默认为 boundaries 列表的长度减一
        if hi is None:
            hi = len(self.boundaries) - 1
    
        # 如果 hi 大于 lo
        if hi > lo:
            # 计算中间位置
            mid = (hi + lo) // 2
            # 如果 x 处于 boundaries[mid] 和 boundaries[mid+1] 之间
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            # 如果 x 小于等于 boundaries[mid]
            elif x <= self.boundaries[mid]:
                # 递归调用 _bisect 函数，在 lo 和 mid 之间继续查找
                return self._bisect(x, lo, mid)
            else:
                # 递归调用 _bisect 函数，在 mid+1 和 hi 之间继续查找
                return self._bisect(x, mid + 1, hi)
        else:
            # 如果 hi 不大于 lo，则返回 -1，表示未找到
            return -1
    
    # 返回样本数量除以批量大小的整数部分，用于确定迭代次数
    def __len__(self):
        return self.num_samples // self.batch_size
```