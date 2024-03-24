# `.\lucidrains\DALLE2-pytorch\dalle2_pytorch\dataloaders\decoder_loader.py`

```py
import os
import webdataset as wds
import torch
from torch.utils.data import DataLoader
import numpy as np
import fsspec
import shutil

def get_shard(filename):
    """
    Filenames with shards in them have a consistent structure that we can take advantage of
    Standard structure: path/to/file/prefix_string_00001.ext
    """
    try:
        return filename.split("_")[-1].split(".")[0]
    except ValueError:
        raise RuntimeError(f"Could not find shard for filename {filename}")

def get_example_file(fs, path, file_format):
    """
    Given a file system and a file extension, return the example file
    """
    return fs.glob(os.path.join(path, f"*.{file_format}"))[0]

def embedding_inserter(samples, embeddings_url, index_width, sample_key='npy', handler=wds.handlers.reraise_exception):
    """Given a datum of {"__key__": str, "__url__": str, ...} adds the corresponding embedding and yields"""
    previous_tar_url = None
    current_embeddings = None
    # Get a reference to an abstract file system where the embeddings are stored
    embeddings_fs, embeddings_path = fsspec.core.url_to_fs(embeddings_url)
    example_embedding_file = get_example_file(embeddings_fs, embeddings_path, "npy")
    example_embedding_shard = get_shard(example_embedding_file)
    emb_shard_width = len(example_embedding_shard)
    # Easier to get the basename without the shard once than search through for the correct file every time
    embedding_file_basename = '_'.join(example_embedding_file.split("_")[:-1]) + "_"

    def load_corresponding_embeds(tar_url):
      """Finds and reads the npy files that contains embeddings for the given webdataset tar"""
      shard = int(tar_url.split("/")[-1].split(".")[0])
      embedding_url = embedding_file_basename + str(shard).zfill(emb_shard_width) + '.npy'
      with embeddings_fs.open(embedding_url) as f:
        data = np.load(f)
      return torch.from_numpy(data)

    for sample in samples:
        try:
            tar_url = sample["__url__"]
            key = sample["__key__"]
            if tar_url != previous_tar_url:
                # If the tar changed, we need to download new embeddings
                # This means if we shuffle before inserting it will load many more files than we expect and be very inefficient.
                previous_tar_url = tar_url
                current_embeddings = load_corresponding_embeds(tar_url)
                
            embedding_index = int(key[-index_width:])
            embedding = current_embeddings[embedding_index]
            # We need to check if this sample is nonzero. If it is, this embedding is not valid and we should continue to the next loop
            if torch.count_nonzero(embedding) == 0:
                raise RuntimeError(f"Webdataset had a sample, but no embedding was found. ImgShard: {key[:-index_width]} - Index: {key[-index_width:]}")
            sample[sample_key] = embedding
            yield sample
        except Exception as exn:  # From wds implementation
            if handler(exn):
                continue
            else:
                break
insert_embedding = wds.filters.pipelinefilter(embedding_inserter)

def unassociated_shard_skipper(tarfiles, embeddings_url, handler=wds.handlers.reraise_exception):
    """Finds if there is a corresponding embedding for the tarfile at { url: [URL] }"""
    embeddings_fs, embeddings_path = fsspec.core.url_to_fs(embeddings_url)
    embedding_files = embeddings_fs.ls(embeddings_path)
    get_embedding_shard = lambda embedding_file: int(embedding_file.split("_")[-1].split(".")[0])
    embedding_shards = set([get_embedding_shard(filename) for filename in embedding_files])  # Sets have O(1) check for member

    get_tar_shard = lambda tar_file: int(tar_file.split("/")[-1].split(".")[0])
    # 遍历 tarfiles 列表中的每个 tarfile
    for tarfile in tarfiles:
        try:
            # 获取 tarfile 对应的 webdataset shard
            webdataset_shard = get_tar_shard(tarfile["url"])
            # 如果该 shard 有关联的 embeddings 文件，则返回该 tarfile
            # 否则继续迭代直到找到有关联的 embeddings 文件
            if webdataset_shard in embedding_shards:
                yield tarfile
        except Exception as exn:  # 从 wds 实现中捕获异常
            # 如果 handler 函数处理了异常，则继续循环
            if handler(exn):
                continue
            # 如果 handler 函数未处理异常，则跳出循环
            else:
                break
# 创建一个过滤器，用于跳过未关联的碎片
skip_unassociated_shards = wds.filters.pipelinefilter(unassociated_shard_skipper)

# 将样本中的img_emb和text_emb键合并为一个键"emb": { "text": text_emb, "img": img_emb }
# 如果text_emb和img_emb中的一个或两个不存在于样本中，则只添加存在的部分
def join_embeddings(samples, handler=wds.handlers.reraise_exception):
    """
    Takes the img_emb and text_emb keys and turns them into one key "emb": { "text": text_emb, "img": img_emb }
    either or both of text_emb and img_emb may not be in the sample so we only add the ones that exist
    """
    for sample in samples:
        try:
            sample['emb'] = {}
            if 'text_emb' in sample:
                sample['emb']['text'] = sample['text_emb']
            if 'img_emb' in sample:
                sample['emb']['img'] = sample['img_emb']
            yield sample
        except Exception as exn:  # From wds implementation
            if handler(exn):
                continue
            else:
                break

# 验证样本中是否存在所需的键，如果不存在则抛出异常
def verify_keys(samples, required_keys, handler=wds.handlers.reraise_exception):
    """
    Requires that both the image and embedding are present in the sample
    This is important to do as a user may forget they do not have embeddings in their webdataset and neglect to add them using the embedding_folder_url parameter.
    """
    for sample in samples:
        try:
            for key in required_keys:
                assert key in sample, f"Sample {sample['__key__']} missing {key}. Has keys {sample.keys()}"
            yield sample
        except Exception as exn:  # From wds implementation
            if handler(exn):
                continue
            else:
                break

# 创建一个过滤器，用于验证样本中是否存在所需的键
key_verifier = wds.filters.pipelinefilter(verify_keys)

# ImageEmbeddingDataset类，是DataPipeline的流式接口包装器，返回图像嵌入对
# 从webdataset中读取npy文件作为嵌入，如果存在的话。如果设置了embedding_folder_url，则会从替代来源插入它们
class ImageEmbeddingDataset(wds.DataPipeline, wds.compat.FluidInterface):
    """
    A fluid interface wrapper for DataPipline that returns image embedding pairs
    Reads embeddings as npy files from the webdataset if they exist. If embedding_folder_url is set, they will be inserted in from the alternate source.
    """

    def __init__(
            self,
            urls,
            img_embedding_folder_url=None,
            text_embedding_folder_url=None,
            index_width=None,
            img_preproc=None,
            extra_keys=[],
            handler=wds.handlers.reraise_exception,
            resample=False,
            shuffle_shards=True
    def preproc(self, sample):
        """Applies the preprocessing for images"""
        if self.img_preproc is not None:
            sample["jpg"] = self.img_preproc(sample["jpg"])
        return sample

# 创建一个图像嵌入数据加载器的便捷函数
def create_image_embedding_dataloader(
    tar_url,
    num_workers,
    batch_size,
    img_embeddings_url=None,
    text_embeddings_url=None,
    index_width=None,
    shuffle_num = None,
    shuffle_shards = True,
    resample_shards = False, 
    img_preproc=None,
    extra_keys=[],
    handler=wds.handlers.reraise_exception#warn_and_continue
):
    """
    Convenience function to create an image embedding dataseta and dataloader in one line

    :param tar_url: A url pointing to the tar files of the webdataset formatted as /path/to/webdataset/{0000..9999}.tar
    :param num_workers: The number of workers to use for the dataloader
    :param batch_size: The batch size to use for the dataloader
    :param embeddings_url: Required if webdataset does not contain embeddings. A url pointing to the npy files of the embeddings. Should have the same number of shards as the webdataset.
        Webdataset image keys should align with the index of the embedding. This means missing image indices must have a corresponding embedding of all zeros.
    :param index_width: The number of digits in the index. This is used to align the embedding index with the image index.
            For example, if a file in the webdataset shard 3 is named 0003039.jpg, we know the shard is 4 digits and the last 3 digits are the index_width.
    :param shuffle_num: If not None, shuffle the dataset with this size buffer after sampling.
    :param shuffle_shards: If true, shuffle the shards before sampling. This cannot be true if resample is true.
    :param resample_shards: 如果为True，则对webdataset分片进行有放回抽样。如果设置为True，则需要设置自己的epoch大小，因为它将无限重采样。
    :param handler: webdataset处理程序。
    """
    # 创建ImageEmbeddingDataset对象
    ds = ImageEmbeddingDataset(
        tar_url,
        img_embedding_folder_url=img_embeddings_url,
        text_embedding_folder_url=text_embeddings_url,
        index_width=index_width,
        shuffle_shards=shuffle_shards,
        resample=resample_shards,
        extra_keys=extra_keys,
        img_preproc=img_preproc,
        handler=handler
    )
    # 如果设置了shuffle_num并且大于0，则对数据集进行洗牌
    if shuffle_num is not None and shuffle_num > 0:
        ds.shuffle(1000)
    # 返回一个DataLoader对象
    return DataLoader(
        ds,
        num_workers=num_workers,
        batch_size=batch_size,
        prefetch_factor=2,  # 这可能是一个好主意，使其较高，以便预取下一个npy文件
        pin_memory=True,
        shuffle=False
    )
```