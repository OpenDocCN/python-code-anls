# `.\lucidrains\tf-bind-transformer\tf_bind_transformer\data_bigwig.py`

```py
# 导入必要的库
from pathlib import Path
import polars as pl
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

# 导入自定义的数据集和函数
from tf_bind_transformer.data import FactorProteinDataset, ContextDataset, cast_list, filter_df_by_tfactor_fastas
from tf_bind_transformer.data import pl_isin, pl_notin, fetch_experiments_index, parse_exp_target_cell, read_bed, cycle, filter_by_col_isin
from tf_bind_transformer.data import CHR_IDS, CHR_NAMES, get_chr_names
from enformer_pytorch import FastaInterval

# 尝试导入 pyBigWig 库，如果导入失败则打印提示信息并退出程序
try:
    import pyBigWig
except ImportError:
    print('pyBigWig needs to be installed - conda install pyBigWig')
    exit()

# 定义一个函数，用于检查变量是否存在
def exists(val):
    return val is not None

# 定义一个函数，用于处理 CHIP ATLAS 数据集中的实验、目标和细胞类型信息
def chip_atlas_add_experiment_target_cell(
    df,
    col_target = 'column_4',
    col_cell_type = 'column_5'
):
    df = df.clone()

    # 提取目标信息并转换为大写格式
    targets = df.select(col_target)
    targets = targets.to_series(0).str.to_uppercase().rename('target')
    df.insert_at_idx(2, targets)

    # 提取细胞类型信息
    cell_type = df.select(col_cell_type)
    cell_type = cell_type.rename({col_cell_type: 'cell_type'}).to_series(0)
    df.insert_at_idx(2, cell_type)

    return df

# 定义一个数据集类，用于处理 BigWig 数据
class BigWigDataset(Dataset):
    def __init__(
        self,
        *,
        factor_fasta_folder,
        bigwig_folder,
        enformer_loci_path,
        fasta_file,
        annot_file = None,
        filter_chromosome_ids = None,
        exclude_targets = None,
        include_targets = None,
        exclude_cell_types = None,
        include_cell_types = None,
        df_frac = 1.,
        experiments_json_path = None,
        include_biotypes_metadata_in_context = False,
        biotypes_metadata_path = None,
        filter_sequences_by = None,
        include_biotypes_metadata_columns = [],
        biotypes_metadata_delimiter = ' | ',
        only_ref = ['mm10', 'hg38'],
        factor_species_priority = ['human', 'mouse'],
        downsample_factor = 128,
        target_length = 896,
        bigwig_reduction_type = 'sum',
        **kwargs
    # 初始化函数，继承父类的初始化方法
    def __init__(
        super().__init__()
        # 断言注释文件存在
        assert exists(annot_file) 

        # 如果 bigwig 文件夹不存在，则设置为无效，目标数量为 0
        if not exists(bigwig_folder):
            self.invalid = True
            self.ntargets = 0
            return

        # 将 bigwig 文件夹路径转换为 Path 对象
        bigwig_folder = Path(bigwig_folder)
        # 断言 bigwig 文件夹存在
        assert bigwig_folder.exists(), 'bigwig folder does not exist'

        # 获取 bigwig 文件夹下所有的 .bw 文件名列表
        bw_experiments = [p.stem for p in bigwig_folder.glob('*.bw')]
        # 断言至少有一个 bigwig 文件存在
        assert len(bw_experiments) > 0, 'no bigwig files found in bigwig folder'

        # 读取 enformer_loci_path 中的 loci 数据
        loci = read_bed(enformer_loci_path)
        # 读取 annot_file 中的注释数据
        annot_df = pl.read_csv(annot_file, sep = "\t", has_headers = False, columns = list(map(lambda i: f'column_{i + 1}', range(17))))

        # 根据 only_ref 列表中的值筛选 annot_df
        annot_df = annot_df.filter(pl_isin('column_2', only_ref))
        # 根据 bw_experiments 列表中的值筛选 annot_df
        annot_df = filter_by_col_isin(annot_df, 'column_1', bw_experiments)

        # 如果 df_frac 小于 1，则对 annot_df 进行采样
        if df_frac < 1:
            annot_df = annot_df.sample(frac = df_frac)

        # 初始化 dataset_chr_ids 为 CHR_IDS
        dataset_chr_ids = CHR_IDS

        # 如果 filter_chromosome_ids 存在，则更新 dataset_chr_ids
        if exists(filter_chromosome_ids):
            dataset_chr_ids = dataset_chr_ids.intersection(set(filter_chromosome_ids))

        # 根据 dataset_chr_ids 中的值筛选 loci
        loci = loci.filter(pl_isin('column_1', get_chr_names(dataset_chr_ids)))

        # 如果 filter_sequences_by 存在，则根据其值筛选 loci
        if exists(filter_sequences_by):
            col_name, col_val = filter_sequences_by
            loci = loci.filter(pl.col(col_name) == col_val)

        # 初始化 FactorProteinDataset 对象
        self.factor_ds = FactorProteinDataset(factor_fasta_folder, species_priority = factor_species_priority)

        # 获取 annot_df 中 column_1 列的唯一值集合
        exp_ids = set(annot_df.get_column('column_1').to_list())

        # 添加实验目标细胞到 annot_df
        annot_df = chip_atlas_add_experiment_target_cell(annot_df)
        # 根据 factor_fasta_folder 筛选 annot_df
        annot_df = filter_df_by_tfactor_fastas(annot_df, factor_fasta_folder)

        # 获取筛选后的 exp_ids
        filtered_exp_ids = set(annot_df.get_column('column_1').to_list())

        # 计算被筛选掉的 exp_ids
        filtered_out_exp_ids = exp_ids - filtered_exp_ids
        print(f'{", ".join(only_ref)} - {len(filtered_out_exp_ids)} experiments filtered out by lack of transcription factor fastas', filtered_out_exp_ids)

        # 根据 include_targets 和 exclude_targets 筛选 annot_df
        include_targets = cast_list(include_targets)
        exclude_targets = cast_list(exclude_targets)

        if include_targets:
            annot_df = annot_df.filter(pl_isin('target', include_targets))

        if exclude_targets:
            annot_df = annot_df.filter(pl_notin('target', exclude_targets))

        # 根据 include_cell_types 和 exclude_cell_types 筛选 annot_df
        include_cell_types = cast_list(include_cell_types)
        exclude_cell_types = cast_list(exclude_cell_types)

        if include_cell_types:
            annot_df = annot_df.filter(pl_isin('cell_type', include_cell_types))

        if exclude_cell_types:
            annot_df = annot_df.filter(pl_notin('cell_type', exclude_cell_types))

        # 初始化 FastaInterval 对象
        self.fasta = FastaInterval(fasta_file = fasta_file, **kwargs)

        # 设置 self.df 和 self.annot
        self.df = loci
        self.annot = annot_df
        self.ntargets = self.annot.shape[0]

        # 初始化 bigwigs 列表
        self.bigwigs = [pyBigWig.open(str(bigwig_folder / f'{str(i)}.bw')) for i in self.annot.get_column("column_1")]

        # 设置 downsample_factor 和 target_length
        self.downsample_factor = downsample_factor
        self.target_length = target_length

        # 设置 bigwig_reduction_type 和 invalid
        self.bigwig_reduction_type = bigwig_reduction_type
        self.invalid = False

    # 返回数据集的长度
    def __len__(self):
        # 如果数据集无效，则长度为 0
        if self.invalid:
            return 0

        # 返回数据集的长度
        return len(self.df) * self.ntargets
    # 从自定义类中获取指定索引的元素
    def __getitem__(self, ind):
        # TODO 返回一个个体的所有目标
        # 从数据框中获取指定索引的染色体名称、起始位置、终止位置和其他信息
        chr_name, begin, end, _ = self.df.row(ind % self.df.shape[0])

        # 从注释中选择目标和细胞类型，并转换为 Series 对象
        targets = self.annot.select('target').to_series(0)
        cell_types = self.annot.select('cell_type').to_series(0)

        # 计算目标在列表中的索引
        ix_target = ind // self.df.shape[0]
    
        # 从列表中获取目标、细胞类型和 bigwig 对象
        target = targets[ix_target]
        context_str = cell_types[ix_target]
        exp_bw = self.bigwigs[ix_target]

        # 获取目标对应的氨基酸序列和基因组序列
        aa_seq = self.factor_ds[target]
        seq = self.fasta(chr_name, begin, end)

        # 计算 bigwig 数据
        output = np.array(exp_bw.values(chr_name, begin, end))
        output = output.reshape((-1, self.downsample_factor))

        # 根据指定的 bigwig 缩减类型进行处理
        if self.bigwig_reduction_type == 'mean':
            om = np.nanmean(output, axis = 1)
        elif self.bigwig_reduction_type == 'sum':
            om = np.nansum(output, axis = 1)
        else:
            raise ValueError(f'unknown reduction type {self.bigwig_reduction_type}')

        # 获取输出数据的长度
        output_length = output.shape[0]

        # 检查输出长度是否小于目标长度
        if output_length < self.target_length:
            assert f'target length {self.target_length} cannot be less than the {output_length}'

        # 计算需要裁剪的部分
        trim = (output.shape[0] - self.target_length) // 2
        om = om[trim:-trim]

        # 将 NaN 值替换为 0
        np.nan_to_num(om, copy = False)

        # 创建 PyTorch 张量作为标签
        label = torch.Tensor(om)
        return seq, aa_seq, context_str, label
# BigWig 数据集，仅包含轨迹

class BigWigTracksOnlyDataset(Dataset):
    def __init__(
        self,
        *,
        bigwig_folder,  # BigWig 文件夹路径
        enformer_loci_path,  # Enformer loci 路径
        fasta_file,  # FASTA 文件路径
        ref,  # 参考
        annot_file = None,  # 注释文件，默认为空
        filter_chromosome_ids = None,  # 过滤染色体 ID，默认为空
        downsample_factor = 128,  # 下采样因子，默认为 128
        target_length = 896,  # 目标长度，默认为 896
        bigwig_reduction_type = 'sum',  # BigWig 减少类型，默认为 'sum'
        filter_sequences_by = None,  # 过滤序列，默认为空
        **kwargs
    ):
        super().__init__()
        assert exists(annot_file)

        if not exists(bigwig_folder):
            self.invalid = True
            self.ntargets = 0
            return

        bigwig_folder = Path(bigwig_folder)
        assert bigwig_folder.exists(), 'bigwig folder does not exist'

        bw_experiments = [p.stem for p in bigwig_folder.glob('*.bw')]
        assert len(bw_experiments) > 0, 'no bigwig files found in bigwig folder'

        loci = read_bed(enformer_loci_path)

        annot_df = pl.read_csv(annot_file, sep = "\t", has_headers = False, columns = list(map(lambda i: f'column_{i + 1}', range(17))))

        annot_df = annot_df.filter(pl.col('column_2') == ref)
        annot_df = filter_by_col_isin(annot_df, 'column_1', bw_experiments)

        dataset_chr_ids = CHR_IDS

        if exists(filter_chromosome_ids):
            dataset_chr_ids = dataset_chr_ids.intersection(set(filter_chromosome_ids))

        # filtering loci by chromosomes
        # as well as training or validation

        loci = loci.filter(pl_isin('column_1', get_chr_names(dataset_chr_ids)))

        if exists(filter_sequences_by):
            col_name, col_val = filter_sequences_by
            loci = loci.filter(pl.col(col_name) == col_val)

        self.fasta = FastaInterval(fasta_file = fasta_file, **kwargs)

        self.df = loci
        self.annot = annot_df
        self.ntargets = self.annot.shape[0]

        # bigwigs

        self.bigwigs = [(str(i), pyBigWig.open(str(bigwig_folder / f'{str(i)}.bw'))) for i in self.annot.get_column("column_1")]
        
        self.downsample_factor = downsample_factor
        self.target_length = target_length

        self.bigwig_reduction_type = bigwig_reduction_type
        self.invalid = False

    def __len__(self):
        if self.invalid:
            return 0

        return len(self.df) * int(self.ntargets > 0)

    def __getitem__(self, ind):
        chr_name, begin, end, _ = self.df.row(ind)

        # figure out ref and fetch appropriate sequence

        seq = self.fasta(chr_name, begin, end)

        # calculate bigwig
        # properly downsample and then crop

        all_bw_values = []

        for bw_path, bw in self.bigwigs:
            try:
                bw_values = bw.values(chr_name, begin, end)
                all_bw_values.append(bw_values)
            except:
                print(f'hitting invalid range for {bw_path} - ({chr_name}, {begin}, {end})')
                exit()

        output = np.stack(all_bw_values, axis = -1)
        output = output.reshape((-1, self.downsample_factor, self.ntargets))

        if self.bigwig_reduction_type == 'mean':
            om = np.nanmean(output, axis = 1)
        elif self.bigwig_reduction_type == 'sum':
            om = np.nansum(output, axis = 1)
        else:
            raise ValueError(f'unknown reduction type {self.bigwig_reduction_type}')

        output_length = output.shape[0]

        if output_length < self.target_length:
            assert f'target length {self.target_length} cannot be less than the {output_length}'

        trim = (output.shape[0] - self.target_length) // 2
        om = om[trim:-trim]

        np.nan_to_num(om, copy = False)

        label = torch.Tensor(om)
        return seq, label

# 数据加载器

def bigwig_collate_fn(data):
    seq, aa_seq, context_str, labels = list(zip(*data))
    return torch.stack(seq), tuple(aa_seq), tuple(context_str), torch.stack(labels)

def get_bigwig_dataloader(ds, cycle_iter = False, **kwargs):
    dataset_len = len(ds)
    # 从参数中获取批量大小
    batch_size = kwargs.get('batch_size')
    # 检查数据集长度是否大于批量大小，以确定是否丢弃最后一批数据
    drop_last = dataset_len > batch_size

    # 使用DataLoader加载数据集，指定数据集、数据处理函数、是否丢弃最后一批数据以及其他参数
    dl = DataLoader(ds, collate_fn = bigwig_collate_fn, drop_last = drop_last, **kwargs)
    # 根据是否循环迭代选择返回迭代器或循环迭代器
    wrapper = cycle if cycle_iter else iter
    # 返回包装后的数据加载器
    return wrapper(dl)
# 定义一个函数，用于获取包含大WIG轨迹数据的数据加载器
def get_bigwig_tracks_dataloader(ds, cycle_iter = False, **kwargs):
    # 获取数据集的长度
    dataset_len = len(ds)
    # 获取批处理大小
    batch_size = kwargs.get('batch_size')
    # 如果数据集长度大于批处理大小，则设置为True，否则为False
    drop_last = dataset_len > batch_size

    # 创建一个数据加载器，根据是否丢弃最后一批数据进行设置
    dl = DataLoader(ds, drop_last = drop_last, **kwargs)
    # 根据cycle_iter参数选择返回数据加载器的迭代器类型
    wrapper = cycle if cycle_iter else iter
    # 返回迭代器类型的数据加载器
    return wrapper(dl)
```