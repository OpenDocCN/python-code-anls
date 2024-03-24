# `.\lucidrains\tf-bind-transformer\tf_bind_transformer\data.py`

```
# 导入所需的模块
from Bio import SeqIO
from random import choice, randrange
from pathlib import Path
import functools
import polars as pl
from collections import defaultdict

import os
import json
import shutil
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from tf_bind_transformer.gene_utils import parse_gene_name
from enformer_pytorch import FastaInterval

from pyfaidx import Fasta
import pybedtools

# 定义函数判断值是否存在
def exists(val):
    return val is not None

# 定义函数返回默认值
def default(val, d):
    return val if exists(val) else d

# 定义函数查找满足条件的第一个元素的索引
def find_first_index(cond, arr):
    for ind, el in enumerate(arr):
        if cond(el):
            return ind
    return -1

# 定义函数将值转换为列表
def cast_list(val = None):
    if not exists(val):
        return []
    return [val] if not isinstance(val, (tuple, list)) else val

# 读取 BED 文件并返回 Polars 数据框
def read_bed(path):
    return pl.read_csv(path, sep = '\t', has_headers = False)

# 将 Polars 数据框保存为 BED 文件
def save_bed(df, path):
    df.to_csv(path, sep = '\t', has_header = False)

# 解析实验、目标和细胞类型
def parse_exp_target_cell(exp_target_cell):
    experiment, target, *cell_type = exp_target_cell.split('.')
    cell_type = '.'.join(cell_type) # 处理细胞类型包含句点的情况
    return experiment, target, cell_type

# 获取数据集的索引，用于提供辅助读取值预测的测序 reads
def fetch_experiments_index(path):
    if not exists(path):
        return dict()

    exp_path = Path(path)
    assert exp_path.exists(), 'path to experiments json must exist'

    root_json = json.loads(exp_path.read_text())
    experiments = root_json['experiments']

    index = {}
    for experiment in experiments:
        exp_id = experiment['accession']

        if 'details' not in experiment:
            continue

        details = experiment['details']

        if 'datasets' not in details:
            continue

        datasets = details['datasets']

        for dataset in datasets:
            dataset_name = dataset['dataset_name']
            index[dataset_name] = dataset['peaks_NR']

    return index

# 根据基因名和 Uniprot ID 获取蛋白质序列
class FactorProteinDatasetByUniprotID(Dataset):
    def __init__(
        self,
        folder,
        species_priority = ['human', 'mouse']
    ):
        super().__init__()
        fasta_paths = [*Path(folder).glob('*.fasta')]
        assert len(fasta_paths) > 0, f'no fasta files found at {folder}'
        self.paths = fasta_paths
        self.index_by_id = dict()

        for path in fasta_paths:
            gene, uniprotid, *_ = path.stem.split('.')
            self.index_by_id[uniprotid] = path

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, uid):
        index = self.index_by_id

        if uid not in index:
            return None

        entry = index[uid]
        fasta = SeqIO.read(entry, 'fasta')
        return str(fasta.seq)

# 获取蛋白质数据集
class FactorProteinDataset(Dataset):
    def __init__(
        self,
        folder,
        species_priority = ['human', 'mouse', 'unknown'],
        return_tuple_only = False
    # 初始化函数，接受一个文件夹路径作为参数
    def __init__(
        super().__init__()
        # 获取文件夹中所有以 .fasta 结尾的文件路径
        fasta_paths = [*Path(folder).glob('*.fasta')]
        # 断言至少找到一个 .fasta 文件，否则抛出异常
        assert len(fasta_paths) > 0, f'no fasta files found at {folder}'
        # 将找到的文件路径列表保存在 self.paths 中

        self.paths = fasta_paths

        # 使用 defaultdict 创建一个以基因名为键，文件路径列表为值的字典
        index_by_gene = defaultdict(list)
        # 是否只返回元组，即使只有一个亚单位
        self.return_tuple_only = return_tuple_only 

        # 遍历每个 .fasta 文件路径
        for path in fasta_paths:
            # 从文件名中提取基因名和 Uniprot ID
            gene, uniprotid, *_ = path.stem.split('.')
            # 将文件路径添加到对应基因名的列表中
            index_by_gene[gene].append(path)

        # 用于从文件路径中提取物种信息的 lambda 函数
        get_species_from_path = lambda p: p.stem.split('_')[-1].lower() if '_' in p.stem else 'unknown'

        # 使用 defaultdict 创建一个以基因名为键，经过物种筛选后的文件路径列表为值的字典
        filtered_index_by_gene = defaultdict(list)

        # 遍历每个基因及其对应的文件路径列表
        for gene, gene_paths in index_by_gene.items():
            # 计算每个物种在该基因下的文件数量
            species_count = list(map(lambda specie: len(list(filter(lambda p: get_species_from_path(p) == specie, gene_paths))), species_priority))
            # 找到第一个文件数量不为零的物种索引
            species_ind_non_zero = find_first_index(lambda t: t > 0, species_count)

            # 如果没有找到文件数量不为零的物种索引，则跳过该基因
            if species_ind_non_zero == -1:
                continue

            # 获取该基因下文件数量不为零的物种
            species = species_priority[species_ind_non_zero]
            # 将该基因下属于指定物种的文件路径添加到筛选后的字典中
            filtered_index_by_gene[gene] = list(filter(lambda p: get_species_from_path(p) == species, gene_paths))

        # 将筛选后的字典保存在 self.index_by_gene 中

        self.index_by_gene = filtered_index_by_gene

    # 返回文件路径列表的长度
    def __len__(self):
        return len(self.paths)

    # 根据未解析的基因名获取对应的序列
    def __getitem__(self, unparsed_gene_name):
        # 获取基因名对应的文件路径列表
        index = self.index_by_gene

        # 解析基因名
        genes = parse_gene_name(unparsed_gene_name)
        seqs = []

        # 遍历每个基因
        for gene in genes:
            entry = index[gene]

            # 如果该基因没有对应的文件路径，则打印提示信息并继续下一个基因
            if len(entry) == 0:
                print(f'no entries for {gene}')
                continue

            # 从文件路径列表中随机选择一个文件路径
            path = choice(entry) if isinstance(entry, list) else entry

            # 读取 fasta 文件中的序列
            fasta = SeqIO.read(path, 'fasta')
            seqs.append(str(fasta.seq))

        # 将序列列表转换为元组
        seqs = tuple(seqs)

        # 如果只有一个序列且不要求返回元组，则返回该序列
        if len(seqs) == 1 and not self.return_tuple_only:
            return seqs[0]

        # 否则返回序列元组
        return seqs
# 重新映射数据框函数

# 获取染色体名称集合
def get_chr_names(ids):
    return set(map(lambda t: f'chr{t}', ids))

# 定义染色体编号集合和染色体名称集合
CHR_IDS = set([*range(1, 23), 'X'])
CHR_NAMES = get_chr_names(CHR_IDS)

# 重新映射数据框并添加实验、目标和细胞类型信息
def remap_df_add_experiment_target_cell(df, col = 'column_4'):
    df = df.clone()

    # 提取实验信息
    exp_id = df.select([pl.col(col).str.extract(r"^([\w\-]+)\.*")])
    exp_id = exp_id.rename({col: 'experiment'}).to_series(0)
    df.insert_at_idx(3, exp_id)

    # 提取目标信息
    targets = df.select([pl.col(col).str.extract(r"[\w\-]+\.([\w\-]+)\.[\w\-]+")])
    targets = targets.rename({col: 'target'}).to_series(0)
    df.insert_at_idx(3, targets)

    # 提取细胞类型信息
    cell_type = df.select([pl.col(col).str.extract(r"^.*\.([\w\-]+)$")])
    cell_type = cell_type.rename({col: 'cell_type'}).to_series(0)
    df.insert_at_idx(3, cell_type)

    return df

# 判断列中元素是否在数组中
def pl_isin(col, arr):
    equalities = list(map(lambda t: pl.col(col) == t, arr))
    return functools.reduce(lambda a, b: a | b, equalities)

# 判断列中元素是否不在数组中
def pl_notin(col, arr):
    equalities = list(map(lambda t: pl.col(col) != t, arr))
    return functools.reduce(lambda a, b: a & b, equalities)

# 根据列中元素是否在数组中进行过滤数据框
def filter_by_col_isin(df, col, arr, chunk_size = 25):
    """
    polars 似乎存在一个 bug
    当 OR 条件超过 25 个时会冻结（对于 pl_isin）
    拆分成 25 个一组进行处理，然后合并
    """
    dataframes = []
    for i in range(0, len(arr), chunk_size):
        sub_arr = arr[i:(i + chunk_size)]
        filtered_df = df.filter(pl_isin(col, sub_arr))
        dataframes.append(filtered_df)
    return pl.concat(dataframes)

# 根据 BED 文件进行过滤
def filter_bed_file_by_(bed_file_1, bed_file_2, output_file):
    # 由 OpenAI Codex 生成

    bed_file_1_bedtool = pybedtools.BedTool(bed_file_1)
    bed_file_2_bedtool = pybedtools.BedTool(bed_file_2)
    bed_file_1_bedtool_intersect_bed_file_2_bedtool = bed_file_1_bedtool.intersect(bed_file_2_bedtool, v = True)
    bed_file_1_bedtool_intersect_bed_file_2_bedtool.saveas(output_file)

# 根据 TF 蛋白质序列文件进行过滤数据框
def filter_df_by_tfactor_fastas(df, folder):
    files = [*Path(folder).glob('**/*.fasta')]
    present_target_names = set([f.stem.split('.')[0] for f in files])
    all_df_targets = df.get_column('target').unique().to_list()

    all_df_targets_with_parsed_name = [(target, parse_gene_name(target)) for target in all_df_targets]
    unknown_targets = [target for target, parsed_target_name in all_df_targets_with_parsed_name for parsed_target_name_sub_el in parsed_target_name if parsed_target_name_sub_el not in present_target_names]

    if len(unknown_targets) > 0:
        df = df.filter(pl_notin('target', unknown_targets))
    return df

# 从 FASTA 文件生成随机范围
def generate_random_ranges_from_fasta(
    fasta_file,
    *,
    output_filename = 'random-ranges.bed',
    context_length,
    filter_bed_files = [],
    num_entries_per_key = 10,
    keys = None,
):
    fasta = Fasta(fasta_file)
    tmp_file = f'/tmp/{output_filename}'

    with open(tmp_file, 'w') as f:
        for chr_name in sorted(CHR_NAMES):
            print(f'generating ranges for {chr_name}')

            if chr_name not in fasta:
                print(f'{chr_name} not found in fasta file')
                continue

            chromosome = fasta[chr_name]
            chromosome_length = len(chromosome)

            start = np.random.randint(0, chromosome_length - context_length, (num_entries_per_key,))
            end = start + context_length
            start_and_end = np.stack((start, end), axis = -1)

            for row in start_and_end.tolist():
                start, end = row
                f.write('\t'.join((chr_name, str(start), str(end))) + '\n')

    for file in filter_bed_files:
        filter_bed_file_by_(tmp_file, file, tmp_file)

    shutil.move(tmp_file, f'./{output_filename}')

    print('success')

# 上下文字符串创建类

class ContextDataset(Dataset):
    def __init__(
        self,
        biotypes_metadata_path = None,
        include_biotypes_metadata_in_context = False,
        include_biotypes_metadata_columns = [],
        biotypes_metadata_delimiter = ' | ',
    # 初始化类的属性，设置是否在上下文中包含生物类型元数据，以及相关的列和分隔符
    def __init__(
        self, include_biotypes_metadata_in_context, include_biotypes_metadata_columns, biotypes_metadata_delimiter
    ):
        self.include_biotypes_metadata_in_context = include_biotypes_metadata_in_context
        self.include_biotypes_metadata_columns = include_biotypes_metadata_columns
        self.biotypes_metadata_delimiter = biotypes_metadata_delimiter

        # 如果要在上下文中包含生物类型元数据
        if include_biotypes_metadata_in_context:
            # 确保要包含的生物类型元数据列数大于0
            assert len(self.include_biotypes_metadata_columns) > 0, 'must have more than one biotype metadata column to include'
            # 确保生物类型元数据路径存在
            assert exists(biotypes_metadata_path), 'biotypes metadata path must be supplied if to be included in context string'

            # 创建路径对象
            p = Path(biotypes_metadata_path)

            # 根据文件后缀选择分隔符
            if p.suffix == '.csv':
                sep = ','
            elif p.suffix == '.tsv':
                sep = '\t'
            else:
                raise ValueError(f'invalid suffix {p.suffix} for biotypes')

            # 读取CSV或TSV文件并存储为DataFrame
            self.df = pl.read_csv(str(p), sep = sep)

    # 返回DataFrame的长度或-1（如果不包含生物类型元数据）
    def __len__():
        return len(self.df) if self.include_biotypes_metadata_in_context else -1

    # 根据生物类型获取上下文字符串
    def __getitem__(self, biotype):
        # 如果不包含生物类型元数据，直接返回生物类型
        if not self.include_biotypes_metadata_in_context:
            return biotype

        # 获取要包含的生物类型元数据列的索引
        col_indices = list(map(self.df.columns.index, self.include_biotypes_metadata_columns))
        # 根据生物类型筛选DataFrame
        filtered = self.df.filter(pl.col('biotype') == biotype)

        # 如果没有找到匹配的行，打印消息并返回生物类型
        if len(filtered) == 0:
            print(f'no rows found for {biotype} in biotype metadata file')
            return biotype

        # 获取匹配行的数据
        row = filtered.row(0)
        # 获取要包含的列的值
        columns = list(map(lambda t: row[t], col_indices))

        # 组合上下文字符串
        context_string = self.biotypes_metadata_delimiter.join([biotype, *columns])
        return context_string
# 定义一个用于重新映射数据的数据集类 RemapAllPeakDataset，继承自 Dataset 类
class RemapAllPeakDataset(Dataset):
    # 初始化函数，接收多个参数
    def __init__(
        self,
        *,
        factor_fasta_folder,  # 因子 fasta 文件夹
        bed_file = None,  # bed 文件，默认为 None
        remap_df = None,  # 重新映射数据框，默认为 None
        filter_chromosome_ids = None,  # 过滤染色体 ID，默认为 None
        exclude_targets = None,  # 排除目标，默认为 None
        include_targets = None,  # 包含目标，默认为 None
        exclude_cell_types = None,  # 排除细胞类型，默认为 None
        include_cell_types = None,  # 包含细胞类型，默认为 None
        remap_df_frac = 1.,  # 重新映射数据框比例，默认为 1
        experiments_json_path = None,  # 实验 JSON 路径，默认为 None
        include_biotypes_metadata_in_context = False,  # 在上下文中包含生物类型元数据，默认为 False
        biotypes_metadata_path = None,  # 生物类型元数据路径，默认为 None
        include_biotypes_metadata_columns = [],  # 包含生物类型元数据列，默认为空列表
        biotypes_metadata_delimiter = ' | ',  # 生物类型元数据分隔符，默认为 ' | '
        balance_sampling_by_target = False,  # 按目标平衡采样，默认为 False
        **kwargs  # 其他关键字参数
    ):
        super().__init__()  # 调用父类的初始化函数
        assert exists(remap_df) ^ exists(bed_file), 'either remap bed file or remap dataframe must be passed in'

        if not exists(remap_df):
            remap_df = read_bed(bed_file)  # 读取 bed 文件并赋值给 remap_df

        if remap_df_frac < 1:
            remap_df = remap_df.sample(frac = remap_df_frac)  # 如果 remap_df_frac 小于 1，则对 remap_df 进行采样

        dataset_chr_ids = CHR_IDS  # 数据集染色体 ID

        if exists(filter_chromosome_ids):
            dataset_chr_ids = dataset_chr_ids.intersection(set(filter_chromosome_ids))  # 如果存在过滤染色体 ID，则取交集

        remap_df = remap_df.filter(pl_isin('column_1', get_chr_names(dataset_chr_ids)))  # 过滤 remap_df 中染色体名称
        remap_df = filter_df_by_tfactor_fastas(remap_df, factor_fasta_folder)  # 根据因子 fasta 文件夹过滤 remap_df

        self.factor_ds = FactorProteinDataset(factor_fasta_folder)  # 初始化因子蛋白数据集

        # 根据包含和排除目标列表过滤数据集
        # (<所有可用目标> 交集 <包含目标>) 减去 <排除目标>

        include_targets = cast_list(include_targets)  # 将包含目标转换为列表
        exclude_targets = cast_list(exclude_targets)  # 将排除目标转换为列表

        if include_targets:
            remap_df = remap_df.filter(pl_isin('target', include_targets))  # 如果包含目标非空，则过滤 remap_df

        if exclude_targets:
            remap_df = remap_df.filter(pl_notin('target', exclude_targets))  # 如果排除目标非空，则过滤 remap_df

        # 根据包含和排除细胞类型列表过滤数据集
        # 与目标相同的逻辑

        include_cell_types = cast_list(include_cell_types)  # 将包含细胞类型转换为列表
        exclude_cell_types = cast_list(exclude_cell_types)  # 将排除细胞类型转换为列表

        if include_cell_types:
            remap_df = remap_df.filter(pl_isin('cell_type', include_cell_types))  # 如果包含细胞类型非空，则过滤 remap_df

        if exclude_cell_types:
            remap_df = remap_df.filter(pl_notin('cell_type', exclude_cell_types))  # 如果排除细胞类型非空，则过滤 remap_df

        assert len(remap_df) > 0, 'dataset is empty by filter criteria'  # 断言数据集不为空

        self.df = remap_df  # 将过滤后的数据集赋值给 self.df
        self.fasta = FastaInterval(**kwargs)  # 初始化 FastaInterval 对象

        self.experiments_index = fetch_experiments_index(experiments_json_path)  # 获取实验索引

        # 平衡目标采样逻辑

        self.balance_sampling_by_target = balance_sampling_by_target  # 平衡目标采样标志

        if self.balance_sampling_by_target:
            self.df_indexed_by_target = []  # 初始化按目标索引的数据集列表

            for target in self.df.get_column('target').unique().to_list():
                df_by_target = self.df.filter(pl.col('target') == target)  # 根据目标过滤数据集
                self.df_indexed_by_target.append(df_by_target)  # 将按目标过滤后的数据集添加到列表中

        # 上下文字符串创建器

        self.context_ds = ContextDataset(
            include_biotypes_metadata_in_context = include_biotypes_metadata_in_context,  # 是否在上下文中包含生物类型元数据
            biotypes_metadata_path = biotypes_metadata_path,  # 生物类型元数据路径
            include_biotypes_metadata_columns = include_biotypes_metadata_columns,  # 包含生物类型元数据列
            biotypes_metadata_delimiter = biotypes_metadata_delimiter  # 生物类型元数据分隔符
        )

    # 返回数据集长度
    def __len__(self):
        if self.balance_sampling_by_target:
            return len(self.df_indexed_by_target)  # 如果按目标平衡采样，则返回按目标索引的数据集长度
        else:
            return len(self.df)  # 否则返回数据集长度
    # 定义特殊方法，用于通过索引获取数据样本
    def __getitem__(self, ind):
        # 如果按目标平衡采样，则从索引数据帧中随机抽取样本
        if self.balance_sampling_by_target:
            # 从按目标索引的数据帧中筛选数据
            filtered_df = self.df_indexed_by_target[ind]
            # 随机选择索引
            rand_ind = randrange(0, len(filtered_df))
            # 获取随机样本
            sample = filtered_df.row(rand_ind)
        else:
            # 否则直接从数据帧中获取样本
            sample = self.df.row(ind)

        # 解包样本数据
        chr_name, begin, end, _, _, _, experiment_target_cell_type, reading, *_ = sample

        # 解析实验、目标和细胞类型
        experiment, target, cell_type = parse_exp_target_cell(experiment_target_cell_type)

        # 获取序列数据
        seq = self.fasta(chr_name, begin, end)
        # 获取氨基酸序列数据
        aa_seq = self.factor_ds[target]
        # 获取上下文字符串数据
        context_str = self.context_ds[cell_type]

        # 将读数转换为张量
        read_value = torch.Tensor([reading])

        # 获取峰值数量
        peaks_nr = self.experiments_index.get(experiment_target_cell_type, 0.)
        # 将峰值数量转换为张量
        peaks_nr = torch.Tensor([peaks_nr])

        # 创建标签张量
        label = torch.Tensor([1.])

        # 返回序列数据、氨基酸序列数据、上下文字符串数据、峰值数量、读数值和标签
        return seq, aa_seq, context_str, peaks_nr, read_value, label
# 为基于保留值的 exp-target-cells 过滤函数

def filter_exp_target_cell(
    arr,
    *,
    exclude_targets = None,  # 排除的目标列表
    include_targets = None,  # 包含的目标列表
    exclude_cell_types = None,  # 排除的细胞类型列表
    include_cell_types = None,  # 包含的细胞类型列表
):
    out = []  # 输出列表

    for el in arr:  # 遍历输入数组
        experiment, target, cell_type = parse_exp_target_cell(el)  # 解析实验、目标和细胞类型

        # 如果包含的目标列表存在且不为空，并且目标不在包含的目标列表中，则跳过
        if exists(include_targets) and len(include_targets) > 0 and target not in include_targets:
            continue

        # 如果排除的目标列表存在且目标在排除的目标列表中，则跳过
        if exists(exclude_targets) and target in exclude_targets:
            continue

        # 如果包含的细胞类型列表存在且不为空，并且细胞类型不在包含的细胞类型列表中，则跳过
        if exists(include_cell_types) and len(include_cell_types) > 0 and cell_type not in include_cell_types:
            continue

        # 如果排除的细胞类型列表存在且细胞类型在排除的细胞类型列表中，则跳过
        if exists(exclude_cell_types) and cell_type in exclude_cell_types:
            continue

        out.append(el)  # 将符合条件的元素添加到输出列表中

    return out  # 返回输出列表


# 为特定 exp-target-celltype 范围的负样本数据集

class ScopedNegativePeakDataset(Dataset):
    def __init__(
        self,
        *,
        fasta_file,
        factor_fasta_folder,
        numpy_folder_with_scoped_negatives,
        exts = '.bed.bool.npy',
        remap_bed_file = None,
        remap_df = None,
        filter_chromosome_ids = None,
        experiments_json_path = None,
        exclude_targets = None,  # 排除的目标列表
        include_targets = None,  # 包含的目标列表
        exclude_cell_types = None,  # 排除的细胞类型列表
        include_cell_types = None,  # 包含的细胞类型列表
        include_biotypes_metadata_in_context = False,
        biotypes_metadata_path = None,
        include_biotypes_metadata_columns = [],
        biotypes_metadata_delimiter = ' | ',
        balance_sampling_by_target = False,
        **kwargs
    # 初始化函数，接受 remap_df 或 remap_bed_file 作为参数
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 断言只能传入 remap_df 或 remap_bed_file 中的一个
        assert exists(remap_df) ^ exists(remap_bed_file), 'either remap bed file or remap dataframe must be passed in'

        # 如果 remap_df 不存在，则从 remap_bed_file 中读取数据
        if not exists(remap_df):
            remap_df = read_bed(remap_bed_file)

        # 初始化 dataset_chr_ids 为全局变量 CHR_IDS
        dataset_chr_ids = CHR_IDS

        # 如果存在 filter_chromosome_ids，则更新 dataset_chr_ids
        if exists(filter_chromosome_ids):
            dataset_chr_ids = dataset_chr_ids.intersection(set(filter_chromosome_ids))

        # 根据 dataset_chr_ids 过滤 remap_df，生成 mask
        filter_map_df = remap_df.with_column(pl.when(pl_isin('column_1', get_chr_names(dataset_chr_ids))).then(True).otherwise(False).alias('mask'))
        mask = filter_map_df.get_column('mask').to_numpy()

        # 统计 mask 中为 True 的数量
        num_scoped_negs = mask.sum()

        # 打印找到的 scoped negative 行数
        print(f'{num_scoped_negs} scoped negative rows found for training')

        # 断言找到的 scoped negative 行数大于 0
        assert num_scoped_negs > 0, 'all remap rows filtered out for scoped negative peak dataset'

        # 设置 self.df 和 self.chromosome_mask
        self.df = remap_df
        self.chromosome_mask = mask

        # 获取 exp-target-cell 到布尔值 numpy 的字典，指示哪些是负样本

        # 获取所有 numpy 文件的路径
        npys_paths = [*Path(numpy_folder_with_scoped_negatives).glob('**/*.npy')]
        exp_target_cell_negatives = [(path.name.rstrip(exts), path) for path in npys_paths]

        # 获取所有 exp_target_cells
        exp_target_cells = [el[0] for el in exp_target_cell_negatives]

        # 根据条件过滤 exp_target_cells
        exp_target_cells = filter_exp_target_cell(
            exp_target_cells,
            include_targets = include_targets,
            exclude_targets = exclude_targets,
            include_cell_types = include_cell_types,
            exclude_cell_types = exclude_cell_types
        )

        # 根据过滤后的 exp_target_cells 过滤 exp_target_cell_negatives
        filtered_exp_target_cell_negatives = list(filter(lambda el: el[0] in exp_target_cells, exp_target_cell_negatives))

        # 设置 self.exp_target_cell_negatives
        self.exp_target_cell_negatives = filtered_exp_target_cell_negatives
        # 断言筛选后的 exp_target_cell_negatives 数量大于 0
        assert len(self.exp_target_cell_negatives) > 0, 'no experiment-target-cell scoped negatives to select from after filtering'

        # 平衡目标采样

        self.balance_sampling_by_target = balance_sampling_by_target

        # 如果需要平衡采样
        if balance_sampling_by_target:
            # 初始化 exp_target_cell_by_target 字典
            self.exp_target_cell_by_target = defaultdict(list)

            # 根据 target 对 exp_target_cell_negatives 进行分组
            for exp_target_cell, filepath in self.exp_target_cell_negatives:
                _, target, *_ = parse_exp_target_cell(exp_target_cell)
                self.exp_target_cell_by_target[target].append((exp_target_cell, filepath))

        # tfactor 数据集

        self.factor_ds = FactorProteinDataset(factor_fasta_folder)

        # 初始化 fasta 文件和 experiments_index
        self.fasta = FastaInterval(fasta_file = fasta_file, **kwargs)
        self.experiments_index = fetch_experiments_index(experiments_json_path)

        # 上下文字符串创建器

        self.context_ds = ContextDataset(
            include_biotypes_metadata_in_context = include_biotypes_metadata_in_context,
            biotypes_metadata_path = biotypes_metadata_path,
            include_biotypes_metadata_columns = include_biotypes_metadata_columns,
            biotypes_metadata_delimiter = biotypes_metadata_delimiter
        )

    # 返回数据集长度
    def __len__(self):
        # 如果需要按目标平衡采样，则返回 exp_target_cell_by_target 的长度
        if self.balance_sampling_by_target:
            return len(self.exp_target_cell_by_target)
        # 否则返回 exp_target_cell_negatives 的长度
        else:
            return len(self.exp_target_cell_negatives)
    # 通过索引获取样本数据
    def __getitem__(self, idx):
        # 如果按目标进行平衡采样
        if self.balance_sampling_by_target:
            # 获取指定索引下的负样本列表
            negatives = list(self.exp_target_cell_by_target.values())[idx]
            # 从负样本列表中随机选择一个样本
            sample = choice(negatives)
        else:
            # 获取指定索引下的负样本
            sample = self.exp_target_cell_negatives[idx]

        # 解析实验、目标和细胞类型
        exp_target_cell, bool_numpy_path = sample
        experiment, target, cell_type = parse_exp_target_cell(exp_target_cell)

        # 加载布尔类型的 numpy 数组，并添加随机噪声
        np_arr = np.load(str(bool_numpy_path))
        np_arr_noised = np_arr.astype(np.float32) + np.random.uniform(low=-1e-1, high=1e-1, size=np_arr.shape[0])

        # 使用染色体掩码进行掩盖
        np_arr_noised *= self.chromosome_mask.astype(np.float32)

        # 选择随机的负峰值
        random_neg_peak_index = np_arr_noised.argmax()

        # 获取染色体名称、起始位置、结束位置和序列
        chr_name, begin, end, *_ = self.df.row(random_neg_peak_index)
        seq = self.fasta(chr_name, begin, end)

        # 获取目标对应的氨基酸序列和细胞类型对应的上下文字符串
        aa_seq = self.factor_ds[target]
        context_str = self.context_ds[cell_type]

        # 获取实验目标细胞对应的峰值数量，并转换为张量
        peaks_nr = self.experiments_index.get(exp_target_cell, 0.)
        peaks_nr = torch.Tensor([peaks_nr])

        # 初始化读取值和标签，并转换为张量
        read_value = torch.Tensor([0.])
        label = torch.Tensor([0.])

        # 返回序列、氨基酸序列、上下文字符串、峰值数量、读取值和标签
        return seq, aa_seq, context_str, peaks_nr, read_value, label
# 定义一个负样本数据集类 NegativePeakDataset，继承自 Dataset 类
class NegativePeakDataset(Dataset):
    # 初始化函数，接收多个参数
    def __init__(
        self,
        *,
        factor_fasta_folder,  # 因子 fasta 文件夹
        negative_bed_file = None,  # 负样本 bed 文件，默认为 None
        remap_bed_file = None,  # 重映射 bed 文件，默认为 None
        remap_df = None,  # 重映射数据框，默认为 None
        negative_df = None,  # 负样本数据框，默认为 None
        filter_chromosome_ids = None,  # 过滤染色体 ID 列表，默认为 None
        exclude_targets = None,  # 排除目标列表，默认为 None
        include_targets = None,  # 包含目标列表，默认为 None
        exclude_cell_types = None,  # 排除细胞类型列表，默认为 None
        include_cell_types = None,  # 包含细胞类型列表，默认为 None
        exp_target_cell_column = 'column_4',  # 实验-目标-细胞列，默认为 'column_4'
        experiments_json_path = None,  # 实验 JSON 路径，默认为 None
        include_biotypes_metadata_in_context = False,  # 在上下文中包含生物类型元数据，默认为 False
        biotypes_metadata_path = None,  # 生物类型元数据路径，默认为 None
        include_biotypes_metadata_columns = [],  # 包含生物类型元数据列，默认为空列表
        biotypes_metadata_delimiter = ' | ',  # 生物类型元数据分隔符，默认为 ' | '
        balance_sampling_by_target = False,  # 按目标平衡采样，默认为 False
        **kwargs  # 其他关键字参数
    ):
        super().__init__()  # 调用父类的初始化函数
        # 断言语句，判断 remap_df 和 remap_bed_file 必须有一个存在
        assert exists(remap_df) ^ exists(remap_bed_file), 'either remap bed file or remap dataframe must be passed in'
        # 断言语句，判断 negative_df 和 negative_bed_file 必须有一个存在
        assert exists(negative_df) ^ exists(negative_bed_file), 'either negative bed file or negative dataframe must be passed in'

        # 如果 remap_df 不存在，则从 remap_bed_file 读取数据框
        if not exists(remap_df):
            remap_df = read_bed(remap_bed_file)

        # 如果 negative_df 不存在，则从 negative_bed_file 读取数据框
        neg_df = negative_df
        if not exists(negative_df):
            neg_df = read_bed(negative_bed_file)

        # 过滤 remap 数据框
        remap_df = filter_df_by_tfactor_fastas(remap_df, factor_fasta_folder)

        # 设置数据集的染色体 ID
        dataset_chr_ids = CHR_IDS

        # 如果存在过滤染色体 ID，则更新数据集的染色体 ID
        if exists(filter_chromosome_ids):
            dataset_chr_ids = dataset_chr_ids.intersection(set(filter_chromosome_ids))

        # 根据染色体名过滤负样本数据框
        neg_df = neg_df.filter(pl_isin('column_1', get_chr_names(dataset_chr_ids)))

        # 断言语句，确保负样本数据框不为空
        assert len(neg_df) > 0, 'dataset is empty by filter criteria'

        self.neg_df = neg_df  # 设置负样本数据框

        # 获取所有实验-目标-细胞，并根据条件过滤
        exp_target_cells = remap_df.get_column(exp_target_cell_column).unique().to_list()

        self.filtered_exp_target_cells = filter_exp_target_cell(
            exp_target_cells,
            include_targets = include_targets,
            exclude_targets = exclude_targets,
            include_cell_types = include_cell_types,
            exclude_cell_types = exclude_cell_types
        )

        # 断言语句，确保还有实验-目标-细胞用于硬负样本集
        assert len(self.filtered_exp_target_cells), 'no experiment-target-cell left for hard negative set'

        # 如果需要按目标平衡采样
        self.balance_sampling_by_target = balance_sampling_by_target

        if balance_sampling_by_target:
            self.exp_target_cell_by_target = defaultdict(list)

            # 根据目标将实验-目标-细胞分组
            for exp_target_cell in self.filtered_exp_target_cells:
                _, target, *_ = parse_exp_target_cell(exp_target_cell)
                self.exp_target_cell_by_target[target].append(exp_target_cell)

        # 因子数据集
        self.factor_ds = FactorProteinDataset(factor_fasta_folder)
        self.fasta = FastaInterval(**kwargs)

        # 获取实验索引
        self.experiments_index = fetch_experiments_index(experiments_json_path)

        # 上下文字符串创建器
        self.context_ds = ContextDataset(
            include_biotypes_metadata_in_context = include_biotypes_metadata_in_context,
            biotypes_metadata_path = biotypes_metadata_path,
            include_biotypes_metadata_columns = include_biotypes_metadata_columns,
            biotypes_metadata_delimiter = biotypes_metadata_delimiter
        )

    # 返回负样本数据集的长度
    def __len__(self):
        return len(self.neg_df)
    # 重载 __getitem__ 方法，用于获取指定索引处的数据
    def __getitem__(self, ind):
        # 从 neg_df 数据框中获取指定索引处的染色体名称、起始位置和终止位置
        chr_name, begin, end = self.neg_df.row(ind)

        # 如果按目标平衡采样
        if self.balance_sampling_by_target:
            # 从 exp_target_cell_by_target 字典中随机选择一个目标细胞类型
            rand_ind = randrange(0, len(self.exp_target_cell_by_target))
            exp_target_cell_by_target_list = list(self.exp_target_cell_by_target.values())
            random_exp_target_cell_type = choice(exp_target_cell_by_target_list[rand_ind])
        else:
            # 从 filtered_exp_target_cells 列表中随机选择一个目标细胞类型
            random_exp_target_cell_type = choice(self.filtered_exp_target_cells)

        # 解析实验、目标和细胞类型
        experiment, target, cell_type = parse_exp_target_cell(random_exp_target_cell_type)

        # 获取指定染色体区间的序列
        seq = self.fasta(chr_name, begin, end)
        # 获取目标对应的氨基酸序列
        aa_seq = self.factor_ds[target]
        # 获取细胞类型对应的上下文字符串
        context_str = self.context_ds[cell_type]

        # 初始化读取值为 0 的张量
        read_value = torch.Tensor([0.])

        # 获取指定目标细胞类型的峰值数
        peaks_nr = self.experiments_index.get(random_exp_target_cell_type, 0.)
        # 将峰值数转换为张量
        peaks_nr = torch.Tensor([peaks_nr])

        # 初始化标签为 0 的张量
        label = torch.Tensor([0.])

        # 返回获取的序列、氨基酸序列、上下文字符串、峰值数、读取值和标签
        return seq, aa_seq, context_str, peaks_nr, read_value, label
# dataloader相关函数

# 将数据集中的数据按照不同的类型解压缩
def collate_fn(data):
    seq, aa_seq, context_str, peaks_nr, read_values, labels = list(zip(*data))
    return torch.stack(seq), tuple(aa_seq), tuple(context_str), torch.stack(peaks_nr, dim=0), torch.stack(read_values, dim=0), torch.cat(labels, dim=0)

# 将多个dataloader的输出合并为一个元组
def collate_dl_outputs(*dl_outputs):
    outputs = list(zip(*dl_outputs))
    ret = []
    for entry in outputs:
        if isinstance(entry[0], torch.Tensor):
            entry = torch.cat(entry, dim=0)
        else:
            entry = (sub_el for el in entry for sub_el in el)
        ret.append(entry)
    return tuple(ret)

# 无限循环生成dataloader中的数据
def cycle(loader):
    while True:
        for data in loader:
            yield data

# 获取dataloader对象
def get_dataloader(ds, cycle_iter=False, **kwargs):
    dataset_len = len(ds)
    batch_size = kwargs.get('batch_size')
    drop_last = dataset_len > batch_size

    # 创建DataLoader对象
    dl = DataLoader(ds, collate_fn=collate_fn, drop_last=drop_last, **kwargs)
    wrapper = cycle if cycle_iter else iter
    return wrapper(dl)
```