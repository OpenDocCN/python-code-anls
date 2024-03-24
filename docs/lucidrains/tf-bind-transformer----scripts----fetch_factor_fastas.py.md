# `.\lucidrains\tf-bind-transformer\scripts\fetch_factor_fastas.py`

```py
# 导入所需的库
import requests
from pathlib import Path
import click
import polars as pl
from tqdm import tqdm
from tf_bind_transformer.gene_utils import parse_gene_name
from tf_bind_transformer.data import read_bed

# 常量定义

# Uniprot 数据库的 URL
UNIPROT_URL = 'http://www.uniprot.org'

# 默认的 remap 文件路径
DEFAULT_REMAP_PATH = dict(
    HUMAN = './remap2022_crm_macs2_hg38_v1_0.bed',
    MOUSE = './remap2022_crm_macs2_mm10_v1_0.bed',
)

# 用于覆盖基因名到 Uniprot ID 的映射
GENE_NAME_TO_ID_OVERRIDE = {
    'SS18-SSX': ['Q8IZH1'],
    'TFIIIC': ['A6ZV34']        # 待办事项: 找出人类条目在 Uniprot 中的位置
}

# 辅助函数

# 根据给定的类型和标识符，获取 Uniprot 映射
def uniprot_mapping(fromtype, totype, identifier):
    params = {
        'from': fromtype,
        'to': totype,
        'format': 'tab',
        'query': identifier,
    }

    response = requests.get(f'{UNIPROT_URL}/mapping', params = params)
    return response.text

# 主要函数

# 命令行入口函数
@click.command()
@click.option('--species', help = 'Species', default = 'human', type = click.Choice(['human', 'mouse']))
@click.option('--remap-bed-path', help = 'Path to species specific remap file')
@click.option('--fasta-folder', help = 'Path to factor fastas', default = './tfactor.fastas')
def fetch_factors(
    species,
    remap_bed_path,
    fasta_folder
):
    species = species.upper()

    # 如果未提供 remap-bed-path，则使用默认路径
    if remap_bed_path is None:
        remap_bed_path = DEFAULT_REMAP_PATH[species]

    remap_bed_path = Path(remap_bed_path)

    # 检查 remap 文件是否存在
    assert remap_bed_path.exists(), f'remap file does not exist at {str(remap_bed_path)}'

    # 加载 bed 文件并从第三列获取所有唯一的目标
    df = read_bed(remap_bed_path)
    genes = set([target for targets in df[:, 3] for target in targets.split(',')])

    print(f'{len(genes)} factors found')

    # 加载所有保存的 fasta 文件，以便可以优雅地恢复
    fasta_files = [str(path) for path in Path('./').glob('*.fasta')]
    processed_genes = set([*map(lambda t: str(t).split('.')[0], fasta_files)])

    results_folder = Path(fasta_folder)
    results_folder.mkdir(exist_ok = True, parents = True)

    # 遍历基因并处理
    for unparsed_gene_name in tqdm(genes):
        for gene_name in parse_gene_name(unparsed_gene_name):

            if gene_name in processed_genes:
                continue

            # 根据基因名获取 Uniprot ID
            if gene_name not in GENE_NAME_TO_ID_OVERRIDE:
                uniprot_resp = uniprot_mapping('GENENAME', 'ID', gene_name)

                # 仅获取人类的条目（待办事项: 使其与物种无关）
                entries = list(filter(lambda t: f'_{species}' in t, uniprot_resp.split('\n')))
                entries = list(map(lambda t: t.split('\t')[1], entries))
            else:
                entries = GENE_NAME_TO_ID_OVERRIDE[gene_name]

            if len(entries) == 0:
                print(f'no entries found for {gene_name}')
                continue

            # 保存所有结果
            for entry in entries:
                response = requests.get(f'{UNIPROT_URL}/uniprot/{entry}.fasta')

                if response.status_code != 200:
                    print(f'<{response.status_code}> error fetching fasta file from gene {gene_name} {entry}')
                    continue

                fasta_path = str(results_folder / f'{gene_name}.{entry}.fasta')

                with open(fasta_path, 'w') as f:
                    f.write(response.text)

            print(f'gene {gene_name} written')

# 执行主函数
if __name__ == '__main__':
    fetch_factors()
```