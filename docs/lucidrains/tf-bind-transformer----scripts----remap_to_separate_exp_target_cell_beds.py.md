# `.\lucidrains\tf-bind-transformer\scripts\remap_to_separate_exp_target_cell_beds.py`

```
# 导入必要的库
import polars as pl
from pathlib import Path
from tf_bind_transformer.data import read_bed, save_bed

# 定义函数，用于生成分离的实验目标细胞类型的 BED 文件
def generate_separate_exp_target_cell_beds(
    remap_file,
    *,
    output_folder = './negative-peaks-per-target',
    exp_target_cell_type_col = 'column_4'
):
    # 将输出文件夹路径转换为 Path 对象，并确保文件夹存在
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok = True, parents = True)

    # 读取 remap 文件内容到 DataFrame
    df = read_bed(remap_file)
    # 获取目标实验的唯一值列表
    target_experiments = df.get_column(exp_target_cell_type_col).unique().to_list()

    # 遍历每个目标实验
    for target_experiment in target_experiments:
        # 根据目标实验筛选 DataFrame
        filtered_df = df.filter(pl.col(exp_target_cell_type_col) == target_experiment)

        # 构建目标实验的 BED 文件路径
        target_bed_path = str(output_folder / f'{target_experiment}.bed')
        # 保存筛选后的 DataFrame 到 BED 文件
        save_bed(filtered_df, target_bed_path)

    # 打印成功信息
    print('success')
```