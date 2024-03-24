# `.\lucidrains\tr-rosetta-pytorch\tr_rosetta_pytorch\cli.py`

```py
# 导入必要的库
import fire
import torch
import tarfile
import numpy as np
from pathlib import Path

# 导入自定义模块
from tr_rosetta_pytorch.tr_rosetta_pytorch import trRosettaNetwork
from tr_rosetta_pytorch.utils import preprocess, d

# 定义路径常量
CURRENT_PATH = Path(__file__).parent
DEFAULT_MODEL_PATH = CURRENT_PATH / 'models'
MODEL_PATH =  DEFAULT_MODEL_PATH / 'models.tar.gz'
MODEL_FILES = [*Path(DEFAULT_MODEL_PATH).glob('*.pt')]

# 如果模型文件未解压，则解压
if len(MODEL_FILES) == 0:
    tar = tarfile.open(str(MODEL_PATH))
    tar.extractall(DEFAULT_MODEL_PATH)
    tar.close()

# 预测函数
@torch.no_grad()
def get_ensembled_predictions(input_file, output_file=None, model_dir=DEFAULT_MODEL_PATH):
    # 创建 trRosettaNetwork 实例
    net = trRosettaNetwork()
    # 预处理输入文件
    i = preprocess(input_file)

    # 如果未指定输出文件，则根据输入文件生成默认输出文件名
    if output_file is None:
        input_path = Path(input_file)
        output_file = f'{input_path.parents[0] / input_path.stem}.npz'

    outputs = []
    model_files = [*Path(model_dir).glob('*.pt')]

    # 如果找不到模型文件，则抛出异常
    if len(model_files) == 0:
        raise 'No model files can be found'

    # 遍历模型文件，加载模型并进行预测
    for model_file in model_files:
        net.load_state_dict(torch.load(model_file, map_location=torch.device(d())))
        net.to(d()).eval()
        output = net(i)
        outputs.append(output)

    # 对模型输出进行平均处理
    averaged_outputs = [torch.stack(model_output).mean(dim=0).cpu().numpy().squeeze(0).transpose(1,2,0) for model_output in zip(*outputs)]
    # 创建包含预测结果的字典
    output_dict = dict(zip(['theta', 'phi', 'dist', 'omega'], averaged_outputs))
    # 保存预测结果到输出文件
    np.savez_compressed(output_file, **output_dict)
    print(f'predictions for {input_file} saved to {output_file}')

# 定义命令行接口
def predict():
    fire.Fire(get_ensembled_predictions)
```