# `.\lucidrains\tf-bind-transformer\scripts\download_experiments.py`

```
# 导入所需的模块
import json
import tqdm
import requests

# 定义 NCBI_TAX_ID 字典，包含人类和小鼠的分类号
NCBI_TAX_ID = dict(
    human = 9606,
    mouse = 10090
)

# 设置 SPECIES 变量为 'human'
SPECIES = 'human'
# 设置 API_URL 变量为 API 的基本 URL
API_URL = 'https://remap.univ-amu.fr/api/v1/'

# 定义函数 get_json，用于获取 JSON 数据
def get_json(url, params = dict()):
    # 设置请求头
    headers = dict(Accept = 'application/json')
    # 发起 GET 请求
    resp = requests.get(url, params = params, headers = headers)
    # 返回 JSON 数据
    return resp.json()

# 定义函数 get_experiments，用于获取实验数据
def get_experiments(species):
    # 检查物种是否在 NCBI_TAX_ID 中
    assert species in NCBI_TAX_ID
    # 获取对应物种的分类号
    taxid = NCBI_TAX_ID[species]
    # 获取实验数据
    experiments = get_json(f'{API_URL}list/experiments/taxid={taxid}')
    return experiments

# 定义函数 get_experiment，用于获取特定实验的详细信息
def get_experiment(experiment_id, species):
    # 检查物种是否在 NCBI_TAX_ID 中
    assert species in NCBI_TAX_ID
    # 获取对应物种的分类号
    taxid = NCBI_TAX_ID[species]
    # 获取特定实验的详细信息
    experiment = get_json(f'http://remap.univ-amu.fr/api/v1/info/byExperiment/experiment={experiment_id}&taxid={taxid}')
    return experiment

# 获取指定物种的实验数据
experiments = get_experiments(SPECIES)

# 遍历实验数据列表，并获取每个实验的详细信息
for experiment in tqdm.tqdm(experiments['experiments']):
    experiment_details = get_experiment(experiment['accession'], SPECIES)
    experiment['details'] = experiment_details

# 将实验数据写入 JSON 文件
with open('data/experiments.json', 'w+') as f:
    contents = json.dumps(experiments, indent = 4, sort_keys = True)
    f.write(contents)

# 打印成功信息
print('success')
```