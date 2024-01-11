# `arknights-mower\arknights_mower\utils\conf.py`

```
# 导入所需的模块
import os
import json
from pathlib import Path
from ruamel import yaml
from flatten_dict import flatten, unflatten
from .. import __rootdir__


# 从模板文件中获取临时配置
def __get_temp_conf():
    with Path(f'{__rootdir__}/templates/conf.yml').open('r', encoding='utf8') as f:
        return yaml.load(f,Loader=yaml.Loader)


# 将配置保存到指定路径的文件中
def save_conf(conf, path="./conf.yml"):
    with Path(path).open('w', encoding='utf8') as f:
        yaml.dump(conf, f, allow_unicode=True)


# 从指定路径的文件中加载配置
def load_conf(path="./conf.yml"):
    # 获取临时配置
    temp_conf = __get_temp_conf()
    # 如果文件不存在，则创建空配置文件并保存临时配置
    if not os.path.isfile(path):
        open(path, 'w')  # 创建空配置文件
        save_conf(temp_conf, path)
        return temp_conf
    else:
        # 从文件中加载配置
        with Path(path).open('r', encoding='utf8') as c:
            conf = yaml.load(c, Loader=yaml.Loader)
            if conf is None:
                conf = {}
            # 将临时配置和文件中的配置合并
            flat_temp = flatten(temp_conf)
            flat_conf = flatten(conf)
            flat_temp.update(flat_conf)
            temp_conf = unflatten(flat_temp)
            return temp_conf


# 从模板文件中获取临时计划
def __get_temp_plan():
    with open(f'{__rootdir__}/templates/plan.json', 'r') as f:
        return json.loads(f.read())


# 从指定路径的文件中加载计划
def load_plan(path="./plan.json"):
    # 获取临时计划
    temp_plan = __get_temp_plan()
    # 如果文件不存在，则创建空json文件并保存临时计划
    if not os.path.isfile(path):
        with open(path, 'w') as f:
            json.dump(temp_plan, f)  # 创建空json文件
        return temp_plan
    # 从文件中加载计划
    with open(path, 'r', encoding='utf8') as fp:
        plan = json.loads(fp.read())
        if 'conf' not in plan.keys():  # 兼容旧版本
            temp_plan['plan1'] = plan
            return temp_plan
        # 获取新版本的计划
        tmp = temp_plan['conf']
        tmp.update(plan['conf'])
        plan['conf'] = tmp
        return plan


# 将计划写入到指定路径的文件中
def write_plan(plan, path="./plan.json"):
    with open(path, 'w', encoding='utf8') as c:
        json.dump(plan, c, ensure_ascii=False)
```