# `.\lucidrains\alphafold2\scripts\refinement.py`

```py
# 导入所需的库和模块
import os
import json
import warnings
# 科学计算库
import numpy as np
# 尝试导入 pyrosetta 模块，如果导入失败则发出警告
try: 
    import pyrosetta
except ModuleNotFoundError:
    msg = "Unable to find an existing installation of the PyRosetta module. " +\
          "Functions involving this module such as the FastRelax pipeline " +\
          "will not work."
    warnings.warn(msg) # no pyRosetta was found


#####################
### ROSETTA STUFF ###
#####################


def pdb2rosetta(route):
    """ Takes pdb file route(s) as input and returns rosetta pose(s). 
        Input:
        * route: list or string.
        Output: list of 1 or many according to input
       """
    # 如果输入是字符串，则返回包含单个 rosetta pose 的列表
    if isinstance(route, str):
        return [pyrosetta.io.pose_from_pdb(route)]
    else:
        return list(pyrosetta.io.poses_from_files(route))

def rosetta2pdb(pose, route, verbose=True):
    """ Takes pose(s) as input and saves pdb(s) to disk.
        Input:
        * pose: list or string. rosetta poses object(s).
        * route: list or string. destin filenames to be written.
        * verbose: bool. warns if lengths dont match and @ every write.
        Inspo:
        * https://www.rosettacommons.org/demos/latest/tutorials/input_and_output/input_and_output#controlling-output_common-structure-output-files_pdb-file
        * https://graylab.jhu.edu/PyRosetta.documentation/pyrosetta.rosetta.core.io.pdb.html#pyrosetta.rosetta.core.io.pdb.dump_pdb
    """
    # 将输入转换为列表
    pose  = [pose] if isinstance(pose, str) else pose
    route = [route] if isinstance(route, str) else route
    # 检查长度是否匹配，如果不匹配则发出警告
    if verbose and ( len(pose) != len(route) ):
        print("Length of pose and route are not the same. Will stop at the minimum.")
    # 转换并保存
    for i,pos in enumerate(pose):
        pyrosetta.rosetta.core.io.pdb.dump_pdb(pos, route[i])
        if verbose:
            print("Saved structure @ "+route)
    return

def run_fast_relax(config_route, pdb_route=None, pose=None):
    """ Runs the Fast-Relax pipeline.
        * config_route: route to json file with config
        * pose: rosetta pose to run the pipeline on
        Output: rosetta pose
    """
    # 加载 rosetta pose - 如果传入字符串或列表，则转换为 pose + 重新调用
    if isinstance(pdb_route, str):
        pose = pdb2rosetta(pdb_route)
        return run_fast_relax(config, pose=pose)
    elif isinstance(pdb_route, list):
        return [run_fast_relax(config, pdb_route=pdb) for pdb in pdb_route]
    # 加载配置文件
    config = json.load(config_route)
    # 运行 Fast-Relax pipeline - 示例:
    # https://colab.research.google.com/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.02-Packing-design-and-regional-relax.ipynb#scrollTo=PYr025Rn1Q8i
    # https://nbviewer.jupyter.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.03-Design-with-a-resfile-and-relax.ipynb
    # https://faculty.washington.edu/dimaio/files/demo2.py
    raise NotImplementedError("Last step. Not implemented yet.")
```