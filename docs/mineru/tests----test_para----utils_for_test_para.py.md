# `.\MinerU\tests\test_para\utils_for_test_para.py`

```
# 导入操作系统模块，用于路径和文件操作
import os

# 定义一个名为 UtilsForTestPara 的类
class UtilsForTestPara:
    # 初始化方法
    def __init__(self):
        # 获取当前文件目录的绝对路径
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取当前目录的父目录
        parent_dir = os.path.dirname(curr_dir)
        # 构建 assets 目录的路径
        assets_dir = os.path.join(parent_dir, "assets")
        # 构建默认的预处理结果输出目录的路径
        self.default_pre_proc_out_dir = os.path.join(assets_dir, "pre_proc_results")

        # 如果 assets 目录不存在，抛出文件未找到异常
        if not os.path.exists(assets_dir):
            raise FileNotFoundError("The assets directory does not exist. Please check the path.")

    # 读取指定目录下的所有 preproc_out.json 文件
    def read_preproc_out_jfiles(self, input_dir=None):
        """
        读取输入目录下所有的 preproc_out.json 文件

        参数
        ----------
        input_dir : str
            包含 preproc_out.json 文件的目录。
            默认使用 default_pre_proc_out_dir。

        返回
        -------
        preproc_out_jsons : list
            preproc_out.json 文件的路径列表。

        """
        # 如果未提供输入目录，使用默认的预处理结果输出目录
        if input_dir is None:
            input_dir = self.default_pre_proc_out_dir

        # 初始化存储预处理输出 JSON 文件路径的列表
        preproc_out_jsons = []
        # 遍历输入目录及其子目录中的文件
        for root, dirs, files in os.walk(input_dir):
            # 遍历当前目录下的文件
            for file in files:
                # 如果文件名以 preproc_out.json 结尾
                if file.endswith("preproc_out.json"):
                    # 构建文件的绝对路径
                    preproc_out_json_abs_path = os.path.join(root, file)
                    # 将文件路径添加到列表中
                    preproc_out_jsons.append(preproc_out_json_abs_path)

        # 返回所有找到的 preproc_out.json 文件路径
        return preproc_out_jsons

# 如果当前模块是主模块
if __name__ == "__main__":
    # 创建 UtilsForTestPara 类的实例
    utils = UtilsForTestPara()
    # 读取默认目录下的 preproc_out.json 文件路径
    preproc_out_jsons = utils.read_preproc_out_jfiles()
    # 遍历并打印每个 JSON 文件的路径
    for preproc_out_json in preproc_out_jsons:
        print(preproc_out_json)
```