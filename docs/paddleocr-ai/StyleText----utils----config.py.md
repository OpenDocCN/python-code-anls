# `.\PaddleOCR\StyleText\utils\config.py`

```py
# 版权声明
# 2020年PaddlePaddle作者保留所有权利。
#
# 根据Apache许可证2.0版（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“原样”基础分发，没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和
# 许可证下的限制
import yaml
import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter

# 递归替换字典或列表中的值
def override(dl, ks, v):
    """
    Recursively replace dict of list

    Args:
        dl(dict or list): dict or list to be replaced
        ks(list): list of keys
        v(str): value to be replaced
    """

    # 将字符串转换为数字
    def str2num(v):
        try:
            return eval(v)
        except Exception:
            return v

    # 断言dl是列表或字典
    assert isinstance(dl, (list, dict)), ("{} should be a list or a dict")
    # 断言键列表长度大于0
    assert len(ks) > 0, ('lenght of keys should larger than 0')
    if isinstance(dl, list):
        k = str2num(ks[0])
        if len(ks) == 1:
            # 断言索引在范围内
            assert k < len(dl), ('index({}) out of range({})'.format(k, dl))
            dl[k] = str2num(v)
        else:
            override(dl[k], ks[1:], v)
    else:
        if len(ks) == 1:
            # 如果键不存在于字典中，则发出警告
            if not ks[0] in dl:
                logger.warning('A new filed ({}) detected!'.format(ks[0], dl))
            dl[ks[0]] = str2num(v)
        else:
            # 断言键存在于字典中
            assert ks[0] in dl, (
                '({}) doesn\'t exist in {}, a new dict field is invalid'.
                format(ks[0], dl))
            override(dl[ks[0]], ks[1:], v)

# 递归覆盖配置
def override_config(config, options=None):
    """
    Recursively override the config
    Args:
        config(dict): dict to be replaced
        options(list): list of pairs(key0.key1.idx.key2=value)
            such as: [
                'topk=2',
                'VALID.transforms.1.ResizeImage.resize_short=300'
            ]

    Returns:
        config(dict): replaced config
    """
    # 如果选项不为空，则遍历选项列表
    if options is not None:
        for opt in options:
            # 检查选项是否为字符串类型
            assert isinstance(opt, str), (
                "option({}) should be a str".format(opt))
            # 检查选项中是否包含等号
            assert "=" in opt, (
                "option({}) should contain a ="
                "to distinguish between key and value".format(opt))
            # 通过等号分割键值对
            pair = opt.split('=')
            # 检查键值对是否只有一个等号
            assert len(pair) == 2, ("there can be only a = in the option")
            key, value = pair
            # 通过点号分割键
            keys = key.split('.')
            # 调用 override 函数替换配置中的键值对
            override(config, keys, value)

    # 返回替换后的配置
    return config
# 定义一个参数解析器类，继承自ArgumentParser类
class ArgsParser(ArgumentParser):
    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法，设置formatter_class为RawDescriptionHelpFormatter
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        # 添加参数-c/--config，用于指定配置文件
        self.add_argument("-c", "--config", help="configuration file to use")
        # 添加参数-t/--tag，默认为"0"，用于标记工作进程
        self.add_argument(
            "-t", "--tag", default="0", help="tag for marking worker")
        # 添加参数-o/--override，设置为可重复出现的参数，用于覆盖配置选项
        self.add_argument(
            '-o',
            '--override',
            action='append',
            default=[],
            help='config options to be overridden')
        # 添加参数--style_image，默认为"examples/style_images/1.jpg"，用于标记工作进程
        self.add_argument(
            "--style_image", default="examples/style_images/1.jpg", help="tag for marking worker")
        # 添加参数--text_corpus，默认为"PaddleOCR"，用于标记工作进程
        self.add_argument(
            "--text_corpus", default="PaddleOCR", help="tag for marking worker")
        # 添加参数--language，默认为"en"，用于标记工作进程
        self.add_argument(
            "--language", default="en", help="tag for marking worker")

    # 解析参数的方法
    def parse_args(self, argv=None):
        # 调用父类的parse_args方法解析参数
        args = super(ArgsParser, self).parse_args(argv)
        # 断言config参数不为None，否则抛出异常
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        return args


# 从yml/yaml文件中加载配置
def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: config
    """
    # 获取文件路径的扩展名
    ext = os.path.splitext(file_path)[1]
    # 断言扩展名为.yml或.yaml，否则抛出异常
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    # 以二进制读取文件，使用yaml.Loader加载配置
    with open(file_path, 'rb') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    return config


# 生成配置文件
def gen_config():
    # 打开config.yml文件，写入base_config的内容
    with open("config.yml", "w") as f:
        yaml.dump(base_config, f)


# 如果作为主程序运行，则调用gen_config方法
if __name__ == '__main__':
    gen_config()
```