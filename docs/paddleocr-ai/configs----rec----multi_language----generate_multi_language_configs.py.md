# `.\PaddleOCR\configs\rec\multi_language\generate_multi_language_configs.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 基于"按原样"的基础分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入 yaml 模块
import yaml
# 从 argparse 模块导入 ArgumentParser 和 RawDescriptionHelpFormatter
from argparse import ArgumentParser, RawDescriptionHelpFormatter
# 导入 os.path 模块
import os.path
# 导入 logging 模块
import logging
# 配置日志级别为 INFO
logging.basicConfig(level=logging.INFO)

# 支持的语言列表，键为语言代码，值为语言名称
support_list = {
    'it': 'italian',
    'xi': 'spanish',
    'pu': 'portuguese',
    'ru': 'russian',
    'ar': 'arabic',
    'ta': 'tamil',
    'ug': 'uyghur',
    'fa': 'persian',
    'ur': 'urdu',
    'rs': 'serbian latin',
    'oc': 'occitan',
    'rsc': 'serbian cyrillic',
    'bg': 'bulgarian',
    'uk': 'ukranian',
    'be': 'belarusian',
    'te': 'telugu',
    'ka': 'kannada',
    'chinese_cht': 'chinese tradition',
    'hi': 'hindi',
    'mr': 'marathi',
    'ne': 'nepali',
}

# 拉丁语言列表
latin_lang = [
    'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
    'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
    'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
    'sw', 'tl', 'tr', 'uz', 'vi', 'latin'
]
# 阿拉伯语言列表
arabic_lang = ['ar', 'fa', 'ug', 'ur']
# 西里尔语言列表
cyrillic_lang = [
    'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
    'dar', 'inh', 'che', 'lbe', 'lez', 'tab', 'cyrillic'
]
# 梵文语言列表
devanagari_lang = [
    'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
    'sa', 'bgc', 'devanagari'
]
# 多语言列表，包括拉丁语、阿拉伯语、西里尔语和梵文语
multi_lang = latin_lang + arabic_lang + cyrillic_lang + devanagari_lang
# 检查是否存在名为"rec_multi_language_lite_train.yml"的文件，如果不存在则抛出异常
assert (os.path.isfile("./rec_multi_language_lite_train.yml")
        ), "Loss basic configuration file rec_multi_language_lite_train.yml.\
You can download it from \
https://github.com/PaddlePaddle/PaddleOCR/tree/dygraph/configs/rec/multi_language/"

# 从文件中加载全局配置信息
global_config = yaml.load(
    open("./rec_multi_language_lite_train.yml", 'rb'), Loader=yaml.Loader)
# 获取项目路径的绝对路径
project_path = os.path.abspath(os.path.join(os.getcwd(), "../../../"))

# 定义参数解析类，继承自ArgumentParser
class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        # 添加参数选项"-o"，用于设置配置选项
        self.add_argument(
            "-o", "--opt", nargs='+', help="set configuration options")
        # 添加参数选项"-l"，用于设置语言类型，支持的语言类型由support_list决定
        self.add_argument(
            "-l",
            "--language",
            nargs='+',
            help="set language type, support {}".format(support_list))
        # 添加参数选项"--train"，用于更改训练数据集的默认路径
        self.add_argument(
            "--train",
            type=str,
            help="you can use this command to change the train dataset default path"
        )
        # 添加参数选项"--val"，用于更改评估数据集的默认路径
        self.add_argument(
            "--val",
            type=str,
            help="you can use this command to change the eval dataset default path"
        )
        # 添加参数选项"--dict"，用于更改字典的默认路径
        self.add_argument(
            "--dict",
            type=str,
            help="you can use this command to change the dictionary default path"
        )
        # 添加参数选项"--data_dir"，用于更改数据集的默认根路径
        self.add_argument(
            "--data_dir",
            type=str,
            help="you can use this command to change the dataset default root path"
        )

    # 重写parse_args方法，解析参数并返回结果
    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        # 解析配置选项
        args.opt = self._parse_opt(args.opt)
        # 设置语言类型
        args.language = self._set_language(args.language)
        return args

    # 解析配置选项
    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=')
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config
    # 设置语言类型
    def _set_language(self, type):
        # 获取语言类型的第一个字符
        lang = type[0]
        # 断言语言类型不为空
        assert (type), "please use -l or --language to choose language type"
        # 断言语言类型在支持列表中或者是多语言列表中
        assert(
                lang in support_list.keys() or lang in multi_lang
               ),"the sub_keys(-l or --language) can only be one of support list: \n{},\nbut get: {}, " \
                 "please check your running command".format(multi_lang, type)
        # 根据语言类型设置为对应的语言
        if lang in latin_lang:
            lang = "latin"
        elif lang in arabic_lang:
            lang = "arabic"
        elif lang in cyrillic_lang:
            lang = "cyrillic"
        elif lang in devanagari_lang:
            lang = "devanagari"
        # 设置全局配置中的字符字典路径
        global_config['Global'][
            'character_dict_path'] = 'ppocr/utils/dict/{}_dict.txt'.format(lang)
        # 设置全局配置中的模型保存路径
        global_config['Global'][
            'save_model_dir'] = './output/rec_{}_lite'.format(lang)
        # 设置全局配置中训练数据集的标签文件列表
        global_config['Train']['dataset'][
            'label_file_list'] = ["train_data/{}_train.txt".format(lang)]
        # 设置全局配置中评估数据集的标签文件列表
        global_config['Eval']['dataset'][
            'label_file_list'] = ["train_data/{}_val.txt".format(lang)]
        # 设置全局配置中的字符类型
        global_config['Global']['character_type'] = lang
        # 断言默认字典文件存在
        assert (
            os.path.isfile(
                os.path.join(project_path, global_config['Global'][
                    'character_dict_path']))
        ), "Loss default dictionary file {}_dict.txt.You can download it from \
# 根据给定的语言生成 GitHub 上 PaddleOCR 项目中的字典文件链接
def lang_link(lang):
    link = "https://github.com/PaddlePaddle/PaddleOCR/tree/dygraph/ppocr/utils/dict/".format(lang)
    return lang

# 将传入的配置合并到全局配置中
def merge_config(config):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    遍历传入的配置字典
    for key, value in config.items():
        如果键中不包含"."，则直接更新全局配置中的对应键值对
        if "." not in key:
            如果值是字典类型且键已存在于全局配置中，则更新对应键的值
            if isinstance(value, dict) and key in global_config:
                global_config[key].update(value)
            否则直接添加新的键值对到全局配置中
            else:
                global_config[key] = value
        否则，处理包含"."的键
        else:
            将键按"."分割
            sub_keys = key.split('.')
            确保第一个子键存在于全局配置中
            assert (
                sub_keys[0] in global_config
            ), "the sub_keys can only be one of global_config: {}, but get: {}, please check your running command".format(
                global_config.keys(), sub_keys[0])
            逐级更新全局配置中的子键值
            cur = global_config[sub_keys[0]
            for idx, sub_key in enumerate(sub_keys[1:]):
                如果是倒数第二个子键，则更新其值
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                否则继续向下一级子键更新
                else:
                    cur = cur[sub_key]

# 检查指定路径的文件是否存在
def loss_file(path):
    确保指定路径的文件存在
    assert (
        os.path.exists(path)
    ), "There is no such file:{},Please do not forget to put in the specified file".format(
        path)

如果作为主程序运行
if __name__ == '__main__':
    解析命令行参数
    FLAGS = ArgsParser().parse_args()
    将命令行参数中的配置合并到全局配置中
    merge_config(FLAGS.opt)
    生成保存文件路径
    save_file_path = 'rec_{}_lite_train.yml'.format(FLAGS.language)
    如果保存文件路径已存在，则删除
    if os.path.isfile(save_file_path):
        os.remove(save_file_path)

    如果需要训练
    if FLAGS.train:
        更新全局配置中训练数据集的标签文件列表
        global_config['Train']['dataset']['label_file_list'] = [FLAGS.train]
        获取训练标签文件路径
        train_label_path = os.path.join(project_path, FLAGS.train)
        检查训练标签文件是否存在
        loss_file(train_label_path)
    如果需要验证
    if FLAGS.val:
        更新全局配置中验证数据集的标签文件列表
        global_config['Eval']['dataset']['label_file_list'] = [FLAGS.val]
        获取验证标签文件路径
        eval_label_path = os.path.join(project_path, FLAGS.val)
        检查验证标签文件是否存在
        loss_file(eval_label_path)
    # 如果指定了字典文件路径，则更新全局配置中的字符字典路径
    if FLAGS.dict:
        global_config['Global']['character_dict_path'] = FLAGS.dict
        # 拼接项目路径和字典文件路径
        dict_path = os.path.join(project_path, FLAGS.dict)
        # 检查字典文件是否存在
        loss_file(dict_path)
    
    # 如果指定了数据目录，则更新全局配置中的训练和评估数据集目录
    if FLAGS.data_dir:
        global_config['Eval']['dataset']['data_dir'] = FLAGS.data_dir
        global_config['Train']['dataset']['data_dir'] = FLAGS.data_dir
        # 拼接项目路径和数据目录
        data_dir = os.path.join(project_path, FLAGS.data_dir)
        # 检查数据目录是否存在
        loss_file(data_dir)

    # 将全局配置字典写入保存文件路径
    with open(save_file_path, 'w') as f:
        yaml.dump(
            dict(global_config), f, default_flow_style=False, sort_keys=False)
    
    # 记录项目路径信息
    logging.info("Project path is          :{}".format(project_path))
    # 记录训练数据集标签文件路径信息
    logging.info("Train list path set to   :{}".format(global_config['Train'][
        'dataset']['label_file_list'][0]))
    # 记录评估数据集标签文件路径信息
    logging.info("Eval list path set to    :{}".format(global_config['Eval'][
        'dataset']['label_file_list'][0]))
    # 记录数据集根目录路径信息
    logging.info("Dataset root path set to :{}".format(global_config['Eval'][
        'dataset']['data_dir']))
    # 记录字符字典文件路径信息
    logging.info("Dict path set to         :{}".format(global_config['Global'][
        'character_dict_path']))
    # 记录配置文件路径信息
    logging.info("Config file set to       :configs/rec/multi_language/{}".
                 format(save_file_path))
```