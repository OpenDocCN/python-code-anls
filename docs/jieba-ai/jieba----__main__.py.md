# `jieba\jieba\__main__.py`

```py
"""Jieba command line interface."""
# 导入必要的库和模块
import sys
import jieba
from argparse import ArgumentParser
from ._compat import *

# 创建参数解析器
parser = ArgumentParser(usage="%s -m jieba [options] filename" % sys.executable, description="Jieba command line interface.", epilog="If no filename specified, use STDIN instead.")
# 添加参数选项
parser.add_argument("-d", "--delimiter", metavar="DELIM", default=' / ',
                    nargs='?', const=' ',
                    help="use DELIM instead of ' / ' for word delimiter; or a space if it is used without DELIM")
parser.add_argument("-p", "--pos", metavar="DELIM", nargs='?', const='_',
                    help="enable POS tagging; if DELIM is specified, use DELIM instead of '_' for POS delimiter")
parser.add_argument("-D", "--dict", help="use DICT as dictionary")
parser.add_argument("-u", "--user-dict",
                    help="use USER_DICT together with the default dictionary or DICT (if specified)")
parser.add_argument("-a", "--cut-all",
                    action="store_true", dest="cutall", default=False,
                    help="full pattern cutting (ignored with POS tagging)")
parser.add_argument("-n", "--no-hmm", dest="hmm", action="store_false",
                    default=True, help="don't use the Hidden Markov Model")
parser.add_argument("-q", "--quiet", action="store_true", default=False,
                    help="don't print loading messages to stderr")
parser.add_argument("-V", '--version', action='version',
                    version="Jieba " + jieba.__version__)
parser.add_argument("filename", nargs='?', help="input file")

# 解析命令行参数
args = parser.parse_args()

# 根据参数设置是否打印加载信息
if args.quiet:
    jieba.setLogLevel(60)
# 如果启用了词性标注
if args.pos:
    import jieba.posseg
    posdelim = args.pos
    # 定义处理函数，根据参数决定是否使用隐马尔可夫模型
    def cutfunc(sentence, _, HMM=True):
        for w, f in jieba.posseg.cut(sentence, HMM):
            yield w + posdelim + f
else:
    cutfunc = jieba.cut

# 获取参数设置的分隔符、是否全模式切分、是否使用隐马尔可夫模型
delim = text_type(args.delimiter)
cutall = args.cutall
hmm = args.hmm
# 打开文件以供读取，如果没有指定文件名则使用标准输入流
fp = open(args.filename, 'r') if args.filename else sys.stdin

# 如果指定了自定义词典，则使用该词典初始化结巴分词
if args.dict:
    jieba.initialize(args.dict)
# 否则使用默认词典初始化结巴分词
else:
    jieba.initialize()
# 如果指定了用户自定义词典，则加载该词典
if args.user_dict:
    jieba.load_userdict(args.user_dict)

# 逐行读取文件内容
ln = fp.readline()
while ln:
    # 去除行末的换行符
    l = ln.rstrip('\r\n')
    # 对当前行进行分词处理，使用指定的分隔符连接结果
    result = delim.join(cutfunc(ln.rstrip('\r\n'), cutall, hmm))
    # 如果是 Python 2 版本，则将结果编码为默认编码格式
    if PY2:
        result = result.encode(default_encoding)
    # 打印处理后的结果
    print(result)
    # 继续读取下一行
    ln = fp.readline()

# 关闭文件流
fp.close()
```