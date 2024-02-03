# `jieba\test\test_change_dictpath.py`

```py
# 设置文件编码为utf-8
# 导入未来版本的print函数
import sys
# 将上级目录添加到sys.path中
sys.path.append("../")
# 导入结巴分词库
import jieba

# 定义一个函数，用于对输入文本进行分词并打印结果
def cuttest(test_sent):
    # 对输入文本进行分词
    result = jieba.cut(test_sent)
    # 打印分词结果，用空格连接
    print("  ".join(result))

# 定义一个测试函数，用于测试分词功能
def testcase():
    # 测试不同文本的分词结果
    cuttest("这是一个伸手不见五指的黑夜。我叫孙悟空，我爱北京，我爱Python和C++。")
    cuttest("我不喜欢日本和服。")
    cuttest("雷猴回归人间。")
    cuttest("工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作")
    cuttest("我需要廉租房")
    cuttest("永和服装饰品有限公司")
    cuttest("我爱北京天安门")
    cuttest("abc")
    cuttest("隐马尔可夫")
    cuttest("雷猴是个好网站")

# 如果当前脚本被直接执行，则执行测试函数
if __name__ == "__main__":
    # 执行测试函数
    testcase()
    # 设置结巴分词的自定义词典为"foobar.txt"
    jieba.set_dictionary("foobar.txt")
    # 打印分隔线
    print("================================")
    # 再次执行测试函数
    testcase()
```