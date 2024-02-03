# `jieba\test\jieba_test.py`

```
# 设置文件编码为 utf-8
# 导入必要的库
from __future__ import unicode_literals, print_function
import sys
# 将上级目录添加到 sys.path 中
sys.path.append("../")
import unittest
import types
# 导入 jieba 库
import jieba
# 如果 Python 版本大于 2，则导入 reload 函数
if sys.version_info[0] > 2:
    from imp import reload

# 初始化 jieba 库
jieba.initialize()

# 测试用例内容
test_contents = [
    "这是一个伸手不见五指的黑夜。我叫孙悟空，我爱北京，我爱Python和C++。",
    "我不喜欢日本和服。",
    "雷猴回归人间。",
    "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作",
    "我需要廉租房",
    "永和服装饰品有限公司",
    "我爱北京天安门",
    "abc",
    "隐马尔可夫",
    "雷猴是个好网站",
    "“Microsoft”一词由“MICROcomputer（微型计算机）”和“SOFTware（软件）”两部分组成",
    "草泥马和欺实马是今年的流行词汇",
    "伊藤洋华堂总府店",
    "中国科学院计算技术研究所",
    "罗密欧与朱丽叶",
    "我购买了道具和服装",
    "PS: 我觉得开源有一个好处，就是能够敦促自己不断改进，避免敞帚自珍",
    "湖北省石首市",
    "湖北省十堰市",
    "总经理完成了这件事情",
    "电脑修好了",
    "做好了这件事情就一了百了了",
    "人们审美的观点是不同的",
    "我们买了一个美的空调",
    "线程初始化时我们要注意",
    "一个分子是由好多原子组织成的",
    "祝你马到功成",
    "他掉进了无底洞里",
    "中国的首都是北京",
    "孙君意",
    "外交部发言人马朝旭",
    "领导人会议和第四届东亚峰会",
    "在过去的这五年",
    "还需要很长的路要走",
    "60周年首都阅兵",
    "你好人们审美的观点是不同的",
    "买水果然后来世博园",
    "买水果然后去世博园",
    "但是后来我才知道你是对的",
    "存在即合理",
    "的的的的的在的的的的就以和和和",
    "I love你，不以为耻，反以为rong",
    "因",
    "",
    "hello你好人们审美的观点是不同的",
    "很好但主要是基于网页形式",
    "hello你好人们审美的观点是不同的",
    "为什么我不能拥有想要的生活",
    "后来我才",
    "此次来中国是为了",
    "使用了它就可以解决一些问题",
    ",使用了它就可以解决一些问题",
    "其实使用了它就可以解决一些问题",
    "好人使用了它就可以解决一些问题",
    "是因为和国家",
    "老年搜索还支持",
    "干脆就把那部蒙人的闲法给废了拉倒！RT @laoshipukong : 27日，全国人大常委会第三次审议侵权责任法草案，删除了有关医疗损害责任“举证倒置”的规定。在医患纠纷中本已处于弱势地位的消费者由此将陷入万劫不复的境地。 ",
    "大",
    "",
    "他说的确实在理",
    "长春市长春节讲话",
    "结婚的和尚未结婚的",
    "结合成分子时",
    "旅游和服务是最好的",
    "这件事情的确是我的错",
    "供大家参考指正",
    "哈尔滨政府公布塌桥原因",
    "我在机场入口处",
    "邢永臣摄影报道",
    "BP神经网络如何训练才能在分类时增加区分度？",
    "南京市长江大桥",
    "应一些使用者的建议，也为了便于利用NiuTrans用于SMT研究",
    '长春市长春药店',
    '邓颖超生前最喜欢的衣服',
    '胡锦涛是热爱世界和平的政治局常委',
    '程序员祝海林和朱会震是在孙健的左面和右面, 范凯在最右面.再往左是李松洪',
    '一次性交多少钱',
    '两块五一套，三块八一斤，四块七一本，五块六一条',
    '小和尚留了一个像大和尚一样的和尚头',
    '我是中华人民共和国公民;我爸爸是共和党党员; 地铁和平门站',
    '张晓梅去人民医院做了个B超然后去买了件T恤',
]
    # 创建一个包含多个字符串的列表
    [
        'AT&T是一件不错的公司，给你发offer了吗？',
        'C++和c#是什么关系？11+122=133，是吗？PI=3.14159',
        '你认识那个和主席握手的的哥吗？他开一辆黑色的士。',
        '枪杆子中出政权'
    ]
# 定义一个测试类 JiebaTestCase，继承自 unittest.TestCase
class JiebaTestCase(unittest.TestCase):
    # 在每个测试方法执行前重新加载 jieba 模块
    def setUp(self):
        reload(jieba)

    # 在每个测试方法执行后不执行任何操作
    def tearDown(self):
        pass

    # 测试默认分词方法
    def testDefaultCut(self):
        # 遍历测试内容列表
        for content in test_contents:
            # 使用 jieba.cut 对内容进行分词
            result = jieba.cut(content)
            # 检查结果是否为生成器类型
            assert isinstance(result, types.GeneratorType), "Test DefaultCut Generator error"
            # 将生成器转换为列表
            result = list(result)
            # 检查结果是否为列表类型
            assert isinstance(result, list), "Test DefaultCut error on content: %s" % content
            # 将分词结果以逗号分隔输出到标准错误流
            print(" , ".join(result), file=sys.stderr)
        # 输出测试方法名称到标准错误流
        print("testDefaultCut", file=sys.stderr)

    # 测试全模式分词方法
    def testCutAll(self):
        # 遍历测试内容列表
        for content in test_contents:
            # 使用 jieba.cut 对内容进行全模式分词
            result = jieba.cut(content, cut_all=True)
            # 检查结果是否为生成器类型
            assert isinstance(result, types.GeneratorType), "Test CutAll Generator error"
            # 将生成器转换为列表
            result = list(result)
            # 检查结果是否为列表类型
            assert isinstance(result, list), "Test CutAll error on content: %s" % content
            # 将全模式分词结果以逗号分隔输出到标准错误流
            print(" , ".join(result), file=sys.stderr)
        # 输出测试方法名称到标准错误流
        print("testCutAll", file=sys.stderr)

    # 测试设置自定义词典方法
    def testSetDictionary(self):
        # 设置自定义词典为 "foobar.txt"
        jieba.set_dictionary("foobar.txt")
        # 遍历测试内容列表
        for content in test_contents:
            # 使用 jieba.cut 对内容进行分词
            result = jieba.cut(content)
            # 检查结果是否为生成器类型
            assert isinstance(result, types.GeneratorType), "Test SetDictionary Generator error"
            # 将生成器转换为列表
            result = list(result)
            # 检查结果是否为列表类型
            assert isinstance(result, list), "Test SetDictionary error on content: %s" % content
            # 将分词结果以逗号分隔输出到标准错误流
            print(" , ".join(result), file=sys.stderr)
        # 输出测试方法名称到标准错误流
        print("testSetDictionary", file=sys.stderr)

    # 测试搜索引擎模式分词方法
    def testCutForSearch(self):
        # 遍历测试内容列表
        for content in test_contents:
            # 使用 jieba.cut_for_search 对内容进行分词
            result = jieba.cut_for_search(content)
            # 检查结果是否为生成器类型
            assert isinstance(result, types.GeneratorType), "Test CutForSearch Generator error"
            # 将生成器转换为列表
            result = list(result)
            # 检查结果是否为列表类型
            assert isinstance(result, list), "Test CutForSearch error on content: %s" % content
            # 将搜索引擎模式分词结果以逗号分隔输出到标准错误流
            print(" , ".join(result), file=sys.stderr)
        # 输出测试方法名称到标准错误流
        print("testCutForSearch", file=sys.stderr)
    # 测试中文分词库 jieba 的词性标注功能
    def testPosseg(self):
        # 导入 jieba 的词性标注模块
        import jieba.posseg as pseg
        # 遍历测试内容列表
        for content in test_contents:
            # 对内容进行词性标注
            result = pseg.cut(content)
            # 检查结果是否为生成器类型
            assert isinstance(result, types.GeneratorType), "Test Posseg Generator error"
            # 将生成器转换为列表
            result = list(result)
            # 检查结果是否为列表类型
            assert isinstance(result, list), "Test Posseg error on content: %s" % content
            # 打印词性标注结果
            print(" , ".join([w.word + " / " + w.flag for w in result]), file=sys.stderr)
        # 打印测试结束信息
        print("testPosseg", file=sys.stderr)

    # 测试中文分词库 jieba 的分词功能
    def testTokenize(self):
        # 遍历测试内容列表
        for content in test_contents:
            # 对内容进行分词
            result = jieba.tokenize(content)
            # 检查结果是否为生成器类型
            assert isinstance(result, types.GeneratorType), "Test Tokenize Generator error"
            # 将生成器转换为列表
            result = list(result)
            # 检查结果是否为列表类型
            assert isinstance(result, list), "Test Tokenize error on content: %s" % content
            # 遍历分词结果并打印
            for tk in result:
                print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]), file=sys.stderr)
        # 打印测试结束信息
        print("testTokenize", file=sys.stderr)

    # 测试中文分词库 jieba 的默认分词功能（关闭 HMM）
    def testDefaultCut_NOHMM(self):
        # 遍历测试内容列表
        for content in test_contents:
            # 对内容进行分词（关闭 HMM）
            result = jieba.cut(content,HMM=False)
            # 检查结果是否为生成器类型
            assert isinstance(result, types.GeneratorType), "Test DefaultCut Generator error"
            # 将生成器转换为列表
            result = list(result)
            # 检查结果是否为列表类型
            assert isinstance(result, list), "Test DefaultCut error on content: %s" % content
            # 打印分词结果
            print(" , ".join(result), file=sys.stderr)
        # 打印测试结束信息
        print("testDefaultCut_NOHMM", file=sys.stderr)

    # 测试中文分词库 jieba 的词性标注功能（关闭 HMM）
    def testPosseg_NOHMM(self):
        # 导入 jieba 的词性标注模块
        import jieba.posseg as pseg
        # 遍历测试内容列表
        for content in test_contents:
            # 对内容进行词性标注（关闭 HMM）
            result = pseg.cut(content,HMM=False)
            # 检查结果是否为生成器类型
            assert isinstance(result, types.GeneratorType), "Test Posseg Generator error"
            # 将生成器转换为列表
            result = list(result)
            # 检查结果是否为列表类型
            assert isinstance(result, list), "Test Posseg error on content: %s" % content
            # 打印词性标注结果
            print(" , ".join([w.word + " / " + w.flag for w in result]), file=sys.stderr)
        # 打印测试结束信息
        print("testPosseg_NOHMM", file=sys.stderr)
    # 测试不使用隐马尔可夫模型进行分词的情况
    def testTokenize_NOHMM(self):
        # 遍历测试内容列表
        for content in test_contents:
            # 使用结巴分词对内容进行分词，不使用隐马尔可夫模型
            result = jieba.tokenize(content, HMM=False)
            # 检查结果是否为生成器类型
            assert isinstance(result, types.GeneratorType), "Test Tokenize Generator error"
            # 将生成器转换为列表
            result = list(result)
            # 检查结果是否为列表类型
            assert isinstance(result, list), "Test Tokenize error on content: %s" % content
            # 遍历分词结果，打印每个词的起始位置和结束位置
            for tk in result:
                print("word %s\t\t start: %d \t\t end:%d" % (tk[0], tk[1], tk[2]), file=sys.stderr)
        # 打印测试结束信息
        print("testTokenize_NOHMM", file=sys.stderr)
    
    # 测试不使用隐马尔可夫模型进行搜索引擎模式分词的情况
    def testCutForSearch_NOHMM(self):
        # 遍历测试内容列表
        for content in test_contents:
            # 使用结巴分词对内容进行搜索引擎模式分词，不使用隐马尔可夫模型
            result = jieba.cut_for_search(content, HMM=False)
            # 检查结果是否为生成器类型
            assert isinstance(result, types.GeneratorType), "Test CutForSearch Generator error"
            # 将生成器转换为列表
            result = list(result)
            # 检查结果是否为列表类型
            assert isinstance(result, list), "Test CutForSearch error on content: %s" % content
            # 打印分词结果，以逗号分隔
            print(" , ".join(result), file=sys.stderr)
        # 打印测试结束信息
        print("testCutForSearch_NOHMM", file=sys.stderr)
# 如果当前脚本被直接执行，则执行单元测试
if __name__ == "__main__":
    unittest.main()
```