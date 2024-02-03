# `jieba\test\parallel\test_disable_hmm.py`

```py
# 设置编码格式为utf-8
# 导入必要的库
from __future__ import print_function
import sys
# 添加上级目录到系统路径中
sys.path.append("../../")
# 导入结巴分词库
import jieba
# 启用并行分词，指定并行数为4
jieba.enable_parallel(4)

# 定义一个函数，用于对输入文本进行结巴分词
def cuttest(test_sent):
    # 对输入文本进行分词，关闭HMM新词发现
    result = jieba.cut(test_sent, HMM=False)
    # 遍历分词结果，逐个打印分词结果和词性
    for word in result:
        print(word, "/", end=' ')
    # 打印换行符
    print("")

# 主函数入口
if __name__ == "__main__":
    # 调用cuttest函数对不同文本进行分词
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
    cuttest("“Microsoft”一词由“MICROcomputer（微型计算机）”和“SOFTware（软件）”两部分组成")
    cuttest("草泥马和欺实马是今年的流行词汇")
    cuttest("伊藤洋华堂总府店")
    cuttest("中国科学院计算技术研究所")
    cuttest("罗密欧与朱丽叶")
    cuttest("我购买了道具和服装")
    cuttest("PS: 我觉得开源有一个好处，就是能够敦促自己不断改进，避免敞帚自珍")
    cuttest("湖北省石首市")
    cuttest("湖北省十堰市")
    cuttest("总经理完成了这件事情")
    cuttest("电脑修好了")
    cuttest("做好了这件事情就一了百了了")
    cuttest("人们审美的观点是不同的")
    cuttest("我们买了一个美的空调")
    cuttest("线程初始化时我们要注意")
    cuttest("一个分子是由好多原子组织成的")
    cuttest("祝你马到功成")
    cuttest("他掉进了无底洞里")
    cuttest("中国的首都是北京")
    cuttest("孙君意")
    cuttest("外交部发言人马朝旭")
    cuttest("领导人会议和第四届东亚峰会")
    cuttest("在过去的这五年")
    cuttest("还需要很长的路要走")
    cuttest("60周年首都阅兵")
    cuttest("你好人们审美的观点是不同的")
    cuttest("买水果然后来世博园")
    cuttest("买水果然后去世博园")
    cuttest("但是后来我才知道你是对的")
    cuttest("存在即合理")
    cuttest("的的的的的在的的的的就以和和和")
    cuttest("I love你，不以为耻，反以为rong")
    cuttest("因")
    cuttest("")
    cuttest("hello你好人们审美的观点是不同的")
    cuttest("很好但主要是基于网页形式")
    cuttest("hello你好人们审美的观点是不同的")
    cuttest("为什么我不能拥有想要的生活")
    cuttest("后来我才")
    cuttest("此次来中国是为了")
    cuttest("使用了它就可以解决一些问题")
    cuttest(",使用了它就可以解决一些问题")
    cuttest("其实使用了它就可以解决一些问题")
    cuttest("好人使用了它就可以解决一些问题")
    cuttest("是因为和国家")
    cuttest("老年搜索还支持")
    cuttest("干脆就把那部蒙人的闲法给废了拉倒！RT @laoshipukong : 27日，全国人大常委会第三次审议侵权责任法草案，删除了有关医疗损害责任“举证倒置”的规定。在医患纠纷中本已处于弱势地位的消费者由此将陷入万劫不复的境地。 ")
    cuttest("大")
    # 调用cuttest函数，传入空字符串作为参数
    cuttest("")
    
    # 调用cuttest函数，传入包含中文的字符串作为参数
    cuttest("他说的确实在理")
    
    # 调用cuttest函数，传入包含中文的字符串作为参数
    cuttest("长春市长春节讲话")
    
    # 调用cuttest函数，传入包含中文的字符串作为参数
    cuttest("结婚的和尚未结婚的")
    
    # 调用cuttest函数，传入包含中文的字符串作为参数
    cuttest("结合成分子时")
    
    # 调用cuttest函数，传入包含中文的字符串作为参数
    cuttest("旅游和服务是最好的")
    
    # 调用cuttest函数，传入包含中文的字符串作为参数
    cuttest("这件事情的确是我的错")
    
    # 调用cuttest函数，传入包含中文的字符串作为参数
    cuttest("供大家参考指正")
    
    # 调用cuttest函数，传入包含中文的字符串作为参数
    cuttest("哈尔滨政府公布塌桥原因")
    
    # 调用cuttest函数，传入包含中文的字符串作为参数
    cuttest("我在机场入口处")
    
    # 调用cuttest函数，传入包含中文的字符串作为参数
    cuttest("邢永臣摄影报道")
    
    # 调用cuttest函数，传入包含中文的字符串作为参数
    cuttest("BP神经网络如何训练才能在分类时增加区分度？")
    
    # 调用cuttest函数，传入包含中文的字符串作为参数
    cuttest("南京市长江大桥")
    
    # 调用cuttest函数，传入包含中文的字符串作为参数
    cuttest("应一些使用者的建议，也为了便于利用NiuTrans用于SMT研究")
    
    # 调用cuttest函数，传入包含中文的字符串作为参数
    cuttest('长春市长春药店')
    
    # 调用cuttest函数，传入包含中文的字符串作为参数
    cuttest('邓颖超生前最喜欢的衣服')
    
    # 调用cuttest函数，传入包含中文的字符串作为参数
    cuttest('胡锦涛是热爱世界和平的政治局常委')
    
    # 调用cuttest函数，传入包含中文的字符串作为参数
    cuttest('程序员祝海林和朱会震是在孙健的左面和右面, 范凯在最右面.再往左是李松洪')
    
    # 调用cuttest函数，传入包含中文的字符串作为参数
    cuttest('一次性交多少钱')
    
    # 调用cuttest函数，传入包含中文的字符串作为参数
    cuttest('两块五一套，三块八一斤，四块七一本，五块六一条')
    
    # 调用cuttest函数，传入包含中文的字符串作为参数
    cuttest('小和尚留了一个像大和尚一样的和尚头')
    
    # 调用cuttest函数，传入包含中文的字符串作为参数
    cuttest('我是中华人民共和国公民;我爸爸是共和党党员; 地铁和平门站')
```