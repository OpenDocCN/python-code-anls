# `jieba\test\test_userdict.py`

```
# 设置文件编码为 utf-8
# 导入必要的库
from __future__ import print_function, unicode_literals
import sys
# 将上级目录添加到系统路径中
sys.path.append("../")
# 导入 jieba 库
import jieba
# 加载用户自定义词典
jieba.load_userdict("userdict.txt")
# 导入 jieba 的词性标注模块
import jieba.posseg as pseg

# 添加自定义词汇
jieba.add_word('石墨烯')
jieba.add_word('凱特琳')
# 删除自定义词汇
jieba.del_word('自定义词')

# 测试文本
test_sent = (
"李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n"
"例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类\n"
"「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。"
)
# 对文本进行分词
words = jieba.cut(test_sent)
# 打印分词结果
print('/'.join(words))

print("="*40)

# 对文本进行词性标注
result = pseg.cut(test_sent)

# 遍历词性标注结果并打印
for w in result:
    print(w.word, "/", w.flag, ", ", end=' ')

print("\n" + "="*40)

# 对指定文本进行分词
terms = jieba.cut('easy_install is great')
# 打印分词结果
print('/'.join(terms))
terms = jieba.cut('python 的正则表达式是好用的')
print('/'.join(terms))

print("="*40)

# 测试词频调整
testlist = [
('今天天气不错', ('今天', '天气')),
('如果放到post中将出错。', ('中', '将')),
('我们中出了一个叛徒', ('中', '出')),
]

# 遍历测试文本和对应的分词结果
for sent, seg in testlist:
    # 对文本进行分词，关闭 HMM
    print('/'.join(jieba.cut(sent, HMM=False)))
    word = ''.join(seg)
    # 打印调整前的词频和调整后的词频
    print('%s Before: %s, After: %s' % (word, jieba.get_FREQ(word), jieba.suggest_freq(seg, True)))
    print('/'.join(jieba.cut(sent, HMM=False)))
    print("-"*40)
```