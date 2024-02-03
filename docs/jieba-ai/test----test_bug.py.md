# `jieba\test\test_bug.py`

```
# 设置文件编码为 UTF-8
# 导入未来版本的 print 函数
import sys
# 将上级目录添加到 sys.path 中
sys.path.append("../")
# 导入 jieba 库
import jieba
# 导入 jieba 的词性标注模块
import jieba.posseg as pseg
# 对字符串 "又跛又啞" 进行分词并标注词性
words = pseg.cut("又跛又啞")
# 遍历分词结果
for w in words:
    # 打印分词结果的词和词性
    print(w.word, w.flag)
```