# `ChatRWKV\rwkv_pip_package\src\rwkv\rwkv_tokenizer.py`

```
# 定义了一个名为TRIE的类，用于实现字典树数据结构
class TRIE:
    # 使用__slots__定义类的属性，以节省内存空间
    __slots__ = tuple("ch,to,values,front".split(","))
    # 初始化方法，front表示父节点，ch表示当前节点的字符
    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]  # 用于存储子节点
        self.values = set()  # 用于存储节点对应的值
        self.front = front  # 父节点

    # 用于返回对象的字符串表示形式
    def __repr__(self):
        fr = self
        ret = []
        while(fr!=None):
            if(fr.ch!=None):
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>"%(ret[::-1], self.values)
    
    # 向字典树中添加键值对
    def add(self, key:bytes, idx:int=0, val=None):
        if(idx == len(key)):
            if(val is None):
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if(self.to[ch] is None):
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx+1, val=val)
    
    # 查找最长匹配的键值对
    def find_longest(self, key:bytes, idx:int=0):
        u:TRIE = self
        ch:int = key[idx]
        
        while(u.to[ch] is not None):
            u = u.to[ch]
            idx += 1
            if(u.values):
                ret = idx, u, u.values
            if(idx==len(key)):
                break
            ch = key[idx]
        return ret

# 定义了一个名为TRIE_TOKENIZER的类
class TRIE_TOKENIZER():
    # 初始化方法，接受文件名参数
    def __init__(self, file_name):
        # 初始化索引到标记的字典
        self.idx2token = {}
        # 创建一个空列表 sorted
        sorted = [] # must be already sorted
        # 打开文件，读取每一行内容
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # 遍历每一行内容
        for l in lines:
            # 获取索引值
            idx = int(l[:l.index(' ')])
            # 获取标记值
            x = eval(l[l.index(' '):l.rindex(' ')])
            # 如果标记值是字符串，则转换成字节流
            x = x.encode("utf-8") if isinstance(x, str) else x
            # 断言标记值是字节流
            assert isinstance(x, bytes)
            # 断言字节流长度与指定长度相等
            assert len(x) == int(l[l.rindex(' '):])
            # 将标记值添加到列表 sorted 中
            sorted += [x]
            # 将索引和标记值添加到索引到标记的字典中
            self.idx2token[idx] = x
    
        # 初始化标记到索引的字典
        self.token2idx = {}
        # 遍历索引到标记的字典
        for k,v in self.idx2token.items():
            # 将标记到索引的字典进行反转
            self.token2idx[v] = int(k)
    
        # 初始化 TRIE 树
        self.root = TRIE()
        # 遍历标记到索引的字典
        for t, i in self.token2idx.items():
            # 向 TRIE 树中添加标记和值
            _ = self.root.add(t, val=(t, i))
    
    # 将字节流编码成标记序列
    def encodeBytes(self, src:bytes):
        # 初始化索引值
        idx:int = 0
        # 初始化标记列表
        tokens = []
        # 循环直到索引值达到字节流长度
        while (idx < len(src)):
            # 保存当前索引值
            _idx:int = idx
            # 在 TRIE 树中查找最长匹配的标记
            idx, _, values = self.root.find_longest(src, idx)
            # 断言索引值已经改变
            assert(idx != _idx)
            # 获取匹配的标记
            _, token = next(iter(values))            
            # 将标记添加到标记列表中
            tokens.append(token)
        # 返回标记列表
        return tokens
    
    # 将标记序列解码成字节流
    def decodeBytes(self, tokens):
        # 将标记列表转换成字节流
        return b''.join(map(lambda i: self.idx2token[i], tokens))
    
    # 将字符串编码成标记序列
    def encode(self, src):
        # 将字符串转换成字节流，然后调用 encodeBytes 方法进行编码
        return self.encodeBytes(src.encode("utf-8"))
    
    # 将标记序列解码成字符串
    def decode(self, tokens):
        try:
            # 尝试将标记序列解码成字符串
            return self.decodeBytes(tokens).decode('utf-8')
        except:
            # 如果解码失败，则返回特定字符串
            return '\ufffd' # bad utf-8
    
    # 打印标记列表对应的标记值
    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                # 尝试将标记值解码成字符串
                s = s.decode('utf-8')
            except:
                pass
            # 打印标记值和索引
            print(f'{repr(s)}{i}', end=' ')
        print()
```