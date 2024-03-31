# `.\rwkv\rwkv_tokenizer.py`

```
# 定义一个 TRIE 类，用于实现字典树数据结构
class TRIE:
    # 定义类的属性 slots
    __slots__ = tuple("ch,to,values,front".split(","))
    to:list
    values:set
    # 初始化方法，front 为前缀节点，ch 为字符
    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    # 重写对象的字符串表示
    def __repr__(self):
        fr = self
        ret = []
        # 遍历节点，获取路径和值
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
        ret = None
        
        while(u.to[ch] is not None):
            u = u.to[ch]
            idx += 1
            if(u.values):
                ret = idx, u, u.values
            if(idx==len(key)):
                break
            ch = key[idx]
        return ret

# 定义 TRIE_TOKENIZER 类
class TRIE_TOKENIZER():
    # 初始化方法，从文件中读取数据构建字典树
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = [] # must be already sorted
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k,v in self.idx2token.items():
            self.token2idx[v] = int(k)

        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    # 将字节流编码为 tokens
    def encodeBytes(self, src:bytes):
        idx:int = 0
        tokens = []
        while (idx < len(src)):
            _idx:int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert(idx != _idx)
            _, token = next(iter(values))            
            tokens.append(token)
        return tokens

    # 将 tokens 解码为字节流
    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    # 编码字符串为 tokens
    def encode(self, src):
        return self.encodeBytes(src.encode("utf-8"))

    # 解码 tokens 为字符串
    def decode(self, tokens):
        try:
            return self.decodeBytes(tokens).decode('utf-8')
        except:
            return '\ufffd' # bad utf-8
    # 定义一个方法，用于打印 tokens
    def printTokens(self, tokens):
        # 遍历 tokens 列表
        for i in tokens:
            # 通过索引获取对应的 token
            s = self.idx2token[i]
            # 尝试将 token 解码为 utf-8 编码
            try:
                s = s.decode('utf-8')
            # 如果解码失败，则跳过
            except:
                pass
            # 打印 token 的 repr 格式和索引 i，以空格结尾
            print(f'{repr(s)}{i}', end=' ')
        # 打印换行符
        print()
```