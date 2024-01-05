# `d:/src/tocomm/basic-computer-games\70_Poetry\csharp\Context.cs`

```
    // 声明私有变量 _io 用于读写操作
    private readonly IReadWrite _io;
    // 声明私有变量 _random 用于生成随机数
    private readonly IRandom _random;
    // 声明私有变量 _phraseNumber 用于存储短语数量
    private int _phraseNumber;
    // 声明私有变量 _groupNumber 用于存储组数量
    private int _groupNumber;
    // 声明私有变量 _skipComma 用于标记是否跳过逗号
    private bool _skipComma;
    // 声明私有变量 _lineCount 用于存储行数
    private int _lineCount;
    // 声明私有变量 _useGroup2 用于标记是否使用第二组
    private bool _useGroup2;
    // 声明私有变量 _atStartOfLine 用于标记是否在行的起始位置

    // 构造函数，接受 IReadWrite 和 IRandom 接口实例作为参数
    public Context(IReadWrite io, IRandom random)
    {
        // 将传入的 io 赋值给私有变量 _io
        _io = io;
        // 将传入的 random 赋值给私有变量 _random
        _random = random;
    }

    // 只读属性，返回 _phraseNumber 减去 1 的值，如果小于 0 则返回 0
    public int PhraseNumber => Math.Max(_phraseNumber - 1, 0); 
    public int GroupNumber 
    { 
        get
        {
            // 如果_useGroup2为true，则返回2，否则返回_groupNumber
            var value = _useGroup2 ? 2 : _groupNumber;
            // 将_useGroup2设置为false
            _useGroup2 = false;
            // 返回value-1和0中的较大值
            return Math.Max(value - 1, 0);
        }
    }

    // 获取或设置短语数量
    public int PhraseCount { get; set; }
    // 检查_groupNumber是否小于5
    public bool GroupNumberIsValid => _groupNumber < 5;

    // 写入短语
    public void WritePhrase()
    {
        // 获取短语并写入_io
        Phrase.GetPhrase(this).Write(_io, this);
        // 将_atStartOfLine设置为false
        _atStartOfLine = false;
    }
    public void MaybeWriteComma()
    {
        // 检查是否需要写逗号
        if (!_skipComma && _random.NextFloat() <= 0.19F && PhraseCount != 0)
        {
            // 写入逗号并更新短语计数
            _io.Write(",");
            PhraseCount = 2;
        }
        // 重置跳过逗号的标志
        _skipComma = false;
    }

    public void WriteSpaceOrNewLine()
    {
        // 写入空格或换行符
        if (_random.NextFloat() <= 0.65F)
        {
            _io.Write(" ");
            // 更新短语计数
            PhraseCount += 1;
        }
        else
        {
            // 调用EndLine方法，换行
            EndLine();
        }
    }
            PhraseCount = 0;  // 重置短语计数器为0
        }
    }

    public void Update(IRandom random)
    {
        _phraseNumber = random.Next(1, 6);  // 生成一个1到6之间的随机数，赋值给_phraseNumber
        _groupNumber += 1;  // _groupNumber增加1
        _lineCount += 1;  // _lineCount增加1
    }

    public void MaybeIndent()
    {
        if (PhraseCount == 0 && _groupNumber % 2 == 0)  // 如果PhraseCount为0且_groupNumber是偶数
        {
            _io.Write("     ");  // 在输出流中写入5个空格，用于缩进
        }
    }
    
    public void ResetGroup()  // 重置组的方法
    {
        _groupNumber = 0;  # 将_groupNumber变量设置为0
        EndLine();  # 调用EndLine函数
    }

    public bool MaybeCompleteStanza()  # 定义一个公共的函数MaybeCompleteStanza，返回布尔值
    {
        if (_lineCount > 20)  # 如果_lineCount大于20
        {
            _io.WriteLine();  # 在_io中写入一个空行
            PhraseCount = _lineCount = 0;  # 将PhraseCount和_lineCount都设置为0
            _useGroup2 = true;  # 将_useGroup2变量设置为true
            return true;  # 返回true
        }

        return false;  # 返回false
    }

    internal string MaybeCapitalise(string text) =>  # 定义一个内部的函数MaybeCapitalise，接受一个字符串参数并返回一个字符串
        _atStartOfLine ? (char.ToUpper(text[0]) + text[1..]) : text;  # 如果_atStartOfLine为真，则将text的第一个字符转换为大写并返回，否则返回原始text
# 跳过下一个逗号
public void SkipNextComma() => _skipComma = true;

# 结束当前行
public void EndLine()
{
    # 写入换行符
    _io.WriteLine();
    # 设置在行的起始位置
    _atStartOfLine = true;
}
```