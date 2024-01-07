# `basic-computer-games\70_Poetry\csharp\Context.cs`

```

// 命名空间声明，表示代码所属的命名空间为Poetry
namespace Poetry;

// 内部类声明，表示Context类只能在当前程序集内部访问
internal class Context
{
    // 私有字段声明，表示Context类的私有成员变量
    private readonly IReadWrite _io; // 用于读写操作的接口
    private readonly IRandom _random; // 用于生成随机数的接口
    private int _phraseNumber; // 短语编号
    private int _groupNumber; // 组编号
    private bool _skipComma; // 是否跳过逗号
    private int _lineCount; // 行数计数
    private bool _useGroup2; // 是否使用第二组
    private bool _atStartOfLine = true; // 是否在行的起始位置

    // 构造函数，初始化Context类的实例
    public Context(IReadWrite io, IRandom random)
    {
        _io = io; // 初始化_io字段
        _random = random; // 初始化_random字段
    }

    // 只读属性，表示短语编号
    public int PhraseNumber => Math.Max(_phraseNumber - 1, 0); 

    // 可读写属性，表示组编号
    public int GroupNumber 
    { 
        get
        {
            var value = _useGroup2 ? 2 : _groupNumber; // 根据_useGroup2的值确定返回值
            _useGroup2 = false; // 重置_useGroup2的值
            return Math.Max(value - 1, 0); // 返回组编号的最大值
        }
    }

    // 可读写属性，表示短语计数
    public int PhraseCount { get; set; }

    // 只读属性，表示组编号是否有效
    public bool GroupNumberIsValid => _groupNumber < 5;

    // 方法，用于写入短语
    public void WritePhrase()
    {
        Phrase.GetPhrase(this).Write(_io, this); // 调用Phrase类的GetPhrase方法并写入_io
        _atStartOfLine = false; // 设置_atStartOfLine为false
    }

    // 方法，用于可能写入逗号
    public void MaybeWriteComma()
    {
        if (!_skipComma && _random.NextFloat() <= 0.19F && PhraseCount != 0)
        {
            _io.Write(","); // 写入逗号
            PhraseCount = 2; // 设置短语计数为2
        }
        _skipComma = false; // 重置_skipComma的值
    }

    // 方法，用于写入空格或换行
    public void WriteSpaceOrNewLine()
    {
        if (_random.NextFloat() <= 0.65F)
        {
            _io.Write(" "); // 写入空格
            PhraseCount += 1; // 短语计数加1
        }
        else
        {
            EndLine(); // 调用EndLine方法
            PhraseCount = 0; // 重置短语计数为0
        }
    }

    // 方法，用于更新短语和组编号
    public void Update(IRandom random)
    {
        _phraseNumber = random.Next(1, 6); // 随机生成短语编号
        _groupNumber += 1; // 组编号加1
        _lineCount += 1; // 行数计数加1
    }

    // 方法，用于可能缩进
    public void MaybeIndent()
    {
        if (PhraseCount == 0 && _groupNumber % 2 == 0)
        {
            _io.Write("     "); // 写入4个空格
        }
    }
    
    // 方法，用于重置组编号
    public void ResetGroup()
    {
        _groupNumber = 0; // 重置组编号为0
        EndLine(); // 调用EndLine方法
    }

    // 方法，用于可能完成诗节
    public bool MaybeCompleteStanza()
    {
        if (_lineCount > 20)
        {
            _io.WriteLine(); // 写入换行
            PhraseCount = _lineCount = 0; // 重置短语计数和行数计数为0
            _useGroup2 = true; // 设置_useGroup2为true
            return true; // 返回true
        }

        return false; // 返回false
    }

    // 内部方法，用于可能大写首字母
    internal string MaybeCapitalise(string text) =>
        _atStartOfLine ? (char.ToUpper(text[0]) + text[1..]) : text;

    // 方法，用于跳过下一个逗号
    public void SkipNextComma() => _skipComma = true;

    // 方法，用于结束行
    public void EndLine()
    {
        _io.WriteLine(); // 写入换行
        _atStartOfLine = true; // 设置_atStartOfLine为true
    }
}

```