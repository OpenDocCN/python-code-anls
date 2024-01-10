# `basic-computer-games\70_Poetry\csharp\Context.cs`

```
namespace Poetry;

internal class Context
{
    private readonly IReadWrite _io;  // 用于读写操作的接口
    private readonly IRandom _random;  // 用于生成随机数的接口
    private int _phraseNumber;  // 当前短语的编号
    private int _groupNumber;  // 当前组的编号
    private bool _skipComma;  // 是否跳过逗号
    private int _lineCount;  // 当前行数
    private bool _useGroup2;  // 是否使用第二组
    private bool _atStartOfLine = true;  // 是否在行的开头

    public Context(IReadWrite io, IRandom random)  // 构造函数，初始化读写接口和随机数生成接口
    {
        _io = io;
        _random = random;
    }

    public int PhraseNumber => Math.Max(_phraseNumber - 1, 0);  // 获取当前短语的编号，最小为0

    public int GroupNumber  // 获取当前组的编号
    { 
        get
        {
            var value = _useGroup2 ? 2 : _groupNumber;  // 如果使用第二组，则编号为2，否则为当前组编号
            _useGroup2 = false;  // 重置使用第二组的标志
            return Math.Max(value - 1, 0);  // 返回当前组编号，最小为0
        }
    }

    public int PhraseCount { get; set; }  // 短语计数
    public bool GroupNumberIsValid => _groupNumber < 5;  // 判断当前组编号是否有效

    public void WritePhrase()  // 写入短语
    {
        Phrase.GetPhrase(this).Write(_io, this);  // 获取短语并写入
        _atStartOfLine = false;  // 设置不在行的开头
    }

    public void MaybeWriteComma()  // 可能写入逗号
    {
        if (!_skipComma && _random.NextFloat() <= 0.19F && PhraseCount != 0)  // 如果不跳过逗号且随机数小于等于0.19且短语计数不为0
        {
            _io.Write(",");  // 写入逗号
            PhraseCount = 2;  // 设置短语计数为2
        }
        _skipComma = false;  // 重置跳过逗号的标志
    }

    public void WriteSpaceOrNewLine()  // 写入空格或换行
    {
        if (_random.NextFloat() <= 0.65F)  // 如果随机数小于等于0.65
        {
            _io.Write(" ");  // 写入空格
            PhraseCount += 1;  // 短语计数加1
        }
        else
        {
            EndLine();  // 结束当前行
            PhraseCount = 0;  // 重置短语计数
        }
    }

    public void Update(IRandom random)  // 更新上下文
    {
        _phraseNumber = random.Next(1, 6);  // 随机生成1到5之间的短语编号
        _groupNumber += 1;  // 组编号加1
        _lineCount += 1;  // 行数加1
    }

    public void MaybeIndent()  // 可能缩进
    {
        if (PhraseCount == 0 && _groupNumber % 2 == 0)  // 如果短语计数为0且组编号为偶数
        {
            _io.Write("     ");  // 写入4个空格
        }
    }
    
    public void ResetGroup()  // 重置组
    {
        _groupNumber = 0;  // 组编号重置为0
        EndLine();  // 结束当前行
    }

    public bool MaybeCompleteStanza()  // 可能完成诗节
    {
        // 如果行数大于20，则换行，并重置计数器和标志位，然后返回true
        if (_lineCount > 20)
        {
            _io.WriteLine();
            PhraseCount = _lineCount = 0;
            _useGroup2 = true;
            return true;
        }

        // 否则返回false
        return false;
    }

    // 如果在行首，则将字符串首字母大写，否则返回原字符串
    internal string MaybeCapitalise(string text) =>
        _atStartOfLine ? (char.ToUpper(text[0]) + text[1..]) : text;

    // 设置跳过下一个逗号的标志位
    public void SkipNextComma() => _skipComma = true;

    // 结束当前行，换行并将行首标志位设为true
    public void EndLine()
    {
        _io.WriteLine();
        _atStartOfLine = true;
    }
# 闭合前面的函数定义
```