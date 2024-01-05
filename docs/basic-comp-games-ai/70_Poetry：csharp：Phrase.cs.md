# `70_Poetry\csharp\Phrase.cs`

```
namespace Poetry;  // 命名空间声明

internal class Phrase  // 定义内部类 Phrase
{
    private readonly static Phrase[][] _phrases = new Phrase[][]  // 声明并初始化静态二维数组 _phrases
    {
        new Phrase[]  // 初始化第一个一维数组
        {
            new("midnight dreary"),  // 创建 Phrase 对象并传入字符串参数
            new("fiery eyes"),  // 创建 Phrase 对象并传入字符串参数
            new("bird or fiend"),  // 创建 Phrase 对象并传入字符串参数
            new("thing of evil"),  // 创建 Phrase 对象并传入字符串参数
            new("prophet")  // 创建 Phrase 对象并传入字符串参数
        },
        new Phrase[]  // 初始化第二个一维数组
        {
            new("beguiling me", ctx => ctx.PhraseCount = 2),  // 创建 Phrase 对象并传入字符串参数和委托参数
            new("thrilled me"),  // 创建 Phrase 对象并传入字符串参数
            new("still sitting....", ctx => ctx.SkipNextComma()),  // 创建 Phrase 对象并传入字符串参数和委托参数
            new("never flitting", ctx => ctx.PhraseCount = 2),  // 创建 Phrase 对象并传入字符串参数和委托参数
        new("burned")  # 创建一个新的短语对象，内容为"burned"
    },
    new Phrase[]  # 创建一个新的短语数组
    {
        new("and my soul"),  # 创建一个新的短语对象，内容为"and my soul"
        new("darkness there"),  # 创建一个新的短语对象，内容为"darkness there"
        new("shall be lifted"),  # 创建一个新的短语对象，内容为"shall be lifted"
        new("quoth the raven"),  # 创建一个新的短语对象，内容为"quoth the raven"
        new(ctx => ctx.PhraseCount != 0, "sign of parting")  # 创建一个新的短语对象，内容为"sign of parting"，并且根据上下文中的短语数量是否为0来确定是否创建该短语
    },
    new Phrase[]  # 创建一个新的短语数组
    {
        new("nothing more"),  # 创建一个新的短语对象，内容为"nothing more"
        new("yet again"),  # 创建一个新的短语对象，内容为"yet again"
        new("slowly creeping"),  # 创建一个新的短语对象，内容为"slowly creeping"
        new("...evermore"),  # 创建一个新的短语对象，内容为"...evermore"
        new("nevermore")  # 创建一个新的短语对象，内容为"nevermore"
    }
};
    # 定义私有成员变量 _condition，类型为 Predicate<Context>，用于存储条件判断函数
    private readonly Predicate<Context> _condition;
    # 定义私有成员变量 _text，类型为 string，用于存储文本信息
    private readonly string _text;
    # 定义私有成员变量 _update，类型为 Action<Context>，用于存储更新操作函数

    # 定义构造函数，参数为条件判断函数和文本信息，调用另一个构造函数并传入默认的更新操作函数
    private Phrase(Predicate<Context> condition, string text)
        : this(condition, text, _ => { })
    {
    }

    # 定义构造函数，参数为文本信息和更新操作函数，调用另一个构造函数并传入默认的条件判断函数
    private Phrase(string text, Action<Context> update)
        : this(_ => true, text, update)
    {
    }

    # 定义构造函数，参数为文本信息，调用另一个构造函数并传入默认的条件判断函数和更新操作函数
    private Phrase(string text)
        : this(_ => true, text, _ => { })
    {
    }

    # 定义构造函数，参数为条件判断函数、文本信息和更新操作函数
    private Phrase(Predicate<Context> condition, string text, Action<Context> update)
    {
        _condition = condition;  # 将传入的条件赋值给私有变量_condition
        _text = text;  # 将传入的文本赋值给私有变量_text
        _update = update;  # 将传入的更新方法赋值给私有变量_update
    }

    public static Phrase GetPhrase(Context context) => _phrases[context.GroupNumber][context.PhraseNumber];  # 静态方法，根据传入的上下文返回对应的短语

    public void Write(IReadWrite io, Context context)  # 公有方法，接受一个IReadWrite类型的参数和一个上下文参数
    {
        if (_condition.Invoke(context))  # 如果条件满足上下文的条件
        {
            io.Write(context.MaybeCapitalise(_text));  # 调用io的Write方法，将文本写入
        }

        _update.Invoke(context);  # 调用更新方法，传入上下文参数
    }
}
```