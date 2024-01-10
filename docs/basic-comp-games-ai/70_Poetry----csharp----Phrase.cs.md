# `basic-computer-games\70_Poetry\csharp\Phrase.cs`

```
namespace Poetry;

internal class Phrase
{
    private readonly static Phrase[][] _phrases = new Phrase[][]
    {
        new Phrase[]  // 创建第一个短语数组
        {
            new("midnight dreary"),  // 创建短语对象并初始化文本
            new("fiery eyes"),  // 创建短语对象并初始化文本
            new("bird or fiend"),  // 创建短语对象并初始化文本
            new("thing of evil"),  // 创建短语对象并初始化文本
            new("prophet")  // 创建短语对象并初始化文本
        },
        new Phrase[]  // 创建第二个短语数组
        {
            new("beguiling me", ctx => ctx.PhraseCount = 2),  // 创建短语对象并初始化文本和更新操作
            new("thrilled me"),  // 创建短语对象并初始化文本
            new("still sitting....", ctx => ctx.SkipNextComma()),  // 创建短语对象并初始化文本和更新操作
            new("never flitting", ctx => ctx.PhraseCount = 2),  // 创建短语对象并初始化文本和更新操作
            new("burned")  // 创建短语对象并初始化文本
        },
        new Phrase[]  // 创建第三个短语数组
        {
            new("and my soul"),  // 创建短语对象并初始化文本
            new("darkness there"),  // 创建短语对象并初始化文本
            new("shall be lifted"),  // 创建短语对象并初始化文本
            new("quoth the raven"),  // 创建短语对象并初始化文本
            new(ctx => ctx.PhraseCount != 0, "sign of parting")  // 创建短语对象并初始化条件和文本
        },
        new Phrase[]  // 创建第四个短语数组
        {
            new("nothing more"),  // 创建短语对象并初始化文本
            new("yet again"),  // 创建短语对象并初始化文本
            new("slowly creeping"),  // 创建短语对象并初始化文本
            new("...evermore"),  // 创建短语对象并初始化文本
            new("nevermore")  // 创建短语对象并初始化文本
        }
    };

    private readonly Predicate<Context> _condition;  // 创建条件委托字段
    private readonly string _text;  // 创建文本字段
    private readonly Action<Context> _update;  // 创建更新操作委托字段

    private Phrase(Predicate<Context> condition, string text)  // 创建构造函数，接受条件委托和文本
        : this(condition, text, _ => { })  // 调用另一个构造函数并传入默认的更新操作

    private Phrase(string text, Action<Context> update)  // 创建构造函数，接受文本和更新操作委托
        : this(_ => true, text, update)  // 调用另一个构造函数并传入默认的条件委托

    private Phrase(string text)  // 创建构造函数，只接受文本
        : this(_ => true, text, _ => { })  // 调用另一个构造函数并传入默认的条件委托和更新操作

    private Phrase(Predicate<Context> condition, string text, Action<Context> update)  // 创建构造函数，接受条件委托、文本和更新操作委托
    {
        _condition = condition;  // 初始化条件委托字段
        _text = text;  // 初始化文本字段
        _update = update;  // 初始化更新操作委托字段
    }

    public static Phrase GetPhrase(Context context) => _phrases[context.GroupNumber][context.PhraseNumber];  // 静态方法，根据上下文返回对应的短语对象

    public void Write(IReadWrite io, Context context)  // 写入方法，接受读写接口和上下文对象
    {
        # 如果条件满足，则调用context的MaybeCapitalise方法对_text进行处理，并将结果写入io
        if (_condition.Invoke(context))
        {
            io.Write(context.MaybeCapitalise(_text));
        }
        # 调用_update对context进行更新操作
        _update.Invoke(context);
    }
# 闭合大括号，表示代码块的结束
```