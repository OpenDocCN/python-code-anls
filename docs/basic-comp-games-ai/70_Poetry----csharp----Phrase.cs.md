# `basic-computer-games\70_Poetry\csharp\Phrase.cs`

```

namespace Poetry;

internal class Phrase
{
    // 定义一个二维数组，用于存储诗句
    private readonly static Phrase[][] _phrases = new Phrase[][]
    {
        // 第一组诗句
        new Phrase[]
        {
            new("midnight dreary"),
            new("fiery eyes"),
            new("bird or fiend"),
            new("thing of evil"),
            new("prophet")
        },
        // 第二组诗句
        new Phrase[]
        {
            new("beguiling me", ctx => ctx.PhraseCount = 2),
            new("thrilled me"),
            new("still sitting....", ctx => ctx.SkipNextComma()),
            new("never flitting", ctx => ctx.PhraseCount = 2),
            new("burned")
        },
        // 第三组诗句
        new Phrase[]
        {
            new("and my soul"),
            new("darkness there"),
            new("shall be lifted"),
            new("quoth the raven"),
            new(ctx => ctx.PhraseCount != 0, "sign of parting")
        },
        // 第四组诗句
        new Phrase[]
        {
            new("nothing more"),
            new("yet again"),
            new("slowly creeping"),
            new("...evermore"),
            new("nevermore")
        }
    };

    private readonly Predicate<Context> _condition;
    private readonly string _text;
    private readonly Action<Context> _update;

    // 构造函数，接受条件和文本
    private Phrase(Predicate<Context> condition, string text)
        : this(condition, text, _ => { })
    {
    }

    // 构造函数，接受文本和更新操作
    private Phrase(string text, Action<Context> update)
        : this(_ => true, text, update)
    {
    }

    // 构造函数，只接受文本
    private Phrase(string text)
        : this(_ => true, text, _ => { })
    {
    }

    // 构造函数，接受条件、文本和更新操作
    private Phrase(Predicate<Context> condition, string text, Action<Context> update)
    {
        _condition = condition;
        _text = text;
        _update = update;
    }

    // 根据上下文获取诗句
    public static Phrase GetPhrase(Context context) => _phrases[context.GroupNumber][context.PhraseNumber];

    // 写入诗句到输出流
    public void Write(IReadWrite io, Context context)
    {
        // 如果满足条件，则写入诗句
        if (_condition.Invoke(context))
        {
            io.Write(context.MaybeCapitalise(_text));
        }

        // 执行更新操作
        _update.Invoke(context);
    }
}

```