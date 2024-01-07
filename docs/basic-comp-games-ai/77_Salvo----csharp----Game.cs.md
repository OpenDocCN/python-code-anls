# `basic-computer-games\77_Salvo\csharp\Game.cs`

```

// 命名空间 Salvo
namespace Salvo;

// 内部类 Game
internal class Game 
{
    // 只读字段 _io 和 _random
    private readonly IReadWrite _io;
    private readonly IRandom _random;

    // 构造函数，接受 IReadWrite 和 IRandom 作为参数
    public Game(IReadWrite io, IRandom random)
    {
        _io = io;
        _random = random;
    }

    // 内部方法 Play
    internal void Play()
    {
        // 使用 _io 输出 Streams.Title
        _io.Write(Streams.Title);

        // 创建 TurnHandler 对象，传入 _io 和 _random
        var turnHandler = new TurnHandler(_io, _random);
        // 使用 _io 输出空行
        _io.WriteLine();

        // 声明 winner 变量，类型为 Winner 枚举的可空类型
        Winner? winner;
        // 循环执行以下代码，直到 winner 不为 null
        do 
        {
            // 调用 turnHandler 的 PlayTurn 方法，将返回值赋给 winner
            winner = turnHandler.PlayTurn();
        } while (winner == null);

        // 根据 winner 的值，使用 _io 输出相应的消息
        _io.Write(winner == Winner.Computer ? Streams.IWon : Streams.YouWon);
    }
}

```