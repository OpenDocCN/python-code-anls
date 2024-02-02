# `basic-computer-games\77_Salvo\csharp\Game.cs`

```py
// 命名空间 Salvo 中的内部类 Game
internal class Game 
{
    // 只读字段 _io，用于读写操作
    private readonly IReadWrite _io;
    // 只读字段 _random，用于生成随机数
    private readonly IRandom _random;

    // 构造函数，接受 IReadWrite 和 IRandom 接口实例作为参数
    public Game(IReadWrite io, IRandom random)
    {
        // 将传入的 io 赋值给 _io
        _io = io;
        // 将传入的 random 赋值给 _random
        _random = random;
    }

    // 游戏进行方法
    internal void Play()
    {
        // 使用 _io 输出 Streams 中的标题
        _io.Write(Streams.Title);

        // 创建一个 TurnHandler 实例，传入 _io 和 _random
        var turnHandler = new TurnHandler(_io, _random);
        // 使用 _io 输出空行
        _io.WriteLine();

        // 声明 winner 变量，用于存储游戏结果
        Winner? winner;
        // 循环执行游戏回合，直到有获胜者产生
        do 
        {
            // 调用 turnHandler 的 PlayTurn 方法，获取游戏结果
            winner = turnHandler.PlayTurn();
        } while (winner == null);

        // 根据获胜者输出不同的消息
        _io.Write(winner == Winner.Computer ? Streams.IWon : Streams.YouWon);
    }
}
```