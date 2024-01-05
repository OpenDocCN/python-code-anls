# `77_Salvo\csharp\Game.cs`

```
namespace Salvo;

internal class Game 
{
    private readonly IReadWrite _io;  # 声明私有的 IReadWrite 接口类型的变量 _io
    private readonly IRandom _random;  # 声明私有的 IRandom 接口类型的变量 _random

    public Game(IReadWrite io, IRandom random)  # 构造函数，接受 IReadWrite 和 IRandom 接口类型的参数
    {
        _io = io;  # 将传入的 io 参数赋值给 _io 变量
        _random = random;  # 将传入的 random 参数赋值给 _random 变量
    }

    internal void Play()  # Play 方法
    {
        _io.Write(Streams.Title);  # 调用 _io 的 Write 方法，输出 Streams.Title

        var turnHandler = new TurnHandler(_io, _random);  # 创建 TurnHandler 对象，传入 _io 和 _random 参数
        _io.WriteLine();  # 调用 _io 的 WriteLine 方法，输出空行
        Winner? winner;  // 声明一个名为 winner 的 Winner 类型的变量
        do 
        {
            winner = turnHandler.PlayTurn();  // 调用 turnHandler 的 PlayTurn 方法，并将返回值赋给 winner 变量
        } while (winner == null);  // 当 winner 变量为 null 时循环执行上述代码块

        _io.Write(winner == Winner.Computer ? Streams.IWon : Streams.YouWon);  // 根据 winner 的值，调用 _io 的 Write 方法输出对应的消息
    }
}
```