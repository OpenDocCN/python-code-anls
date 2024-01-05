# `d:/src/tocomm/basic-computer-games\72_Queen\csharp\Game.cs`

```
namespace Queen;  # 命名空间声明

internal class Game  # 内部类 Game 声明
{
    private readonly IReadWrite _io;  # 声明私有只读字段 _io，类型为 IReadWrite 接口
    private readonly IRandom _random;  # 声明私有只读字段 _random，类型为 IRandom 接口
    private readonly Computer _computer;  # 声明私有只读字段 _computer，类型为 Computer 类

    public Game(IReadWrite io, IRandom random)  # Game 类的构造函数，接受 IReadWrite 和 IRandom 接口类型的参数
    {
        _io = io;  # 将传入的 io 参数赋值给 _io 字段
        _random = random;  # 将传入的 random 参数赋值给 _random 字段
        _computer = new Computer(random);  # 使用传入的 random 参数创建一个新的 Computer 对象，并赋值给 _computer 字段
    }

    internal void PlaySeries()  # 内部方法 PlaySeries 声明
    {
        _io.Write(Streams.Title);  # 使用 _io 对象的 Write 方法，输出 Streams.Title 的内容
        if (_io.ReadYesNo(Prompts.Instructions)) { _io.Write(Streams.Instructions); }  # 使用 _io 对象的 ReadYesNo 方法，根据返回值决定是否输出 Streams.Instructions 的内容
        while (true)  # 创建一个无限循环，直到条件为false才会退出循环
        {
            var result = PlayGame();  # 调用PlayGame()函数，并将返回值赋给result变量
            _io.Write(result switch  # 使用result的值进行条件判断
            {
                Result.HumanForfeits => Streams.Forfeit,  # 如果result为Result.HumanForfeits，则写入Streams.Forfeit
                Result.HumanWins => Streams.Congratulations,  # 如果result为Result.HumanWins，则写入Streams.Congratulations
                Result.ComputerWins => Streams.IWin,  # 如果result为Result.ComputerWins，则写入Streams.IWin
                _ => throw new InvalidOperationException($"Unexpected result {result}")  # 如果result为其他值，则抛出异常
            });

            if (!_io.ReadYesNo(Prompts.Anyone)) { break; }  # 调用ReadYesNo()函数，根据返回值判断是否继续循环
        }

        _io.Write(Streams.Thanks);  # 循环结束后，写入Streams.Thanks
    }

    private Result PlayGame()  # 定义一个名为PlayGame的私有函数，返回类型为Result
    {
        _io.Write(Streams.Board);  # 写入Streams.Board
        # 从输入输出对象中读取人类玩家的位置，如果位置为起始位置则重复提示直到输入有效
        var humanPosition = _io.ReadPosition(Prompts.Start, p => p.IsStart, Streams.IllegalStart, repeatPrompt: true);
        # 如果人类玩家位置为零，则返回人类玩家放弃的结果
        if (humanPosition.IsZero) { return Result.HumanForfeits; }

        # 无限循环，直到游戏结束
        while (true)
        {
            # 计算计算机玩家的下一步位置
            var computerPosition = _computer.GetMove(humanPosition);
            # 输出计算机玩家的移动位置
            _io.Write(Strings.ComputerMove(computerPosition));
            # 如果计算机玩家位置为终点，则返回计算机玩家获胜的结果
            if (computerPosition.IsEnd) { return Result.ComputerWins; }

            # 从输入输出对象中读取人类玩家的下一步位置，直到输入有效
            humanPosition = _io.ReadPosition(Prompts.Move, p => (p - computerPosition).IsValid, Streams.IllegalMove);
            # 如果人类玩家位置为零，则返回人类玩家放弃的结果
            if (humanPosition.IsZero) { return Result.HumanForfeits; }
            # 如果人类玩家位置为终点，则返回人类玩家获胜的结果
            if (humanPosition.IsEnd) { return Result.HumanWins; }
        }
    }

    # 定义游戏结果的枚举类型
    private enum Result { ComputerWins, HumanWins, HumanForfeits };
}
```