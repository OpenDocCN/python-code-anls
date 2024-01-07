# `basic-computer-games\72_Queen\csharp\Game.cs`

```

namespace Queen;

// Queen 命名空间下的 Game 类
internal class Game
{
    // 私有成员变量，用于读写和随机数生成
    private readonly IReadWrite _io;
    private readonly IRandom _random;
    private readonly Computer _computer;

    // Game 类的构造函数，接受 IReadWrite 和 IRandom 接口实例
    public Game(IReadWrite io, IRandom random)
    {
        _io = io;
        _random = random;
        _computer = new Computer(random);
    }

    // PlaySeries 方法，用于进行游戏系列
    internal void PlaySeries()
    {
        // 输出游戏标题
        _io.Write(Streams.Title);
        // 如果玩家选择阅读游戏说明，则输出游戏说明
        if (_io.ReadYesNo(Prompts.Instructions)) { _io.Write(Streams.Instructions); }

        // 循环进行游戏
        while (true)
        {
            // 进行游戏并获取结果
            var result = PlayGame();
            // 根据游戏结果输出不同的消息
            _io.Write(result switch
            {
                Result.HumanForfeits => Streams.Forfeit,
                Result.HumanWins => Streams.Congratulations,
                Result.ComputerWins => Streams.IWin,
                _ => throw new InvalidOperationException($"Unexpected result {result}")
            });

            // 如果玩家选择继续游戏，则继续循环，否则跳出循环
            if (!_io.ReadYesNo(Prompts.Anyone)) { break; }
        }

        // 输出感谢消息
        _io.Write(Streams.Thanks);
    }

    // PlayGame 方法，用于进行单局游戏
    private Result PlayGame()
    {
        // 输出游戏棋盘
        _io.Write(Streams.Board);
        // 玩家选择起始位置
        var humanPosition = _io.ReadPosition(Prompts.Start, p => p.IsStart, Streams.IllegalStart, repeatPrompt: true);
        // 如果玩家选择放弃，则返回 HumanForfeits 结果
        if (humanPosition.IsZero) { return Result.HumanForfeits; }

        // 循环进行游戏
        while (true)
        {
            // 计算计算机的移动位置
            var computerPosition = _computer.GetMove(humanPosition);
            // 输出计算机的移动位置
            _io.Write(Strings.ComputerMove(computerPosition));
            // 如果计算机到达终点，则返回 ComputerWins 结果
            if (computerPosition.IsEnd) { return Result.ComputerWins; }

            // 玩家选择移动位置
            humanPosition = _io.ReadPosition(Prompts.Move, p => (p - computerPosition).IsValid, Streams.IllegalMove);
            // 如果玩家选择放弃，则返回 HumanForfeits 结果
            if (humanPosition.IsZero) { return Result.HumanForfeits; }
            // 如果玩家到达终点，则返回 HumanWins 结果
            if (humanPosition.IsEnd) { return Result.HumanWins; }
        }
    }

    // 游戏结果枚举
    private enum Result { ComputerWins, HumanWins, HumanForfeits };
}

```