# `basic-computer-games\72_Queen\csharp\Game.cs`

```
namespace Queen;

internal class Game
{
    private readonly IReadWrite _io;  // 用于输入输出的接口
    private readonly IRandom _random;  // 用于生成随机数的接口
    private readonly Computer _computer;  // 电脑玩家对象

    public Game(IReadWrite io, IRandom random)
    {
        _io = io;  // 初始化输入输出接口
        _random = random;  // 初始化随机数接口
        _computer = new Computer(random);  // 初始化电脑玩家对象
    }

    internal void PlaySeries()
    {
        _io.Write(Streams.Title);  // 输出游戏标题
        if (_io.ReadYesNo(Prompts.Instructions)) { _io.Write(Streams.Instructions); }  // 如果玩家选择查看游戏说明，则输出游戏说明

        while (true)
        {
            var result = PlayGame();  // 进行游戏
            _io.Write(result switch  // 根据游戏结果输出不同的消息
            {
                Result.HumanForfeits => Streams.Forfeit,
                Result.HumanWins => Streams.Congratulations,
                Result.ComputerWins => Streams.IWin,
                _ => throw new InvalidOperationException($"Unexpected result {result}")  // 如果结果不在预期范围内，则抛出异常
            });

            if (!_io.ReadYesNo(Prompts.Anyone)) { break; }  // 如果玩家选择结束游戏，则退出循环
        }

        _io.Write(Streams.Thanks);  // 输出感谢消息
    }

    private Result PlayGame()
    {
        _io.Write(Streams.Board);  // 输出游戏棋盘
        var humanPosition = _io.ReadPosition(Prompts.Start, p => p.IsStart, Streams.IllegalStart, repeatPrompt: true);  // 读取玩家的起始位置
        if (humanPosition.IsZero) { return Result.HumanForfeits; }  // 如果玩家选择放弃，则返回玩家放弃的结果

        while (true)
        {
            var computerPosition = _computer.GetMove(humanPosition);  // 获取电脑玩家的移动位置
            _io.Write(Strings.ComputerMove(computerPosition));  // 输出电脑玩家的移动位置
            if (computerPosition.IsEnd) { return Result.ComputerWins; }  // 如果电脑玩家到达终点，则返回电脑玩家获胜的结果

            humanPosition = _io.ReadPosition(Prompts.Move, p => (p - computerPosition).IsValid, Streams.IllegalMove);  // 读取玩家的移动位置
            if (humanPosition.IsZero) { return Result.HumanForfeits; }  // 如果玩家选择放弃，则返回玩家放弃的结果
            if (humanPosition.IsEnd) { return Result.HumanWins; }  // 如果玩家到达终点，则返回玩家获胜的结果
        }
    }

    private enum Result { ComputerWins, HumanWins, HumanForfeits };  // 定义游戏结果的枚举类型
}
```