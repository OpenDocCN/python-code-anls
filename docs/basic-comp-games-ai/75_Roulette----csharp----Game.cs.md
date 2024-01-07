# `basic-computer-games\75_Roulette\csharp\Game.cs`

```

// 命名空间 Roulette，包含了游戏相关的类
namespace Roulette;

// 游戏类，包含了游戏的逻辑
internal class Game
{
    // 私有成员变量，用于输入输出、随机数生成、游戏桌、荷官
    private readonly IReadWrite _io;
    private readonly IRandom _random;
    private readonly Table _table;
    private readonly Croupier _croupier;

    // 游戏类的构造函数，初始化输入输出和随机数生成，创建游戏桌和荷官
    public Game(IReadWrite io, IRandom random)
    {
        _io = io;
        _random = random;
        _croupier = new(); // 创建荷官对象
        _table = new(_croupier, io, random); // 创建游戏桌对象
    }

    // 游戏进行方法
    public void Play()
    {
        _io.Write(Streams.Title); // 输出游戏标题

        // 如果玩家不想阅读游戏说明，则直接开始游戏
        if (!_io.ReadString(Prompts.Instructions).ToLowerInvariant().StartsWith('n'))
        {
            _io.Write(Streams.Instructions); // 输出游戏说明
        }

        // 循环进行游戏直到游戏结束
        while (_table.Play());

        // 如果玩家破产，输出破产信息并结束游戏
        if (_croupier.PlayerIsBroke)
        {
            _io.Write(Streams.LastDollar);
            _io.Write(Streams.Thanks);
            return;
        }

        // 如果庄家破产，输出庄家破产信息
        if (_croupier.HouseIsBroke)
        {
            _io.Write(Streams.BrokeHouse);
        }

        // 结算游戏结果
        _croupier.CutCheck(_io, _random);
    }
}

```