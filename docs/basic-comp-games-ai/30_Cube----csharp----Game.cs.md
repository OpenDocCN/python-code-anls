# `basic-computer-games\30_Cube\csharp\Game.cs`

```

namespace Cube;

internal class Game
{
    // 初始余额
    private const int _initialBalance = 500;
    // 种子位置列表
    private readonly IEnumerable<(int, int, int)> _seeds = new List<(int, int, int)>
    {
        (3, 2, 3), (1, 3, 3), (3, 3, 2), (3, 2, 3), (3, 1, 3)
    };
    // 起始位置
    private readonly (float, float, float) _startLocation = (1, 1, 1);
    // 目标位置
    private readonly (float, float, float) _goalLocation = (3, 3, 3);

    // 输入输出接口
    private readonly IReadWrite _io;
    // 随机数生成器
    private readonly IRandom _random;

    // 构造函数
    public Game(IReadWrite io, IRandom random)
    {
        _io = io;
        _random = random;
    }

    // 游戏开始
    public void Play()
    {
        // 输出游戏介绍
        _io.Write(Streams.Introduction);

        // 如果输入不为0，输出游戏说明
        if (_io.ReadNumber("") != 0)
        {
            _io.Write(Streams.Instructions);
        }

        // 进行游戏系列
        PlaySeries(_initialBalance);

        // 输出结束语
        _io.Write(Streams.Goodbye);
    }

    // 进行游戏系列
    private void PlaySeries(float balance)
    {
        while (true)
        {
            // 读取赌注
            var wager = _io.ReadWager(balance);

            // 进行游戏
            var gameWon = PlayGame();

            if (wager.HasValue)
            {
                // 根据游戏结果更新余额
                balance = gameWon ? (balance + wager.Value) : (balance - wager.Value);
                // 如果余额不足，输出破产信息并结束游戏
                if (balance <= 0)
                {
                    _io.Write(Streams.Bust);
                    return;
                }
                // 输出余额信息
                _io.WriteLine(Formats.Balance, balance);
            }

            // 询问是否再玩一次
            if (_io.ReadNumber(Prompts.TryAgain) != 1) { return; }
        }
    }

    // 进行游戏
    private bool PlayGame()
    {
        // 生成地雷位置
        var mineLocations = _seeds.Select(seed => _random.NextLocation(seed)).ToHashSet();
        var currentLocation = _startLocation;
        var prompt = Prompts.YourMove;

        while (true)
        {
            // 读取新位置
            var newLocation = _io.Read3Numbers(prompt);

            // 判断移动是否合法
            if (!MoveIsLegal(currentLocation, newLocation)) { return Lose(Streams.IllegalMove); }

            currentLocation = newLocation;

            // 判断是否到达目标位置
            if (currentLocation == _goalLocation) { return Win(Streams.Congratulations); }

            // 判断是否踩到地雷
            if (mineLocations.Contains(currentLocation)) { return Lose(Streams.Bang); }

            prompt = Prompts.NextMove;
        }
    }

    // 输掉游戏
    private bool Lose(Stream text)
    {
        _io.Write(text);
        return false;
    }

    // 赢得游戏
    private bool Win(Stream text)
    {
        _io.Write(text);
        return true;
    }

    // 判断移动是否合法
    private bool MoveIsLegal((float, float, float) from, (float, float, float) to)
        => (to.Item1 - from.Item1, to.Item2 - from.Item2, to.Item3 - from.Item3) switch
        {
            ( > 1, _, _) => false,
            (_, > 1, _) => false,
            (_, _, > 1) => false,
            (1, 1, _) => false,
            (1, _, 1) => false,
            (_, 1, 1) => false,
            _ => true
        };
}

```