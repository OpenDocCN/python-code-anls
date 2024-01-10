# `basic-computer-games\30_Cube\csharp\Game.cs`

```
namespace Cube;

internal class Game
{
    private const int _initialBalance = 500;  # 设置初始余额为500
    private readonly IEnumerable<(int, int, int)> _seeds = new List<(int, int, int)>  # 创建一个包含元组的只读列表
    {
        (3, 2, 3), (1, 3, 3), (3, 3, 2), (3, 2, 3), (3, 1, 3)  # 列表中包含五个元组
    };
    private readonly (float, float, float) _startLocation = (1, 1, 1);  # 设置初始位置坐标
    private readonly (float, float, float) _goalLocation = (3, 3, 3);  # 设置目标位置坐标

    private readonly IReadWrite _io;  # 创建只读的IReadWrite接口对象
    private readonly IRandom _random;  # 创建只读的IRandom接口对象

    public Game(IReadWrite io, IRandom random)  # 构造函数，接受IReadWrite和IRandom对象
    {
        _io = io;  # 初始化_io对象
        _random = random;  # 初始化_random对象
    }

    public void Play()  # 游戏进行方法
    {
        _io.Write(Streams.Introduction);  # 输出游戏介绍

        if (_io.ReadNumber("") != 0)  # 如果输入的数字不为0
        {
            _io.Write(Streams.Instructions);  # 输出游戏指南
        }

        PlaySeries(_initialBalance);  # 进行游戏系列

        _io.Write(Streams.Goodbye);  # 输出结束语
    }

    private void PlaySeries(float balance)  # 游戏系列方法，接受余额参数
    {
        while (true)  # 无限循环
        {
            var wager = _io.ReadWager(balance);  # 读取下注金额

            var gameWon = PlayGame();  # 进行游戏

            if (wager.HasValue)  # 如果下注金额有值
            {
                balance = gameWon ? (balance + wager.Value) : (balance - wager.Value);  # 根据游戏结果更新余额
                if (balance <= 0)  # 如果余额小于等于0
                {
                    _io.Write(Streams.Bust);  # 输出破产信息
                    return;  # 结束游戏系列
                }
                _io.WriteLine(Formats.Balance, balance);  # 输出余额信息
            }

            if (_io.ReadNumber(Prompts.TryAgain) != 1) { return; }  # 如果输入的数字不为1，结束游戏系列
        }
    }

    private bool PlayGame()  # 进行游戏方法
    {
        // 从种子列表中选择种子，并根据种子生成随机位置的集合
        var mineLocations = _seeds.Select(seed => _random.NextLocation(seed)).ToHashSet();
        // 初始化当前位置为起始位置
        var currentLocation = _startLocation;
        // 初始化提示信息为"YourMove"
        var prompt = Prompts.YourMove;
    
        // 进入游戏循环
        while (true)
        {
            // 从输入中读取3个数字作为新位置
            var newLocation = _io.Read3Numbers(prompt);
    
            // 检查新位置是否合法，如果不合法则返回失败
            if (!MoveIsLegal(currentLocation, newLocation)) { return Lose(Streams.IllegalMove); }
    
            // 更新当前位置为新位置
            currentLocation = newLocation;
    
            // 如果当前位置等于目标位置，则返回成功
            if (currentLocation == _goalLocation) { return Win(Streams.Congratulations); }
    
            // 如果当前位置是地雷位置，则返回失败
            if (mineLocations.Contains(currentLocation)) { return Lose(Streams.Bang); }
    
            // 更新提示信息为"NextMove"
            prompt = Prompts.NextMove;
        }
    }
    
    // 返回失败，并输出指定的文本流
    private bool Lose(Stream text)
    {
        _io.Write(text);
        return false;
    }
    
    // 返回成功，并输出指定的文本流
    private bool Win(Stream text)
    {
        _io.Write(text);
        return true;
    }
    
    // 判断移动是否合法
    private bool MoveIsLegal((float, float, float) from, (float, float, float) to)
        => (to.Item1 - from.Item1, to.Item2 - from.Item2, to.Item3 - from.Item3) switch
        {
            // 判断移动是否超出范围，超出范围则返回false
            ( > 1, _, _) => false,
            (_, > 1, _) => false,
            (_, _, > 1) => false,
            (1, 1, _) => false,
            (1, _, 1) => false,
            (_, 1, 1) => false,
            // 其他情况返回true
            _ => true
        };
    }
# 闭合前面的函数定义
```