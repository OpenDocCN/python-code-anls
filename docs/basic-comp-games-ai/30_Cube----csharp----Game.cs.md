# `30_Cube\csharp\Game.cs`

```
namespace Cube;  // 命名空间声明

internal class Game  // 内部类 Game 声明
{
    private const int _initialBalance = 500;  // 声明私有常量 _initialBalance 并赋值为 500
    private readonly IEnumerable<(int, int, int)> _seeds = new List<(int, int, int)>  // 声明只读字段 _seeds，类型为元组的集合，并初始化为包含元组的列表
    {
        (3, 2, 3), (1, 3, 3), (3, 3, 2), (3, 2, 3), (3, 1, 3)  // 初始化 _seeds 列表
    };
    private readonly (float, float, float) _startLocation = (1, 1, 1);  // 声明只读字段 _startLocation，类型为元组，并初始化为 (1, 1, 1)
    private readonly (float, float, float) _goalLocation = (3, 3, 3);  // 声明只读字段 _goalLocation，类型为元组，并初始化为 (3, 3, 3)

    private readonly IReadWrite _io;  // 声明只读字段 _io，类型为 IReadWrite 接口
    private readonly IRandom _random;  // 声明只读字段 _random，类型为 IRandom 接口

    public Game(IReadWrite io, IRandom random)  // Game 类的构造函数，接受 IReadWrite 和 IRandom 接口类型的参数
    {
        _io = io;  // 将传入的 io 参数赋值给 _io 字段
        _random = random;  // 将传入的 random 参数赋值给 _random 字段
    }
}
    # 定义一个名为 Play 的公共方法
    public void Play()
    {
        # 使用 _io 对象的 Write 方法输出介绍信息
        _io.Write(Streams.Introduction);

        # 如果用户输入的数字不等于 0
        if (_io.ReadNumber("") != 0)
        {
            # 使用 _io 对象的 Write 方法输出游戏说明
            _io.Write(Streams.Instructions);
        }

        # 调用 PlaySeries 方法，传入初始余额作为参数
        PlaySeries(_initialBalance);

        # 使用 _io 对象的 Write 方法输出道别信息
        _io.Write(Streams.Goodbye);
    }

    # 定义一个名为 PlaySeries 的私有方法，接受一个浮点数余额作为参数
    private void PlaySeries(float balance)
    {
        # 创建一个无限循环
        while (true)
        {
            # 使用 _io 对象的 ReadWager 方法读取用户的赌注，传入余额作为参数
            # 调用 PlayGame() 方法，将返回值赋给 gameWon 变量
            var gameWon = PlayGame();

            # 如果有下注金额
            if (wager.HasValue)
            {
                # 根据游戏结果更新余额
                balance = gameWon ? (balance + wager.Value) : (balance - wager.Value)
                # 如果余额不足，输出 Bust 消息并返回
                if (balance <= 0)
                {
                    _io.Write(Streams.Bust);
                    return;
                }
                # 输出更新后的余额
                _io.WriteLine(Formats.Balance, balance);
            }

            # 如果用户不想再玩游戏，直接返回
            if (_io.ReadNumber(Prompts.TryAgain) != 1) { return; }
        }
    }

    # PlayGame 方法，用于执行游戏并返回结果
    private bool PlayGame()
    {
        // 从种子列表中选择一个种子，然后根据种子生成一个随机位置的集合
        var mineLocations = _seeds.Select(seed => _random.NextLocation(seed)).ToHashSet();
        // 设置当前位置为起始位置
        var currentLocation = _startLocation;
        // 设置提示信息为“YourMove”
        var prompt = Prompts.YourMove;

        // 进入游戏循环
        while (true)
        {
            // 从输入中读取3个数字作为新位置
            var newLocation = _io.Read3Numbers(prompt);

            // 如果新位置不合法，返回失败信息
            if (!MoveIsLegal(currentLocation, newLocation)) { return Lose(Streams.IllegalMove); }

            // 更新当前位置为新位置
            currentLocation = newLocation;

            // 如果当前位置等于目标位置，返回胜利信息
            if (currentLocation == _goalLocation) { return Win(Streams.Congratulations); }

            // 如果当前位置是地雷位置，返回失败信息
            if (mineLocations.Contains(currentLocation)) { return Lose(Streams.Bang); }

            // 更新提示信息为“NextMove”
            prompt = Prompts.NextMove;
        }
    }
    private bool Lose(Stream text)
    {
        _io.Write(text);  // 将文本写入到 _io 对象中
        return false;  // 返回 false
    }

    private bool Win(Stream text)
    {
        _io.Write(text);  // 将文本写入到 _io 对象中
        return true;  // 返回 true
    }

    private bool MoveIsLegal((float, float, float) from, (float, float, float) to)
        => (to.Item1 - from.Item1, to.Item2 - from.Item2, to.Item3 - from.Item3) switch  // 根据移动的坐标计算差值并进行匹配
        {
            ( > 1, _, _) => false,  // 如果 x 轴方向的差值大于 1，则返回 false
            (_, > 1, _) => false,  // 如果 y 轴方向的差值大于 1，则返回 false
            (_, _, > 1) => false,  // 如果 z 轴方向的差值大于 1，则返回 false
            (1, 1, _) => false,  // 如果 x 轴和 y 轴方向的差值都为 1，则返回 false
            (1, _, 1) => false,  // 如果 x 轴和 z 轴方向的差值都为 1，则返回 false
# 定义一个匿名函数，接受三个参数，如果第一个参数为下划线，第二个参数为1，第三个参数为1，则返回false
# 否则返回true
```