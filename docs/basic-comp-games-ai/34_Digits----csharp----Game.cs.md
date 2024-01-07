# `basic-computer-games\34_Digits\csharp\Game.cs`

```

namespace Digits;

internal class GameSeries
{
    // 创建一个只读的整数列表，用于存储权重值
    private readonly IReadOnlyList<int> _weights = new List<int> { 0, 1, 3 }.AsReadOnly();

    // 创建只读的 IReadWrite 接口和 IRandom 接口的实例变量
    private readonly IReadWrite _io;
    private readonly IRandom _random;

    // 构造函数，接受 IReadWrite 接口和 IRandom 接口的实例作为参数
    public GameSeries(IReadWrite io, IRandom random)
    {
        _io = io;
        _random = random;
    }

    // 游戏进行方法
    internal void Play()
    {
        // 输出游戏介绍
        _io.Write(Streams.Introduction);

        // 如果用户输入不为0，输出游戏说明
        if (_io.ReadNumber(Prompts.ForInstructions) != 0)
        {
            _io.Write(Streams.Instructions);
        }

        // 循环进行游戏直到用户不想再试
        do
        {
            new Game(_io, _random).Play();
        } while (_io.ReadNumber(Prompts.WantToTryAgain) == 1);

        // 输出感谢信息
        _io.Write(Streams.Thanks);
    }
}

internal class Game
{
    // 创建只读的 IReadWrite 接口和 Guesser 类的实例变量
    private readonly IReadWrite _io;
    private readonly Guesser _guesser;

    // 构造函数，接受 IReadWrite 接口和 IRandom 接口的实例作为参数
    public Game(IReadWrite io, IRandom random)
    {
        _io = io;
        _guesser = new Guesser(random);
    }

    // 游戏进行方法
    public void Play()
    {
        // 初始化正确猜测次数
        var correctGuesses = 0;

        // 进行3轮猜数字游戏
        for (int round = 0; round < 3; round++)
        {
            // 从用户输入中读取10个数字
            var digits = _io.Read10Digits(Prompts.TenNumbers, Streams.TryAgain);

            // 进行数字猜测，并更新正确猜测次数
            correctGuesses = GuessDigits(digits, correctGuesses);
        }

        // 根据正确猜测次数输出不同的消息
        _io.Write(correctGuesses switch
        {
            < 10 => Streams.YouWin,
            10 => Streams.ItsATie,
            > 10 => Streams.IWin
        });
    }

    // 猜数字方法
    private int GuessDigits(IEnumerable<int> digits, int correctGuesses)
    {
        // 输出猜数字的标题
        _io.Write(Streams.Headings);

        // 遍历用户输入的数字
        foreach (var digit in digits)
        {
            // 猜测下一个数字
            var guess = _guesser.GuessNextDigit();
            // 如果猜测正确，更新正确猜测次数
            if (guess == digit) { correctGuesses++; }

            // 输出猜测结果
            _io.WriteLine(Formats.GuessResult, guess, digit, guess == digit ? "Right" : "Wrong", correctGuesses);

            // 观察实际数字
            _guesser.ObserveActualDigit(digit);
        }

        return correctGuesses;
    }
}

```