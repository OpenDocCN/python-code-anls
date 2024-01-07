# `basic-computer-games\41_Guess\csharp\Game.cs`

```

// 命名空间 Guess 下的内部类 Game
internal class Game
{
    // 读写接口和随机数生成器
    private readonly IReadWrite _io;
    private readonly IRandom _random;

    // 构造函数，初始化读写接口和随机数生成器
    public Game(IReadWrite io, IRandom random)
    {
        _io = io;
        _random = random;
    }

    // 游戏主体
    public void Play()
    {
        // 游戏循环
        while (true)
        {
            // 输出游戏介绍
            _io.Write(Streams.Introduction);

            // 读取用户输入的限制值
            var limit = _io.ReadNumber(Prompts.Limit);
            _io.WriteLine();

            // 计算目标猜测次数，存在原始代码中的一个 bug，如果输入的限制值 <= 0，则程序会崩溃
            var targetGuessCount = checked((int)Math.Log2(limit) + 1);

            // 进行猜数字游戏
            PlayGuessingRounds(limit, targetGuessCount);

            // 输出空行
            _io.Write(Streams.BlankLines);
        }
    }

    // 猜数字游戏
    private void PlayGuessingRounds(float limit, int targetGuessCount)
    {
        // 游戏循环
        while (true)
        {
            // 输出提示信息
            _io.WriteLine(Formats.Thinking, limit);

            // 存在原始代码中的一个 bug，如果输入的限制值为非整数，则可能导致秘密数字为大于限制值的下一个整数
            var secretNumber = (int)_random.NextFloat(limit) + 1;

            var guessCount = 0;

            // 猜数字循环
            while (true)
            {
                // 读取用户猜测的数字
                var guess = _io.ReadNumber("");
                if (guess <= 0) { return; } // 如果猜测值小于等于0，则结束游戏
                guessCount++;
                if (IsGuessCorrect(guess, secretNumber)) { break; } // 如果猜测正确，则跳出循环
            }

            // 报告结果
            ReportResult(guessCount, targetGuessCount);

            // 输出空行
            _io.Write(Streams.BlankLines);
        }
    }

    // 判断猜测是否正确
    private bool IsGuessCorrect(float guess, int secretNumber)
    {
        if (guess < secretNumber) { _io.Write(Streams.TooLow); } // 如果猜测值小于秘密数字，输出提示信息
        if (guess > secretNumber) { _io.Write(Streams.TooHigh); } // 如果猜测值大于秘密数字，输出提示信息

        return guess == secretNumber; // 返回猜测是否正确的结果
    }

    // 报告结果
    private void ReportResult(int guessCount, int targetGuessCount)
    {
        _io.WriteLine(Formats.ThatsIt, guessCount); // 输出猜测次数
        _io.WriteLine(
            (guessCount - targetGuessCount) switch
            {
                < 0 => Strings.VeryGood, // 如果猜测次数小于目标次数，输出提示信息
                0 => Strings.Good, // 如果猜测次数等于目标次数，输出提示信息
                > 0 => string.Format(Formats.ShouldHave, targetGuessCount) // 如果猜测次数大于目标次数，输出提示信息
            });
    }
}

```