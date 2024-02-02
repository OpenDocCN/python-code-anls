# `basic-computer-games\41_Guess\csharp\Game.cs`

```py
namespace Guess;

internal class Game
{
    private readonly IReadWrite _io;  // 声明私有字段_io，用于输入输出操作
    private readonly IRandom _random;  // 声明私有字段_random，用于生成随机数

    public Game(IReadWrite io, IRandom random)  // 构造函数，接受io和random对象作为参数
    {
        _io = io;  // 初始化_io字段
        _random = random;  // 初始化_random字段
    }

    public void Play()  // 游戏主逻辑
    {
        while (true)  // 无限循环
        {
            _io.Write(Streams.Introduction);  // 输出游戏介绍

            var limit = _io.ReadNumber(Prompts.Limit);  // 读取用户输入的限制值
            _io.WriteLine();  // 输出空行

            // There's a bug here that exists in the original code. 
            // If the limit entered is <= 0 then the program will crash.
            var targetGuessCount = checked((int)Math.Log2(limit) + 1);  // 计算目标猜测次数

            PlayGuessingRounds(limit, targetGuessCount);  // 调用PlayGuessingRounds方法进行猜数游戏

            _io.Write(Streams.BlankLines);  // 输出空行
        }
    }

    private void PlayGuessingRounds(float limit, int targetGuessCount)  // 猜数游戏逻辑
    {
        while (true)  // 无限循环
        {
            _io.WriteLine(Formats.Thinking, limit);  // 输出提示信息

            // There's a bug here that exists in the original code. If a non-integer is entered as the limit
            // then it's possible for the secret number to be the next integer greater than the limit.
            var secretNumber = (int)_random.NextFloat(limit) + 1;  // 生成秘密数字

            var guessCount = 0;  // 初始化猜测次数为0

            while (true)  // 无限循环
            {
                var guess = _io.ReadNumber("");  // 读取用户猜测的数字
                if (guess <= 0) { return; }  // 如果猜测值小于等于0，则退出当前游戏
                guessCount++;  // 猜测次数加1
                if (IsGuessCorrect(guess, secretNumber)) { break; }  // 如果猜测正确，则跳出循环
            }

            ReportResult(guessCount, targetGuessCount);  // 报告猜测结果

            _io.Write(Streams.BlankLines);  // 输出空行
        }
    }

    private bool IsGuessCorrect(float guess, int secretNumber)  // 判断猜测是否正确
    {
        if (guess < secretNumber) { _io.Write(Streams.TooLow); }  // 如果猜测值小于秘密数字，输出提示信息
        if (guess > secretNumber) { _io.Write(Streams.TooHigh); }  // 如果猜测值大于秘密数字，输出提示信息

        return guess == secretNumber;  // 返回猜测是否正确的结果
    }

    private void ReportResult(int guessCount, int targetGuessCount)  // 报告猜测结果
    {
        # 输出猜测次数的消息
        _io.WriteLine(Formats.ThatsIt, guessCount);
        # 根据猜测次数与目标次数的差值输出不同的消息
        _io.WriteLine(
            (guessCount - targetGuessCount) switch
            {
                # 如果差值小于0，输出"非常好"
                < 0 => Strings.VeryGood,
                # 如果差值等于0，输出"好"
                0 => Strings.Good,
                # 如果差值大于0，输出"应该是{targetGuessCount}次"
                > 0 => string.Format(Formats.ShouldHave, targetGuessCount)
            });
    }
# 闭合了一个代码块
```