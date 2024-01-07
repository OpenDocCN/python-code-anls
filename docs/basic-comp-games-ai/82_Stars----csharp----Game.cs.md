# `basic-computer-games\82_Stars\csharp\Game.cs`

```

// 引入命名空间
using System;
using Games.Common.IO;
using Games.Common.Randomness;
using Stars.Resources;

// 定义 Game 类
namespace Stars
{
    // Game 类
    internal class Game
    {
        // 私有字段
        private readonly TextIO _io;
        private readonly IRandom _random;
        private readonly int _maxNumber;
        private readonly int _maxGuessCount;

        // 构造函数
        public Game(TextIO io, IRandom random, int maxNumber, int maxGuessCount)
        {
            _io = io;
            _random = random;
            _maxNumber = maxNumber;
            _maxGuessCount = maxGuessCount;
        }

        // Play 方法
        internal void Play(Func<bool> playAgain)
        {
            // 显示游戏介绍
            DisplayIntroduction();

            // 循环游戏直到玩家选择退出
            do
            {
                Play();
            } while (playAgain.Invoke());
        }

        // 显示游戏介绍
        private void DisplayIntroduction()
        {
            _io.Write(Resource.Streams.Title);

            // 如果玩家选择不需要说明，则直接返回
            if (_io.ReadString("Do you want instructions").Equals("N", StringComparison.InvariantCultureIgnoreCase))
            {
                return;
            }

            // 显示游戏说明
            _io.WriteLine(Resource.Formats.Instructions, _maxNumber, _maxGuessCount);
        }

        // 游戏主体
        private void Play()
        {
            _io.WriteLine();
            _io.WriteLine();

            // 生成目标数字
            var target = _random.Next(_maxNumber) + 1;

            _io.WriteLine("Ok, I am thinking of a number.  Start guessing.");

            // 接受玩家猜测
            AcceptGuesses(target);
        }

        // 接受玩家猜测
        private void AcceptGuesses(int target)
        {
            for (int guessCount = 1; guessCount <= _maxGuessCount; guessCount++)
            {
                _io.WriteLine();
                var guess = _io.ReadNumber("Your guess");

                // 如果猜对了，显示胜利信息并返回
                if (guess == target)
                {
                    DisplayWin(guessCount);
                    return;
                }

                // 否则显示猜测结果
                DisplayStars(target, guess);
            }

            // 如果猜测次数用完，显示失败信息
            DisplayLoss(target);
        }

        // 根据猜测结果显示星星
        private void DisplayStars(int target, float guess)
        {
            var stars = Math.Abs(guess - target) switch
            {
                >= 64 => "*",
                >= 32 => "**",
                >= 16 => "***",
                >= 8  => "****",
                >= 4  => "*****",
                >= 2  => "******",
                _     => "*******"
            };

            _io.WriteLine(stars);
        }

        // 显示胜利信息
        private void DisplayWin(int guessCount)
        {
            _io.WriteLine();
            _io.WriteLine(new string('*', 79));
            _io.WriteLine();
            _io.WriteLine($"You got it in {guessCount} guesses!!!  Let's play again...");
        }

        // 显示失败信息
        private void DisplayLoss(int target)
        {
            _io.WriteLine();
            _io.WriteLine($"Sorry, that's {_maxGuessCount} guesses. The number was {target}.");
        }
    }
}

```