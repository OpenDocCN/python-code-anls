# `basic-computer-games\05_Bagels\csharp\GameBase.cs`

```

// 引入 System 命名空间
using System;

// 定义 Bagels 命名空间下的 GameBase 类
namespace BasicComputerGames.Bagels
{
    // 定义 GameBase 类
    public class GameBase
    {
        // 创建一个 Random 对象，用于生成随机数
        protected Random Rnd { get; } = new Random();

        /// <summary>
        /// 提示玩家再次尝试，并等待他们按下 Y 或 N。
        /// </summary>
        /// <returns>如果玩家想再次尝试，则返回 true，如果他们已经完成游戏，则返回 false。</returns>
        protected bool TryAgain()
        {
            // 设置控制台前景色为白色，显示提示信息
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine("Would you like to try again? (Press 'Y' for yes or 'N' for no)");

            // 设置控制台前景色为黄色，显示输入提示符
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.Write("> ");

            char pressedKey;
            // 循环直到获得一个被识别的输入
            do
            {
                // 读取一个键，不在屏幕上显示
                ConsoleKeyInfo key = Console.ReadKey(true);
                // 转换为大写，这样我们就不需要关心大小写
                pressedKey = Char.ToUpper(key.KeyChar);
                // 这是我们认识的键吗？如果不是，继续循环
            } while (pressedKey != 'Y' && pressedKey != 'N');
            // 在屏幕上显示结果
            Console.WriteLine(pressedKey);

            // 如果玩家按下 'Y'，则返回 true，否则返回 false
            return (pressedKey == 'Y');
        }

    }
}

```