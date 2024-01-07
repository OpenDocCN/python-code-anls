# `basic-computer-games\33_Dice\csharp\RollGenerator.cs`

```

// 引入命名空间
using System;
using System.Collections.Generic;

// 定义 RollGenerator 类
namespace BasicComputerGames.Dice
{
    public class RollGenerator
    {
        // 创建静态的随机数生成器对象
        static Random _rnd = new Random();

        // 重新设置随机数生成器的种子
        public static void ReseedRNG(int seed) => _rnd = new Random(seed);

        // 生成骰子点数的序列
        public IEnumerable<(int die1, int die2)> Rolls()
        {
            // 无限循环，生成骰子点数的元组
            while (true)
            {
                yield return (_rnd.Next(1, 7), _rnd.Next(1, 7));
            }
        }
    }
}

```