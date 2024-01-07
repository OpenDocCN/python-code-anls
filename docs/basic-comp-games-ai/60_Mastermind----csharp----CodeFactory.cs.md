# `basic-computer-games\60_Mastermind\csharp\CodeFactory.cs`

```

// 引入命名空间
using System;
using System.Collections.Generic;
using System.Linq;

// 命名空间 Game
namespace Game
{
    /// <summary>
    /// 提供生成指定位置和颜色的代码的方法
    /// </summary>
    public class CodeFactory
    {
        /// <summary>
        /// 获取此工厂生成的代码中颜色的数量
        /// </summary>
        public int Colors { get; }

        /// <summary>
        /// 获取此工厂生成的代码中位置的数量
        /// </summary>
        public int Positions { get; }

        /// <summary>
        /// 获取此工厂可以生成的不同代码的数量
        /// </summary>
        public int Possibilities { get; }

        /// <summary>
        /// 初始化 CodeFactory 类的新实例
        /// </summary>
        /// <param name="positions">
        /// 位置的数量
        /// </param>
        /// <param name="colors">
        /// 颜色的数量
        /// </param>
        public CodeFactory(int positions, int colors)
        {
            if (positions < 1)
                throw new ArgumentException("A code must contain at least one position");

            if (colors < 1)
                throw new ArgumentException("A code must contain at least one color");

            if (colors > Game.Colors.List.Length)
                throw new ArgumentException($"A code can contain no more than {Game.Colors.List.Length} colors");

            Positions     = positions;
            Colors        = colors;
            Possibilities = (int)Math.Pow(colors, positions);
        }

        /// <summary>
        /// 创建指定的代码
        /// </summary>
        /// <param name="number">
        /// 要创建的代码的编号，从 0 到 Possibilities - 1
        /// </param>
        public Code Create(int number) =>
            EnumerateCodes().Skip(number).First();

        /// <summary>
        /// 使用提供的随机数生成器创建随机代码
        /// </summary>
        /// <param name="random">
        /// 随机数生成器
        /// </param>
        public Code Create(Random random) =>
            Create(random.Next(Possibilities));

        /// <summary>
        /// 生成包含此工厂可以精确创建的每个代码的代码集合
        /// </summary>
        public IEnumerable<Code> EnumerateCodes()
        {
            var current = new int[Positions];
            var position = default(int);

            do
            {
                yield return new Code(current);

                position = 0;
                while (position < Positions && ++current[position] == Colors)
                    current[position++] = 0;
            }
            while (position < Positions);
        }
    }
}

```