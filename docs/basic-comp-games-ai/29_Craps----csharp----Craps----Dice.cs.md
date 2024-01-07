# `basic-computer-games\29_Craps\csharp\Craps\Dice.cs`

```

// 引入 System 命名空间
using System;

// 声明 Craps 命名空间
namespace Craps
{
    // 声明 Dice 类
    public class Dice
    {
        // 声明私有的 Random 对象 rand
        private Random rand = new Random();
        // 声明只读的 sides 变量
        public readonly int sides;

        // 无参构造函数，将 sides 初始化为 6
        public Dice()
        {
            sides = 6;
        }

        // 有参构造函数，接受 sides 参数并赋值给 this.sides
        public Dice(int sides)
        {
            this.sides = sides;
        }

        // Roll 方法，返回一个 1 到 sides 之间的随机数
        public int Roll() => rand.Next(1, sides + 1);
    }
}

```