# `basic-computer-games\14_Bowling\csharp\Pins.cs`

```

// 引入命名空间
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// 声明 Bowling 命名空间
namespace Bowling
{
    // 定义 Pins 类
    public class Pins
    {
        // 定义枚举类型 State，表示保龄球瓶的状态
        public enum State { Up, Down };
        // 定义 TotalPinCount 常量，表示保龄球瓶的总数
        public static readonly int TotalPinCount = 10;
        // 创建 Random 对象
        private readonly Random random = new();

        // 创建 PinSet 属性，表示保龄球瓶的状态数组
        private State[] PinSet { get; set; }

        // 构造函数，初始化 PinSet 数组
        public Pins()
        {
            PinSet = new State[TotalPinCount];
        }
        // 索引器，用于获取和设置指定位置的保龄球瓶状态
        public State this[int i]
        {
            get { return PinSet[i]; }
            set { PinSet[i] = value; }
        }
        // 掷球方法，模拟保龄球的投掷过程
        public int Roll()
        {
            // 使用模 '15' 系统生成球的位置
            for (int i = 0; i < 20; ++i)
            {
                var x = random.Next(100) + 1;
                int j;
                for (j = 1; j <= 10; ++j)
                {
                    if (x < 15 * j)
                        break;
                }
                var pindex = 15 * j - x;
                if (pindex > 0 && pindex <= TotalPinCount)
                    PinSet[--pindex] = State.Down;
            }
            return GetPinsDown();
        }
        // 重置保龄球瓶状态
        public void Reset()
        {
            for (int i = 0; i < PinSet.Length; ++i)
            {
                PinSet[i] = State.Up;
            }
        }
        // 获取倒下的保龄球瓶数量
        public int GetPinsDown()
        {
            return PinSet.Count(p => p == State.Down);
        }
    }
}

```