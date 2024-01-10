# `basic-computer-games\14_Bowling\csharp\Pins.cs`

```
namespace Bowling
{
    public class Pins
    {
        // 枚举表示保龄球瓶的状态：站立或倒下
        public enum State { Up, Down };
        // 总瓶数
        public static readonly int TotalPinCount = 10;
        // 随机数生成器
        private readonly Random random = new();

        // 瓶子状态数组
        private State[] PinSet { get; set; }

        // 构造函数，初始化瓶子状态数组
        public Pins()
        {
            PinSet = new State[TotalPinCount];
        }
        // 获取或设置指定位置的瓶子状态
        public State this[int i]
        {
            get { return PinSet[i]; }
            set { PinSet[i] = value; }
        }
        // 掷球方法
        public int Roll()
        {
            // 使用模 '15' 系统生成球的算法
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
        // 重置瓶子状态数组，全部设置为站立
        public void Reset()
        {
            for (int i = 0; i < PinSet.Length; ++i)
            {
                PinSet[i] = State.Up;
            }
        }
        // 获取倒下的瓶子数量
        public int GetPinsDown()
        {
            return PinSet.Count(p => p == State.Down);
        }
    }
}
```