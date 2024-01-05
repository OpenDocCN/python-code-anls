# `14_Bowling\csharp\Pins.cs`

```
# 引入所需的命名空间
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Bowling
{
    # 创建名为Pins的公共类
    public class Pins
    {
        # 创建枚举类型State，包含Up和Down两个状态
        public enum State { Up, Down };
        # 创建只读的静态整数变量TotalPinCount，并赋值为10
        public static readonly int TotalPinCount = 10;
        # 创建只读的Random对象random，并使用默认构造函数进行初始化
        private readonly Random random = new();

        # 创建私有的State数组PinSet，并提供读写访问权限
        private State[] PinSet { get; set; }

        # 创建Pins类的构造函数
        public Pins()
        {
            # 初始化PinSet数组，长度为TotalPinCount
            PinSet = new State[TotalPinCount];
        }
        public State this[int i]  // 定义索引器，用于访问 PinSet 数组中的元素
        {
            get { return PinSet[i]; }  // 获取 PinSet 数组中指定索引位置的元素
            set { PinSet[i] = value; }  // 设置 PinSet 数组中指定索引位置的元素的值
        }
        public int Roll()  // 定义 Roll 方法，用于模拟掷球
        {
            // REM ARK BALL GENERATOR USING MOD '15' SYSTEM
            // 使用模 '15' 系统生成随机数
            for (int i = 0; i < 20; ++i)  // 循环 20 次
            {
                var x = random.Next(100) + 1;  // 生成 1 到 100 之间的随机数
                int j;
                for (j = 1; j <= 10; ++j)  // 循环 10 次
                {
                    if (x < 15 * j)  // 判断 x 是否小于 15*j
                        break;  // 如果成立，跳出循环
                }
                var pindex = 15 * j - x;  // 计算 pindex 的值
                if (pindex > 0 && pindex <= TotalPinCount)  // 判断 pindex 是否在有效范围内
                    PinSet[--pindex] = State.Down;  // 设置 PinSet 数组中指定索引位置的元素为 State.Down
            }
        }
        return GetPinsDown();  # 返回被击倒的销钉数量
    }
    public void Reset()  # 重置销钉状态为全部竖立
    {
        for (int i = 0; i < PinSet.Length; ++i)  # 遍历销钉数组
        {
            PinSet[i] = State.Up;  # 将每个销钉状态设置为竖立
        }
    }
    public int GetPinsDown()  # 获取被击倒的销钉数量
    {
        return PinSet.Count(p => p == State.Down);  # 返回销钉数组中状态为被击倒的数量
    }
}
```