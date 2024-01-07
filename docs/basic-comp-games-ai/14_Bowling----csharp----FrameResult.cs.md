# `basic-computer-games\14_Bowling\csharp\FrameResult.cs`

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
    // 声明 FrameResult 类
    public class FrameResult
    {
        // 声明 Points 枚举类型，包括 None, Error, Spare, Strike 四个值
        public enum Points { None, Error, Spare, Strike };

        // 声明 PinsBall1 和 PinsBall2 属性，用于表示球1和球2的击倒的瓶数
        public int PinsBall1 { get; set; }
        public int PinsBall2 { get; set; }
        
        // 声明 Score 属性，用于表示该帧的得分情况
        public Points Score { get; set; }

        // 声明 Reset 方法，用于重置帧的数据
        public void Reset()
        {
            PinsBall1 = PinsBall2 = 0; // 重置球1和球2的瓶数
            Score = Points.None; // 重置得分情况为 None
        }
    }
}

```