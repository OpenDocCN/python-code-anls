# `basic-computer-games\14_Bowling\csharp\FrameResult.cs`

```py
# 引入所需的命名空间
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

# 命名空间定义
namespace Bowling
{
    # 定义名为 FrameResult 的公共类
    public class FrameResult
    {
        # 定义枚举类型 Points，包括 None, Error, Spare, Strike 四个取值
        public enum Points { None, Error, Spare, Strike };

        # 定义整型属性 PinsBall1，用于存储球1击倒的瓶数
        public int PinsBall1 { get; set; }
        # 定义整型属性 PinsBall2，用于存储球2击倒的瓶数
        public int PinsBall2 { get; set; }
        # 定义枚举类型属性 Score，用于存储该帧的得分情况
        public Points Score { get; set; }

        # 定义重置方法，将球1和球2的瓶数重置为0，得分情况重置为None
        public void Reset()
        {
            PinsBall1 = PinsBall2 = 0;
            Score = Points.None;
        }
    }
}
```