# `basic-computer-games\19_Bunny\csharp\BasicData.cs`

```
// 引入命名空间
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// 命名空间定义
namespace Bunny
{
    // 内部类 BasicData 定义
    internal class BasicData
    {
        // 只读整型数组 data
        private readonly int[] data;

        // 整型变量 index
        private int index;

        // BasicData 类的构造函数，接受一个整型数组参数
        public BasicData(int[] data)
        {
            // 将传入的整型数组赋值给类的只读整型数组 data
            this.data = data;
            // 将 index 初始化为 0
            index = 0;
        }
        
        // Read 方法，用于读取数组中的元素
        public int Read()
        {
            // 返回数组中索引为 index 的元素，并将 index 自增
            return data[index++];
        }
    }
}
```