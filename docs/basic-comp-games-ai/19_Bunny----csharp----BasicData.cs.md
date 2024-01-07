# `basic-computer-games\19_Bunny\csharp\BasicData.cs`

```

// 引入命名空间
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// 命名空间 Bunny
namespace Bunny
{
    // 内部类 BasicData
    internal class BasicData
    {
        // 只读整型数组 data
        private readonly int[] data;

        // 整型变量 index
        private int index;

        // 构造函数，接受一个整型数组作为参数
        public BasicData(int[] data)
        {
            // 初始化 data 数组
            this.data = data;
            // 初始化 index 为 0
            index = 0;
        }
        // 读取方法，返回当前 index 对应的 data 值，并将 index 自增
        public int Read()
        {
            return data[index++];
        }
    }
}

```