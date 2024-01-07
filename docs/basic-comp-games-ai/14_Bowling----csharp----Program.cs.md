# `basic-computer-games\14_Bowling\csharp\Program.cs`

```

// 引入命名空间
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// 声明命名空间 Bowling
namespace Bowling
{
    // 声明公共静态类 Program
    public static class Program
    {
        // 声明公共静态方法 Main
        public static void Main()
        {
            // 创建 Bowling 对象并调用 Play 方法
            new Bowling().Play();
        }
    }
}

```