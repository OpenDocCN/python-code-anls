# `basic-computer-games\11_Bombardment\csharp\Program.cs`

```py
# 引入 System 命名空间
using System;

# 定义 Bombardment 命名空间
namespace Bombardment
{
    # 定义 Program 类
    class Program
    {
        # 定义程序入口 Main 方法，参数为字符串数组
        static void Main(string[] args)
        {
            # 创建 Bombardment 对象
            var bombardment = new Bombardment();
            # 调用 Bombardment 对象的 Play 方法
            bombardment.Play();
        }
    }
}
```