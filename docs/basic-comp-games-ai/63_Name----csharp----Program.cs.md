# `63_Name\csharp\Program.cs`

```
using System;

namespace Name
{
    public class Program
    {
        static void Main(string[] args)
        {
            // 输出"NAME"并居中对齐
            Console.WriteLine("NAME".CentreAlign());
            // 输出"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"并居中对齐
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY".CentreAlign());
            // 输出空行
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            // 输出"HELLO."
            Console.WriteLine("HELLO.");
            // 输出"MY NAME IS CREATIVE COMPUTER."
            Console.WriteLine("MY NAME IS CREATIVE COMPUTER.");
            // 输出"WHAT'S YOUR NAME (FIRST AND LAST? "并等待用户输入
            var name = Console.ReadLine();
            // 输出空行
            Console.WriteLine();
            // 输出"THANK YOU, {name.Reverse()}."，其中{name.Reverse()}表示将用户输入的名字反转后输出
            Console.WriteLine($"THANK YOU, {name.Reverse()}.");
            // 输出"OOPS!  I GUESS I GOT IT BACKWARDS.  A SMART"
```
```csharp
        }
    }
}
            # 打印错误信息
            Console.WriteLine("COMPUTER LIKE ME SHOULDN'T MAKE A MISTAKE LIKE THAT!")
            # 打印空行
            Console.WriteLine()
            # 打印提示信息
            Console.WriteLine("BUT I JUST NOTICED YOUR LETTERS ARE OUT OF ORDER.")
            # 对名字进行排序并打印排序后的结果
            Console.WriteLine($"LET'S PUT THEM IN ORDER LIKE THIS: {name.Sort()}")
            # 打印空行
            Console.WriteLine()
            # 提示用户是否喜欢排序后的名字
            Console.Write("DON'T YOU LIKE THAT BETTER? ")
            # 读取用户输入
            var like = Console.ReadLine()
            # 打印空行
            Console.WriteLine()

            # 判断用户输入是否为"YES"，如果是则打印肯定的回答
            if (like.ToUpperInvariant() == "YES"):
                Console.WriteLine("I KNEW YOU'D AGREE!!")
            # 如果不是则打印否定的回答
            else:
                Console.WriteLine("I'M SORRY YOU DON'T LIKE IT THAT WAY.")

            # 打印空行
            Console.WriteLine()
            # 打印感谢信息
            Console.WriteLine($"I REALLY ENJOYED MEETING YOU {name}.")
# 输出字符串 "HAVE A NICE DAY!" 到控制台
Console.WriteLine("HAVE A NICE DAY!");
# 结束 Main 方法
}
# 结束 Program 类
}
```