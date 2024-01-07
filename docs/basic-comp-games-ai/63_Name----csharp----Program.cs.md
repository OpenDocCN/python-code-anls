# `basic-computer-games\63_Name\csharp\Program.cs`

```

// 引入 System 命名空间
using System;

// 定义 Program 类
namespace Name
{
    public class Program
    {
        // 程序入口
        static void Main(string[] args)
        {
            // 居中打印字符串 "NAME"
            Console.WriteLine("NAME".CentreAlign());
            // 居中打印字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY".CentreAlign());
            // 打印空行
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            // 打印 "HELLO."
            Console.WriteLine("HELLO.");
            // 打印 "MY NAME IS CREATIVE COMPUTER."
            Console.WriteLine("MY NAME IS CREATIVE COMPUTER.");
            // 提示用户输入名字
            Console.Write("WHAT'S YOUR NAME (FIRST AND LAST? ");
            var name = Console.ReadLine();
            // 打印空行
            Console.WriteLine();
            // 打印倒序后的名字
            Console.WriteLine($"THANK YOU, {name.Reverse()}.");
            // 打印提示信息
            Console.WriteLine("OOPS!  I GUESS I GOT IT BACKWARDS.  A SMART");
            Console.WriteLine("COMPUTER LIKE ME SHOULDN'T MAKE A MISTAKE LIKE THAT!");
            // 打印空行
            Console.WriteLine();
            // 打印提示信息
            Console.WriteLine("BUT I JUST NOTICED YOUR LETTERS ARE OUT OF ORDER.");
            // 打印按字母顺序排序后的名字
            Console.WriteLine($"LET'S PUT THEM IN ORDER LIKE THIS: {name.Sort()}");
            // 打印空行
            Console.WriteLine();
            // 提示用户是否喜欢按字母顺序排序后的名字
            Console.Write("DON'T YOU LIKE THAT BETTER? ");
            var like = Console.ReadLine();
            // 打印空行
            Console.WriteLine();

            // 判断用户输入是否为 "YES"，并打印相应信息
            if (like.ToUpperInvariant() == "YES")
            {
                Console.WriteLine("I KNEW YOU'D AGREE!!");
            }
            else
            {
                Console.WriteLine("I'M SORRY YOU DON'T LIKE IT THAT WAY.");
            }

            // 打印空行
            Console.WriteLine();
            // 打印问候语
            Console.WriteLine($"I REALLY ENJOYED MEETING YOU {name}.");
            Console.WriteLine("HAVE A NICE DAY!");
        }
    }
}

```