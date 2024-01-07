# `basic-computer-games\51_Hurkle\csharp\Program.cs`

```

// 引入 System 命名空间
using System;

// 定义 hurkle 命名空间
namespace hurkle
{
    // 定义 Program 类
    class Program
    {
        // 程序入口
        static void Main(string[] args)
        {
            /*
            Original source transscription
            10 PRINT TAB(33);"HURKLE"
            20 PRINT TAB(15);"CREATIVE COMPUTING NORRISTOWN, NEW JERSEY"
            30 PRINT;PRINT;PRINT
            */
            // 输出字符串 "HURKLE"，使用空格填充到指定长度
            Console.WriteLine(new string(' ', 33) + @"HURKLE");
            // 输出字符串 "CREATIVE COMPUTING NORRISTOWN, NEW JERSEY"，使用空格填充到指定长度
            Console.WriteLine(new string(' ', 15) + @"CREATIVE COMPUTING NORRISTOWN, NEW JERSEY");
            
            /*
            110 N=5
            120 G=10
            */
            // 定义变量 N 和 G，并赋值
            var N=5;
            var G=10;
            
            /*
            210 PRINT
            220 PRINT "A HURKLE IS HIDING ON A";G;"BY";G;"GRID. HOMEBASE"
            230 PRINT "ON THE GRID IS POINT 0,0 AND ANY GRIDPOINT IS A"
            240 PRINT "PAIR OF WHOLE NUMBERS SEPERATED BY A COMMA. TRY TO"
            250 PRINT "GUESS THE HURKLE'S GRIDPOINT. YOU GET";N;"TRIES."
            260 PRINT "AFTER EACH TRY, I WILL TELL YOU THE APPROXIMATE"
            270 PRINT "DIRECTION TO GO TO LOOK FOR THE HURKLE."
            280 PRINT
            */
            // 输出多行字符串
            Console.WriteLine();
            Console.WriteLine($"A HURKLE IS HIDING ON A {G} BY {G} GRID. HOMEBASE");
            Console.WriteLine(@"ON THE GRID IS POINT 0,0 AND ANY GRIDPOINT IS A");
            Console.WriteLine(@"PAIR OF WHOLE NUMBERS SEPERATED BY A COMMA. TRY TO");
            Console.WriteLine($"GUESS THE HURKLE'S GRIDPOINT. YOU GET {N} TRIES.");
            Console.WriteLine(@"AFTER EACH TRY, I WILL TELL YOU THE APPROXIMATE");
            Console.WriteLine(@"DIRECTION TO GO TO LOOK FOR THE HURKLE.");
            Console.WriteLine();

            // 创建 ConsoleHurkleView 和 HurkleGame 对象
            var view = new ConsoleHurkleView();
            var hurkle = new HurkleGame(N,G, view);
            
            // 游戏循环
            while(true)
            {
                hurkle.PlayGame();

                // 提示用户是否再玩一次
                Console.WriteLine("PLAY AGAIN? (Y)ES/(N)O");
                var playAgainResponse = Console.ReadLine();
                if(playAgainResponse.Trim().StartsWith("y", StringComparison.InvariantCultureIgnoreCase))
                {
                    Console.WriteLine();
                    Console.WriteLine("LET'S PLAY AGAIN. HURKLE IS HIDING");
                    Console.WriteLine();
                }else{
                    Console.WriteLine("THANKS FOR PLAYING!");
                    break;
                }

            }
        }
    }
}

```