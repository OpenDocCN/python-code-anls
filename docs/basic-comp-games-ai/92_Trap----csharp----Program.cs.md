# `basic-computer-games\92_Trap\csharp\Program.cs`

```

// 导入 System 命名空间
using System;

// 定义 Program 类
namespace trap_cs
{
  class Program
  {
    // 定义常量 maxGuesses，表示最大猜测次数
    const int maxGuesses = 6;
    // 定义常量 maxNumber，表示最大数字
    const int maxNumber = 100;
    
    // 主函数
    static void Main(string[] args)
    {
      // 初始化 lowGuess 和 highGuess
      int lowGuess  = 0;
      int highGuess = 0;

      // 创建随机数生成器对象
      Random randomNumberGenerator = new ();

      // 输出字符串 "TRAP"
      Print("TRAP");
      // 输出字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
      Print("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
      // 输出空行
      Print();
      // 输出空行
      Print();
      // 输出空行
      Print();

      // 输出游戏说明
      PrintInstructions();

      // 生成要猜测的随机数
      int numberToGuess = randomNumberGenerator.Next(1, maxNumber);

      // 循环猜测
      for (int nGuess = 1; nGuess <= maxGuesses + 1; nGuess++)
      {
        // 如果猜测次数超过最大次数
        if (nGuess > maxGuesses)
        {
          // 输出提示信息
          Print(string.Format("SORRY, THAT'S {0} GUESSES. THE NUMBER WAS {1}", maxGuesses, numberToGuess));
          // 输出空行
          Print();
          // 退出循环
          break;
        }

        // 获取玩家猜测的数字范围
        GetGuesses(nGuess, ref lowGuess, ref highGuess);

        // 判断玩家猜测结果
        if(lowGuess == highGuess && lowGuess == numberToGuess)
        {
          // 输出提示信息
          Print("YOU GOT IT!!!");
          // 输出空行
          Print();
          // 输出提示信息
          Print("TRY AGAIN.");
          // 输出空行
          Print();
          // 退出循环
          break;
        }
        if (highGuess < numberToGuess)
        {
          // 输出提示信息
          Print("MY NUMBER IS LARGER THAN YOUR TRAP NUMBERS.");
        }
        else if (lowGuess > numberToGuess)
        {
          // 输出提示信息
          Print("MY NUMBER IS SMALLER THAN YOUR TRAP NUMBERS.");
        }
        else
        {
          // 输出提示信息
          Print("YOU HAVE TRAPPED MY NUMBER.");
        }
      }
    }

    // 输出游戏说明
    static void PrintInstructions()
    {
      // 输出提示信息
      Print("INSTRUCTIONS ?");

      // 获取用户输入的响应
      char response = Console.ReadKey().KeyChar;
      // 如果响应为 'Y'
      if (response == 'Y')
      {
        // 输出提示信息
        Print(string.Format("I AM THINKING OF A NUMBER BETWEEN 1 AND {0}", maxNumber));
        // 输出提示信息
        Print("TRY TO GUESS MY NUMBER. ON EACH GUESS,");
        // 输出提示信息
        Print("YOU ARE TO ENTER 2 NUMBERS, TRYING TO TRAP");
        // 输出提示信息
        Print("MY NUMBER BETWEEN THE TWO NUMBERS. I WILL");
        // 输出提示信息
        Print("TELL YOU IF YOU HAVE TRAPPED MY NUMBER, IF MY");
        // 输出提示信息
        Print("NUMBER IS LARGER THAN YOUR TWO NUMBERS, OR IF");
        // 输出提示信息
        Print("MY NUMBER IS SMALLER THAN YOUR TWO NUMBERS.");
        // 输出提示信息
        Print("IF YOU WANT TO GUESS ONE SINGLE NUMBER, TYPE");
        // 输出提示信息
        Print("YOUR GUESS FOR BOTH YOUR TRAP NUMBERS.");
        // 输出提示信息
        Print(string.Format("YOU GET {0} GUESSES TO GET MY NUMBER.", maxGuesses));
      }
    }
    
    // 输出字符串
    static void Print(string stringToPrint)
    {
      Console.WriteLine(stringToPrint);
    }
    
    // 输出空行
    static void Print()
    {
      Console.WriteLine();
    }
    
    // 获取玩家猜测的数字范围
    static void GetGuesses(int nGuess, ref int lowGuess, ref int highGuess)
    {
      // 输出空行
      Print();
      // 输出提示信息
      Print(string.Format("GUESS #{0}", nGuess));

      // 获取玩家输入的低猜测和高猜测
      lowGuess  = GetIntFromConsole("Type low guess");
      highGuess = GetIntFromConsole("Type high guess");

      // 如果低猜测大于高猜测，交换它们的值
      if(lowGuess > highGuess)
      {
        int tempGuess = lowGuess;

        lowGuess = highGuess;
        highGuess = tempGuess;
      }
    }
    
    // 从控制台获取整数
    static int GetIntFromConsole(string prompt)
    {
      // 输出提示信息
      Console.Write( prompt + " > ");
      // 从控制台读取输入的字符串
      string intAsString = Console.ReadLine();

      // 如果无法将输入的字符串转换为整数，设置默认值为 1
      if(int.TryParse(intAsString, out int intValue) ==false)
      {
        intValue = 1;
      }

      return intValue;
    }
  }
}

```