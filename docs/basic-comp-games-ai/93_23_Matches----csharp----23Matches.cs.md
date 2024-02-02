# `basic-computer-games\93_23_Matches\csharp\23Matches.cs`

```py
// 引入 System 命名空间
using System;

// 定义 Program 类
namespace Program
{
  // 定义 Program 类
  class Program
  {

    // 初始化 3 个公共变量，以便在代码的任何地方访问它们
    public static int numberOfMatches;
    public static int numberOfMatchesRemovedByPlayer;
    public static bool playerGoesFirst = false; // 一个标志，表示玩家赢得了抛硬币
    static void Main(string[] args)
    }

    // 定义玩家回合的方法
    static void PlayerTurn()
    {
      // 输出提示信息
      Console.WriteLine("\n");
      Console.WriteLine("YOUR TURN -- YOU MAY TAKE 1, 2, OR 3 MATCHES.");
      Console.Write("HOW MANY DO YOU WISH TO REMOVE ?? ");
      // 获取玩家输入
      numberOfMatchesRemovedByPlayer = ReadPlayerInput();
      // 如果输入无效（不是 1、2 或 3），则要求玩家重新输入
      while (numberOfMatchesRemovedByPlayer > 3 || numberOfMatchesRemovedByPlayer <= 0)
      {
        Console.WriteLine("VERY FUNNY! DUMMY!");
        Console.WriteLine("DO YOU WANT TO PLAY OR GOOF AROUND?");
        Console.Write("NOW, HOW MANY MATCHES DO YOU WANT                 ?? ");
        numberOfMatchesRemovedByPlayer = ReadPlayerInput();
      }

      // 移除玩家指定数量的火柴
      numberOfMatches = numberOfMatches - numberOfMatchesRemovedByPlayer;

      Console.WriteLine("THE ARE NOW " + numberOfMatches + " MATCHES REMAINING");      
    }
    // 定义计算机回合的方法
    static void ComputerTurn()
    {
      // 初始化计算机移除的火柴数量
      int numberOfMatchesRemovedByComputer = 0;
      // 根据火柴数量进行不同的处理
      switch (numberOfMatches)
      {
        case 4:
          numberOfMatchesRemovedByComputer = 3;
          break;
        case 3:
          numberOfMatchesRemovedByComputer = 2;
          break;
        case 2:
          numberOfMatchesRemovedByComputer = 1;
          break;
        case 1: case 0: // 如果计算机输了，则执行此 case
          Console.WriteLine("YOU WON, FLOPPY EARS !");
          Console.WriteLine("THING YOU'RE PRETTY SMART !");
          Console.WriteLine("LETS PLAY AGAIN AND I'LL BLOW YOUR SHOES OFF !!");
          break;
        default: // 如果火柴数量大于 4，则执行此 case
          numberOfMatchesRemovedByComputer = 4 - numberOfMatchesRemovedByPlayer;
          break;
      }
      // 如果计算机移除的火柴数量已经更新，则执行以下代码，否则计算机输了
      if (numberOfMatchesRemovedByComputer != 0)
      {
        Console.WriteLine("MY TURN ! I REMOVE " + numberOfMatchesRemovedByComputer + " MATCHES");
        numberOfMatches = numberOfMatches - numberOfMatchesRemovedByComputer;
        // 如果火柴数量少于或等于 1，则玩家输了
        if (numberOfMatches <= 1)
        {
          Console.Write("\n");
          Console.WriteLine("YOU POOR BOOB! YOU TOOK THE LAST MATCH! I GOTCHA!!");
          Console.WriteLine("HA ! HA ! I BEAT YOU !!!");
          Console.Write("\n");
          Console.WriteLine("GOOD BYE LOSER!");
        }
      }
    }
    
    // 此方法处理玩家输入，并处理不正确的输入
    static int ReadPlayerInput()
    {
      // 读取用户输入并转换为整数
      int playerInput = 0;
      // 尝试读取玩家输入
      try
      {
        playerInput = Convert.ToInt32(Console.ReadLine());
      }
      // 如果玩家输入出现错误
      catch (System.Exception)
      {
        Console.WriteLine("?REENTER");
        Console.Write("?? ");
        // 要求玩家重新输入
        playerInput = ReadPlayerInput();
      }
      // 返回玩家输入的整数
      return playerInput;      
    }
  }
# 闭合函数或代码块的右大括号
```