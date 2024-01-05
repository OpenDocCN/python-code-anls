# `d:/src/tocomm/basic-computer-games\29_Craps\java\src\Craps.java`

```
import java.util.Random;  // 导入 Random 类，用于生成随机数
import java.util.Scanner;  // 导入 Scanner 类，用于接收用户输入

/**
 *  Port of Craps from BASIC to Java 17.
 */
public class Craps {
  public static final Random random = new Random();  // 创建 Random 对象，用于生成随机数

  public static void main(String[] args) {
    System.out.println("""
                                                            CRAPS
                                          CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY


                           2,3,12 ARE LOSERS; 4,5,6,8,9,10 ARE POINTS; 7,11 ARE NATURAL WINNERS.
                           """);  // 打印游戏规则
    double winnings = 0.0;  // 初始化赢钱数为 0
    do {
      winnings = playCraps(winnings);  // 调用 playCraps 方法进行游戏，并更新赢钱数
  } while (stillInterested(winnings));  // 当仍然对赢得的奖金感兴趣时继续循环
  winningsReport(winnings);  // 报告赢得的奖金
}

public static double playCraps(double winnings) {
  double wager = getWager();  // 获取赌注
  System.out.println("I WILL NOW THROW THE DICE");  // 打印信息
  int roll = rollDice();  // 掷骰子
  double payout = switch (roll) {  // 根据骰子点数进行不同的操作
    case 7, 11 -> naturalWin(roll, wager);  // 点数为7或11时赢得赌注
    case 2, 3, 12 -> lose(roll, wager);  // 点数为2、3或12时输掉赌注
    default -> setPoint(roll, wager);  // 其他点数时设定点数
  };
  return winnings + payout;  // 返回赢得的奖金
}

public static int rollDice() {
  return random.nextInt(1, 7) + random.nextInt(1, 7);  // 掷骰子并返回点数之和
}
  private static double setPoint(int point, double wager) {
    // 打印出点数，并返回通过makePoint函数计算的结果
    System.out.printf("%1$ d IS THE POINT. I WILL ROLL AGAIN%n",point);
    return makePoint(point, wager);
  }

  private static double makePoint(int point, double wager) {
    // 掷骰子，如果结果为7，则返回lose函数计算的结果
    int roll = rollDice();
    if (roll == 7)
      return lose(roll, wager);
    // 如果结果等于点数，则返回win函数计算的结果
    if (roll == point)
      return win(roll, wager);
    // 打印出结果，并通过递归调用makePoint函数继续进行下一轮
    System.out.printf("%1$ d - NO POINT. I WILL ROLL AGAIN%n", roll);
    return makePoint(point, wager);  // recursive
  }

  private static double win(int roll, double wager) {
    // 计算赔率并打印出赢得的金额
    double payout = 2 * wager;
    System.out.printf("%1$ d - A WINNER.........CONGRATS!!!!!!!!%n", roll);
    System.out.printf("%1$ d AT 2 TO 1 ODDS PAYS YOU...LET ME SEE...$%2$3.2f%n",
                      roll, payout);
    return payout;  # 返回赔付金额

  }

  private static double lose(int roll, double wager) {  # 定义输的情况
    String msg = roll == 2 ? "SNAKE EYES.":"CRAPS";  # 根据骰子点数判断消息
    System.out.printf("%1$ d - %2$s...YOU LOSE.%n", roll, msg);  # 打印输的消息
    System.out.printf("YOU LOSE $%3.2f%n", wager);  # 打印输的赌注金额
    return -wager;  # 返回负赌注金额
  }

  public static double naturalWin(int roll, double wager) {  # 定义自然赢的情况
    System.out.printf("%1$ d - NATURAL....A WINNER!!!!%n", roll);  # 打印自然赢的消息
    System.out.printf("%1$ d PAYS EVEN MONEY, YOU WIN $%2$3.2f%n", roll, wager);  # 打印赢的赌注金额
    return wager;  # 返回赌注金额
  }

  public static void winningsUpdate(double winnings) {  # 更新赢得的金额
    System.out.println(switch ((int) Math.signum(winnings)) {  # 使用switch语句根据赢得的金额情况打印消息
      case 1 -> "YOU ARE NOW AHEAD $%3.2f".formatted(winnings);  # 如果赢得的金额大于0，打印赢得的金额
      case 0 -> "YOU ARE NOW EVEN AT 0";  # 如果赢得的金额等于0，打印平局消息
      default -> "YOU ARE NOW UNDER $%3.2f".formatted(-winnings);
    });
  }
```
这段代码是一个switch语句，根据winnings的值进行不同的处理。如果winnings为1，则输出"CONGRATULATIONS---YOU CAME OUT A WINNER. COME AGAIN!"；如果winnings为0，则输出"CONGRATULATIONS---YOU CAME OUT EVEN, NOT BAD FOR AN AMATEUR"；否则输出"TOO BAD, YOU ARE IN THE HOLE. COME AGAIN."。

```
  public static void winningsReport(double winnings) {
    System.out.println(
        switch ((int) Math.signum(winnings)) {
          case 1 -> "CONGRATULATIONS---YOU CAME OUT A WINNER. COME AGAIN!";
          case 0 -> "CONGRATULATIONS---YOU CAME OUT EVEN, NOT BAD FOR AN AMATEUR";
          default -> "TOO BAD, YOU ARE IN THE HOLE. COME AGAIN.";
        }
    );
  }
```
这段代码定义了一个名为winningsReport的方法，接受一个double类型的参数winnings。根据winnings的值输出不同的中奖报告。

```
  public static boolean stillInterested(double winnings) {
    System.out.print(" IF YOU WANT TO PLAY AGAIN PRINT 5 IF NOT PRINT 2 ");
    int fiveOrTwo = (int)getInput();
    winningsUpdate(winnings);
    return fiveOrTwo == 5;
  }
```
这段代码定义了一个名为stillInterested的方法，接受一个double类型的参数winnings。首先打印提示信息，然后获取用户输入的值，调用winningsUpdate方法更新winnings的值，最后返回用户输入的值是否等于5。
  public static double getWager() {
    System.out.print("INPUT THE AMOUNT OF YOUR WAGER. "); // 打印提示信息，要求输入赌注金额
    return getInput(); // 调用 getInput() 方法获取输入的赌注金额
  }

  public static double getInput() {
    Scanner scanner = new Scanner(System.in); // 创建一个 Scanner 对象，用于接收用户输入
    System.out.print("> "); // 打印提示符
    while (true) { // 进入循环，直到获取有效的输入
      try {
        return scanner.nextDouble(); // 尝试获取用户输入的 double 类型数据
      } catch (Exception ex) { // 捕获异常，处理非数字输入
        try {
          scanner.nextLine(); // 清空非数字输入的内容
        } catch (Exception ns_ex) { // 捕获异常，处理输入结束情况
          System.out.println("END OF INPUT, STOPPING PROGRAM."); // 打印输入结束信息
          System.exit(1); // 退出程序
        }
      }
    }
  }
      System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE");  // 打印错误消息，指示输入的不是数字，需要重新输入
      System.out.print("> ");  // 打印提示符，等待用户重新输入
    }
  }
}
```