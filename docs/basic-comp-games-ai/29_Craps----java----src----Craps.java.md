# `basic-computer-games\29_Craps\java\src\Craps.java`

```
import java.util.Random;  // 导入 Random 类
import java.util.Scanner;  // 导入 Scanner 类

/**
 *  从 BASIC 移植到 Java 17 的 Craps 游戏。
 */
public class Craps {
  public static final Random random = new Random();  // 创建 Random 对象

  public static void main(String[] args) {
    System.out.println("""
                                                            CRAPS
                                          CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY


                           2,3,12 ARE LOSERS; 4,5,6,8,9,10 ARE POINTS; 7,11 ARE NATURAL WINNERS.
                           """);  // 打印游戏规则
    double winnings = 0.0;  // 初始化赢钱数
    do {
      winnings = playCraps(winnings);  // 进行 Craps 游戏
    } while (stillInterested(winnings));  // 判断是否继续游戏
    winningsReport(winnings);  // 打印游戏结果
  }

  public static double playCraps(double winnings) {
    double wager = getWager();  // 获取赌注
    System.out.println("I WILL NOW THROW THE DICE");  // 打印掷骰子提示
    int roll = rollDice();  // 掷骰子
    double payout = switch (roll) {  // 根据骰子点数进行不同的操作
      case 7, 11 -> naturalWin(roll, wager);  // 自然胜
      case 2, 3, 12 -> lose(roll, wager);  // 输
      default -> setPoint(roll, wager);  // 设置点数
    };
    return winnings + payout;  // 返回赢钱数
  }

  public static int rollDice() {
    return random.nextInt(1, 7) + random.nextInt(1, 7);  // 随机生成两个骰子点数之和
  }

  private static double setPoint(int point, double wager) {
    System.out.printf("%1$ d IS THE POINT. I WILL ROLL AGAIN%n",point);  // 打印点数并提示再次掷骰子
    return makePoint(point, wager);  // 返回设置点数后的结果
  }

  private static double makePoint(int point, double wager) {
    int roll = rollDice();  // 再次掷骰子
    if (roll == 7)
      return lose(roll, wager);  // 如果点数为7，则输
    if (roll == point)
      return win(roll, wager);  // 如果点数等于设定点数，则赢
    System.out.printf("%1$ d - NO POINT. I WILL ROLL AGAIN%n", roll);  // 打印点数并提示再次掷骰子
    return makePoint(point, wager);  // 递归调用，继续进行游戏
  }

  private static double win(int roll, double wager) {
    double payout = 2 * wager;  // 计算赢得的奖金
    System.out.printf("%1$ d - A WINNER.........CONGRATS!!!!!!!!%n", roll);  // 打印赢得的点数
    System.out.printf("%1$ d AT 2 TO 1 ODDS PAYS YOU...LET ME SEE...$%2$3.2f%n",
                      roll, payout);  // 打印赢得的奖金
    return payout;  // 返回赢得的奖金
  }

  private static double lose(int roll, double wager) {
    // 根据掷骰子的结果判断消息内容，如果是2则为"SNAKE EYES."，否则为"CRAPS"
    String msg = roll == 2 ? "SNAKE EYES.":"CRAPS";
    // 格式化输出掷骰子的结果和消息内容，表示输了
    System.out.printf("%1$ d - %2$s...YOU LOSE.%n", roll, msg);
    // 格式化输出输掉的赌注金额
    System.out.printf("YOU LOSE $%3.2f%n", wager);
    // 返回输掉的赌注金额的负值
    return -wager;
  }

  // 表示掷骰子结果为自然胜利
  public static double naturalWin(int roll, double wager) {
    // 格式化输出掷骰子的结果，表示自然胜利
    System.out.printf("%1$ d - NATURAL....A WINNER!!!!%n", roll);
    // 格式化输出掷骰子的结果和赌注金额，表示赢得了赌注
    System.out.printf("%1$ d PAYS EVEN MONEY, YOU WIN $%2$3.2f%n", roll, wager);
    // 返回赌注金额
    return wager;
  }

  // 更新赢得的金额信息
  public static void winningsUpdate(double winnings) {
    // 根据赢得的金额情况进行输出
    System.out.println(switch ((int) Math.signum(winnings)) {
      case 1 -> "YOU ARE NOW AHEAD $%3.2f".formatted(winnings);
      case 0 -> "YOU ARE NOW EVEN AT 0";
      default -> "YOU ARE NOW UNDER $%3.2f".formatted(-winnings);
    });
  }

  // 报告赢得的金额情况
  public static void winningsReport(double winnings) {
    // 根据赢得的金额情况进行输出
    System.out.println(
        switch ((int) Math.signum(winnings)) {
          case 1 -> "CONGRATULATIONS---YOU CAME OUT A WINNER. COME AGAIN!";
          case 0 -> "CONGRATULATIONS---YOU CAME OUT EVEN, NOT BAD FOR AN AMATEUR";
          default -> "TOO BAD, YOU ARE IN THE HOLE. COME AGAIN.";
        }
    );
  }

  // 判断是否还有兴趣继续游戏
  public static boolean stillInterested(double winnings) {
    // 提示用户输入是否继续游戏
    System.out.print(" IF YOU WANT TO PLAY AGAIN PRINT 5 IF NOT PRINT 2 ");
    // 获取用户输入的数字
    int fiveOrTwo = (int)getInput();
    // 更新赢得的金额信息
    winningsUpdate(winnings);
    // 返回用户输入的数字是否为5
    return fiveOrTwo == 5;
  }

  // 获取用户输入的赌注金额
  public static double getWager() {
    // 提示用户输入赌注金额
    System.out.print("INPUT THE AMOUNT OF YOUR WAGER. ");
    // 获取用户输入的赌注金额
    return getInput();
  }

  // 获取用户输入的数值
  public static double getInput() {
    // 创建一个Scanner对象用于获取用户输入
    Scanner scanner = new Scanner(System.in);
    // 提示用户输入
    System.out.print("> ");
    # 进入循环，不断尝试读取输入的 double 类型数据
    while (true) {
      # 尝试读取 double 类型数据，如果成功则返回
      try {
        return scanner.nextDouble();
      } catch (Exception ex) {
        # 如果捕获到异常，则尝试读取并丢弃当前输入行的内容
        try {
          scanner.nextLine(); // flush whatever this non number stuff is.
        } catch (Exception ns_ex) { // 捕获到输入流结束异常，表示输入结束
          System.out.println("END OF INPUT, STOPPING PROGRAM.");
          System.exit(1);
        }
      }
      # 如果无法成功读取 double 类型数据，则输出错误信息并提示重新输入
      System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE");
      System.out.print("> ");
    }
  }
# 闭合前面的函数定义
```