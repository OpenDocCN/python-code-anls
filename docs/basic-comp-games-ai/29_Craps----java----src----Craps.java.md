# `basic-computer-games\29_Craps\java\src\Craps.java`

```

// 导入 Random 和 Scanner 类
import java.util.Random;
import java.util.Scanner;

/**
 *  从 BASIC 移植到 Java 17 的 Craps 游戏。
 */
public class Craps {
  // 创建一个静态的 Random 对象
  public static final Random random = new Random();

  // 主函数
  public static void main(String[] args) {
    // 打印游戏介绍
    System.out.println("""
                                                            CRAPS
                                          CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY


                           2,3,12 ARE LOSERS; 4,5,6,8,9,10 ARE POINTS; 7,11 ARE NATURAL WINNERS.
                           """);
    // 初始化赢得的金额
    double winnings = 0.0;
    // 循环进行游戏直到不再感兴趣
    do {
      winnings = playCraps(winnings);
    } while (stillInterested(winnings));
    // 打印游戏结果
    winningsReport(winnings);
  }

  // 游戏逻辑
  public static double playCraps(double winnings) {
    // 获取赌注
    double wager = getWager();
    // 打印掷骰子的信息
    System.out.println("I WILL NOW THROW THE DICE");
    // 掷骰子
    int roll = rollDice();
    // 根据骰子点数进行不同的操作
    double payout = switch (roll) {
      case 7, 11 -> naturalWin(roll, wager);
      case 2, 3, 12 -> lose(roll, wager);
      default -> setPoint(roll, wager);
    };
    return winnings + payout;
  }

  // 掷骰子
  public static int rollDice() {
    return random.nextInt(1, 7) + random.nextInt(1, 7);
  }

  // 设置点数
  private static double setPoint(int point, double wager) {
    System.out.printf("%1$ d IS THE POINT. I WILL ROLL AGAIN%n",point);
    return makePoint(point, wager);
  }

  // 重新掷骰子
  private static double makePoint(int point, double wager) {
    int roll = rollDice();
    if (roll == 7)
      return lose(roll, wager);
    if (roll == point)
      return win(roll, wager);
    System.out.printf("%1$ d - NO POINT. I WILL ROLL AGAIN%n", roll);
    return makePoint(point, wager);  // 递归调用
  }

  // 赢得游戏
  private static double win(int roll, double wager) {
    double payout = 2 * wager;
    System.out.printf("%1$ d - A WINNER.........CONGRATS!!!!!!!!%n", roll);
    System.out.printf("%1$ d AT 2 TO 1 ODDS PAYS YOU...LET ME SEE...$%2$3.2f%n",
                      roll, payout);
    return payout;
  }

  // 输掉游戏
  private static double lose(int roll, double wager) {
    String msg = roll == 2 ? "SNAKE EYES.":"CRAPS";
    System.out.printf("%1$ d - %2$s...YOU LOSE.%n", roll, msg);
    System.out.printf("YOU LOSE $%3.2f%n", wager);
    return -wager;
  }

  // 自然获胜
  public static double naturalWin(int roll, double wager) {
    System.out.printf("%1$ d - NATURAL....A WINNER!!!!%n", roll);
    System.out.printf("%1$ d PAYS EVEN MONEY, YOU WIN $%2$3.2f%n", roll, wager);
    return wager;
  }

  // 更新赢得的金额
  public static void winningsUpdate(double winnings) {
    System.out.println(switch ((int) Math.signum(winnings)) {
      case 1 -> "YOU ARE NOW AHEAD $%3.2f".formatted(winnings);
      case 0 -> "YOU ARE NOW EVEN AT 0";
      default -> "YOU ARE NOW UNDER $%3.2f".formatted(-winnings);
    });
  }

  // 打印游戏结果
  public static void winningsReport(double winnings) {
    System.out.println(
        switch ((int) Math.signum(winnings)) {
          case 1 -> "CONGRATULATIONS---YOU CAME OUT A WINNER. COME AGAIN!";
          case 0 -> "CONGRATULATIONS---YOU CAME OUT EVEN, NOT BAD FOR AN AMATEUR";
          default -> "TOO BAD, YOU ARE IN THE HOLE. COME AGAIN.";
        }
    );
  }

  // 是否继续游戏
  public static boolean stillInterested(double winnings) {
    System.out.print(" IF YOU WANT TO PLAY AGAIN PRINT 5 IF NOT PRINT 2 ");
    int fiveOrTwo = (int)getInput();
    winningsUpdate(winnings);
    return fiveOrTwo == 5;
  }

  // 获取赌注
  public static double getWager() {
    System.out.print("INPUT THE AMOUNT OF YOUR WAGER. ");
    return getInput();
  }

  // 获取输入
  public static double getInput() {
    Scanner scanner = new Scanner(System.in);
    System.out.print("> ");
    while (true) {
      try {
        return scanner.nextDouble();
      } catch (Exception ex) {
        try {
          scanner.nextLine(); // 清空非数字的输入
        } catch (Exception ns_ex) { // 收到 EOF (ctrl-d 或者在 Windows 下是 ctrl-z)
          System.out.println("END OF INPUT, STOPPING PROGRAM.");
          System.exit(1);
        }
      }
      System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE");
      System.out.print("> ");
    }
  }
}

```