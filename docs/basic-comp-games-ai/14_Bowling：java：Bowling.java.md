# `14_Bowling\java\Bowling.java`

```
import java.util.Scanner;  # 导入 Scanner 类，用于用户输入
import java.lang.Math;  # 导入 Math 类，用于数学运算

/**
 * Game of Bowling
 * <p>
 * Based on the BASIC game of Bowling here
 * https://github.com/coding-horror/basic-computer-games/blob/main/14%20Bowling/bowling.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

public class Bowling {

  private final Scanner scan;  // 用于用户输入的 Scanner 对象

  public Bowling() {
    scan = new Scanner(System.in);  // 创建一个新的 Scanner 对象，用于从控制台读取输入

  }  // 构造函数 Bowling 的结束

  public void play() {

    showIntro();  // 调用 showIntro 方法显示游戏介绍
    startGame();  // 调用 startGame 方法开始游戏

  }  // 方法 play 的结束

  private static void showIntro() {

    System.out.println(" ".repeat(33) + "BOWL");  // 在控制台打印游戏标题
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 在控制台打印游戏信息
    System.out.println("\n\n");  // 在控制台打印空行

  }  // 方法 showIntro 的结束
    private void startGame() {  // 定义一个名为startGame的私有方法

    int ball = 0;  // 初始化变量ball为0，用于记录当前球的次数
    int bell = 0;  // 初始化变量bell为0，暂时未知作用
    int frame = 0;  // 初始化变量frame为0，用于记录当前帧数
    int ii = 0;  // 初始化变量ii为0，用作循环迭代器
    int jj = 0;  // 初始化变量jj为0，用作循环迭代器
    int kk = 0;  // 初始化变量kk为0，用作循环迭代器
    int numPlayers = 0;  // 初始化变量numPlayers为0，用于记录玩家数量
    int pinsDownBefore = 0;  // 初始化变量pinsDownBefore为0，用于记录之前击倒的瓶子数量
    int pinsDownNow = 0;  // 初始化变量pinsDownNow为0，用于记录当前击倒的瓶子数量
    int player = 0;  // 初始化变量player为0，用于记录当前玩家
    int randVal = 0;  // 初始化变量randVal为0，用于记录随机值
    int result = 0;  // 初始化变量result为0，用于记录结果

    int[] pins = new int[16];  // 创建一个长度为16的整型数组pins，用于记录瓶子的状态

    int[][] scores = new int[101][7];  // 创建一个大小为101x7的二维整型数组scores，用于记录每个玩家的得分

    String userResponse = "";  // 初始化变量userResponse为空字符串，用于记录用户的响应
    System.out.println("WELCOME TO THE ALLEY"); // 打印欢迎信息
    System.out.println("BRING YOUR FRIENDS"); // 打印邀请朋友参加游戏
    System.out.println("OKAY LET'S FIRST GET ACQUAINTED"); // 打印游戏开始前的问候
    System.out.println(""); // 打印空行
    System.out.println("THE INSTRUCTIONS (Y/N)"); // 打印提示用户输入是否需要游戏说明
    System.out.print("? "); // 打印提示符

    userResponse = scan.nextLine(); // 从用户输入中读取响应

    if (userResponse.toUpperCase().equals("Y")) { // 如果用户响应是Y，则执行下面的代码
      printRules(); // 调用打印游戏规则的函数
    }

    System.out.print("FIRST OF ALL...HOW MANY ARE PLAYING? "); // 打印提示输入玩家数量的信息
    numPlayers = Integer.parseInt(scan.nextLine()); // 从用户输入中读取玩家数量并转换为整数赋值给numPlayers

    System.out.println(""); // 打印空行
    System.out.println("VERY GOOD..."); // 打印提示信息
    // 开始外部 while 循环
    while (true):

      for (ii = 1; ii <= 100; ii++):
        for (jj = 1; jj <= 6; jj++):
          scores[ii][jj] = 0;

      frame = 1;

      // 开始帧循环
      while (frame < 11):

        // 开始循环遍历所有玩家
        for (player = 1; player <= numPlayers; player++):

          pinsDownBefore = 0;
          ball = 1;
          result = 0;
          for (ii = 1; ii <= 15; ii++) {
            pins[ii] = 0;  // 初始化数组 pins，将每个元素的值设为 0
          }

          while (true) {

            // Ball generator using mod '15' system
            // 使用模 '15' 系统生成球

            System.out.println("TYPE ROLL TO GET THE BALL GOING.");  // 打印提示信息
            System.out.print("? ");  // 打印提示信息
            scan.nextLine();  // 读取用户输入

            kk = 0;  // 初始化变量 kk，将其值设为 0
            pinsDownNow = 0;  // 初始化变量 pinsDownNow，将其值设为 0

            for (ii = 1; ii <= 20; ii++) {  // 循环 20 次

              randVal = (int)(Math.random() * 100) + 1;  // 生成一个 1 到 100 之间的随机整数
              for (jj = 1; jj <= 10; jj++) {  # 使用循环遍历1到10之间的数字

                if (randVal < 15 * jj) {  # 如果随机值小于15乘以当前循环变量的值
                  break;  # 跳出循环
                }
              }
              pins[15 * jj - randVal] = 1;  # 将计算后的值赋给数组pins的相应位置

            }

            // Pin diagram  # 打印针脚图

            System.out.println("PLAYER: " + player + " FRAME: " + frame + " BALL: " + ball);  # 打印玩家、帧和球的信息

            for (ii = 0; ii <= 3; ii++) {  # 使用循环遍历0到3之间的数字

              System.out.println("");  # 打印空行

              System.out.print(" ".repeat(ii));  # 打印指定数量的空格

              for (jj = 1; jj <= 4 - ii; jj++) {  # 使用循环遍历1到4减去ii的值之间的数字
                kk++;  // 增加 kk 的值，用于遍历数组

                if (pins[kk] == 1) {  // 如果数组中 pins[kk] 的值为 1
                  System.out.print("O ");  // 打印 "O "
                } else {
                  System.out.print("+ ");  // 否则打印 "+ "
                }
              }
            }

            System.out.println("");  // 打印空行

            // Roll analysis  // 掷球分析

            for (ii = 1; ii <= 10; ii++) {  // 遍历 1 到 10
              pinsDownNow += pins[ii];  // 将 pins[ii] 的值加到 pinsDownNow 中
            }

            if (pinsDownNow - pinsDownBefore == 0) {
              // 如果当前击倒的瓶数减去之前的瓶数等于0，打印“GUTTER!!”
              System.out.println("GUTTER!!");
            }

            if (ball == 1 && pinsDownNow == 10) {
              // 如果是第一次投球并且当前击倒的瓶数为10，打印“STRIKE!!!!!”
              System.out.println("STRIKE!!!!!");

              // 播放铃声
              for (bell = 1; bell <= 4; bell++) {
                System.out.print("\007");
                try {
                  Thread.sleep(500);
                } catch (InterruptedException e) {
                  Thread.currentThread().interrupt();
                }
              }
              result = 3;
            }
            // 如果是第二次投球且击倒了全部的 10 个球
            if (ball == 2 && pinsDownNow == 10) {
              System.out.println("SPARE!!!!");  // 打印出“SPARE!!!!”
              result = 2;  // 将结果设置为 2
            }

            // 如果是第二次投球且击倒的球数小于 10
            if (ball == 2 && pinsDownNow < 10) {
              System.out.println("ERROR!!!");  // 打印出“ERROR!!!”
              result = 1;  // 将结果设置为 1
            }

            // 如果是第一次投球且击倒的球数小于 10
            if (ball == 1 && pinsDownNow < 10) {
              System.out.println("ROLL YOUR 2ND BALL");  // 打印出“ROLL YOUR 2ND BALL”
            }

            // 存储得分

            System.out.println("");  // 打印一个空行

            scores[frame * player][ball] = pinsDownNow;  // 将击倒的球数存储到得分数组中
            if (ball != 2) {  # 如果球不是第二次投掷
              ball = 2;  # 将球的次数设置为2
              pinsDownBefore = pinsDownNow;  # 记录当前球之前击倒的瓶数

              if (result != 3) {  # 如果结果不是补中
                scores[frame * player][ball] = pinsDownNow - pinsDownBefore;  # 计算当前球的得分
                if (result == 0) {  # 如果是失误
                  continue;  # 继续下一轮投掷
                }
              } else {  # 如果是补中
                scores[frame * player][ball] = pinsDownNow;  # 记录当前球的得分
              }

            }
            break;  # 结束当前投掷
          }

          scores[frame * player][3] = result;  # 记录当前帧的总得分
        }  // End loop through all players  // 结束对所有玩家的循环

        frame++;  // 帧数加一

      }  // End frame while loop  // 结束帧循环

      System.out.println("FRAMES");  // 打印"FRAMES"

      System.out.print(" ");  // 打印空格
      for (ii = 1; ii <= 10; ii++) {  // 循环10次
        System.out.print(ii + " ");  // 打印ii和空格
      }

      System.out.println("");  // 打印换行

      for (player = 1; player <= numPlayers; player++) {  // 循环每个玩家
        for (ii = 1; ii <= 3; ii++) {  // 循环3次
          System.out.print(" ");  // 打印空格
          for (jj = 1; jj <= 10; jj++) {  // 循环10次
            System.out.print (scores[jj * player][ii] + " ");  // 打印分数和空格
          }
          System.out.println("");
        }
        System.out.println("");
      }

      System.out.println("DO YOU WANT ANOTHER GAME");
      System.out.print("? ");

      userResponse = scan.nextLine();

      if (!String.valueOf(userResponse.toUpperCase().charAt(0)).equals("Y")) {
        break;
      }

    }  // End outer while loop

  }  // End of method startGame

  public static void printRules() {
```

这部分代码是Java语言的程序，不是Python。因此，无法为这部分代码添加注释。
    # 打印保龄球游戏的规则说明
    print("THE GAME OF BOWLING TAKES MIND AND SKILL.DURING THE GAME")
    print("THE COMPUTER WILL KEEP SCORE.YOU MAY COMPETE WITH")
    print("OTHER PLAYERS[UP TO FOUR].YOU WILL BE PLAYING TEN FRAMES")
    print("ON THE PIN DIAGRAM 'O' MEANS THE PIN IS DOWN...'+' MEANS THE")
    print("PIN IS STANDING.AFTER THE GAME THE COMPUTER WILL SHOW YOUR")
    print("SCORES .")

  }  # 方法printRules结束

  def main():
    # 创建Bowling对象
    game = Bowling()
    # 开始游戏
    game.play()

  }  # 方法main结束

}  # Bowling类结束
```