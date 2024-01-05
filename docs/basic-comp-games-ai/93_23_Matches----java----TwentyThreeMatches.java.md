# `93_23_Matches\java\TwentyThreeMatches.java`

```
import java.util.Random;  // 导入 Random 类，用于生成随机数
import java.util.Scanner;  // 导入 Scanner 类，用于接收用户输入

public class TwentyThreeMatches {

    private static final int MATCH_COUNT_START = 23;  // 定义初始火柴数量
    private static final Random RAND = new Random();  // 创建 Random 对象，用于生成随机数
    private final Scanner scan = new Scanner(System.in);  // 创建 Scanner 对象，用于接收用户输入

    public void startGame() {
        //Initialize values  初始化变量
        int cpuRemoves = 0;  // 初始化 CPU 移除的火柴数量
        int matchesLeft = MATCH_COUNT_START;  // 初始化剩余火柴数量
        int playerRemoves = 0;  // 初始化玩家移除的火柴数量

        //Flip coin and decide who goes first.  抛硬币决定谁先开始
        CoinSide coinSide = flipCoin();  // 调用 flipCoin 方法，得到硬币的一面
        if (coinSide == CoinSide.HEADS) {  // 如果硬币是正面
            System.out.println(Messages.HEADS);  // 输出提示信息
            matchesLeft -= 2;  // 剩余火柴数量减去2
        } else {
            System.out.println(Messages.TAILS); // 如果硬币是反面，打印"Tails"消息
        }

        // 游戏循环
        while (true) {
            // 如果CPU先走或者玩家已经移除了火柴，显示剩余的火柴数量
            if (coinSide == CoinSide.HEADS) {
                System.out.format(Messages.MATCHES_LEFT, matchesLeft); // 格式化打印剩余的火柴数量
            }
            coinSide = CoinSide.HEADS; // 将硬币设为正面

            // 玩家移除火柴
            System.out.println(Messages.REMOVE_MATCHES_QUESTION); // 打印移除火柴的提示
            playerRemoves = turnOfPlayer(); // 玩家移除火柴
            matchesLeft -= playerRemoves; // 更新剩余火柴数量
            System.out.format(Messages.REMAINING_MATCHES, matchesLeft); // 格式化打印剩余的火柴数量

            // 如果剩余火柴数量小于等于1，CPU必须取走最后一根。你赢了！
            if (matchesLeft <= 1) {
            // 如果剩余火柴棍数量小于等于4，CPU 移除的火柴棍数量为剩余数量减1
            if (matchesLeft <= 4) {
                cpuRemoves = matchesLeft - 1;
            } else {
                // 如果剩余火柴棍数量大于4，CPU 移除的火柴棍数量为4减去玩家移除的数量
                cpuRemoves = 4 - playerRemoves;
            }
            // 打印 CPU 移除的火柴棍数量
            System.out.format(Messages.CPU_TURN, cpuRemoves);
            // 更新剩余火柴棍数量
            matchesLeft -= cpuRemoves;

            // 如果剩余火柴棍数量小于等于1，玩家必须取走最后一根，玩家输了
            if (matchesLeft <= 1) {
                // 打印玩家输了的消息
                System.out.println(Messages.LOSE);
                // 结束游戏
                return;
            }
    }  # 结束 turnOfPlayer 方法的定义

    private CoinSide flipCoin() {  # 定义一个私有方法 flipCoin，返回一个 CoinSide 枚举类型的值
        return RAND.nextBoolean() ? CoinSide.HEADS : CoinSide.TAILS;  # 使用随机数生成器来随机返回硬币的正面或反面
    }

    private int turnOfPlayer() {  # 定义一个私有方法 turnOfPlayer，返回一个整数值
        while (true) {  # 进入一个无限循环
            int playerRemoves = scan.nextInt();  # 从输入中获取玩家移除的数量
            // Handle invalid entries  # 处理无效的输入
            if ((playerRemoves > 3) || (playerRemoves <= 0)) {  # 如果玩家移除的数量大于3或小于等于0
                System.out.println(Messages.INVALID);  # 输出提示信息
                continue;  # 继续下一次循环
            }
            return playerRemoves;  # 返回玩家移除的数量
        }
    }

}  # 结束类定义
```