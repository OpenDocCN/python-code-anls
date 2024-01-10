# `basic-computer-games\93_23_Matches\java\TwentyThreeMatches.java`

```
# 导入 java.util 包中的 Random 和 Scanner 类
import java.util.Random;
import java.util.Scanner;

# 创建名为 TwentyThreeMatches 的类
public class TwentyThreeMatches {

    # 创建名为 MATCH_COUNT_START 的常量，初始值为 23，表示初始火柴数量
    private static final int MATCH_COUNT_START = 23;
    # 创建名为 RAND 的常量，表示随机数生成器
    private static final Random RAND = new Random();
    # 创建名为 scan 的实例变量，表示输入扫描器
    private final Scanner scan = new Scanner(System.in);
    // 开始游戏
    public void startGame() {
        // 初始化数值
        int cpuRemoves = 0;
        int matchesLeft = MATCH_COUNT_START;
        int playerRemoves = 0;

        // 抛硬币决定谁先开始
        CoinSide coinSide = flipCoin();
        if (coinSide == CoinSide.HEADS) {
            System.out.println(Messages.HEADS);
            matchesLeft -= 2;
        } else {
            System.out.println(Messages.TAILS);
        }

        // 游戏循环
        while (true) {
            // 如果 CPU 先开始或者玩家已经移除了火柴棍，显示剩余的火柴棍数量
            if (coinSide == CoinSide.HEADS) {
                System.out.format(Messages.MATCHES_LEFT, matchesLeft);
            }
            coinSide = CoinSide.HEADS;

            // 玩家移除火柴棍
            System.out.println(Messages.REMOVE_MATCHES_QUESTION);
            playerRemoves = turnOfPlayer();
            matchesLeft -= playerRemoves;
            System.out.format(Messages.REMAINING_MATCHES, matchesLeft);

            // 如果剩下1根火柴棍，CPU必须拿走它。你赢了！
            if (matchesLeft <= 1) {
                System.out.println(Messages.WIN);
                return;
            }

            // CPU移除火柴棍
            // 至少剩下两根火柴棍，因为上面的胜利条件没有触发。
            if (matchesLeft <= 4) {
                cpuRemoves = matchesLeft - 1;
            } else {
                cpuRemoves = 4 - playerRemoves;
            }
            System.out.format(Messages.CPU_TURN, cpuRemoves);
            matchesLeft -= cpuRemoves;

            // 如果剩下1根火柴棍，玩家必须拿走它。你输了！
            if (matchesLeft <= 1) {
                System.out.println(Messages.LOSE);
                return;
            }
        }
    }

    // 抛硬币决定谁先开始
    private CoinSide flipCoin() {
        return RAND.nextBoolean() ? CoinSide.HEADS : CoinSide.TAILS;
    }
    # 定义一个私有方法，用于获取玩家的输入并返回玩家移除的数量
    private int turnOfPlayer() {
        # 循环，直到获取有效的玩家输入
        while (true) {
            # 从控制台获取玩家输入的移除数量
            int playerRemoves = scan.nextInt();
            # 处理无效的输入，如果玩家移除数量大于3或小于等于0，则输出错误消息并继续循环
            if ((playerRemoves > 3) || (playerRemoves <= 0)) {
                System.out.println(Messages.INVALID);
                continue;
            }
            # 返回玩家输入的有效移除数量
            return playerRemoves;
        }
    }
# 闭合前面的函数定义
```