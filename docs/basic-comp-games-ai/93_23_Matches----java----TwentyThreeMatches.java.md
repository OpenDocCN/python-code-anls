# `basic-computer-games\93_23_Matches\java\TwentyThreeMatches.java`

```

// 导入必要的类
import java.util.Random;
import java.util.Scanner;

// 创建名为TwentyThreeMatches的类
public class TwentyThreeMatches {

    // 定义初始火柴数量
    private static final int MATCH_COUNT_START = 23;
    // 创建Random对象
    private static final Random RAND = new Random();
    // 创建Scanner对象
    private final Scanner scan = new Scanner(System.in);

    // 开始游戏的方法
    public void startGame() {
        // 初始化变量
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
            // 如果CPU先开始或者玩家已经移除了火柴，显示剩余火柴数量
            if (coinSide == CoinSide.HEADS) {
                System.out.format(Messages.MATCHES_LEFT, matchesLeft);
            }
            coinSide = CoinSide.HEADS;

            // 玩家移除火柴
            System.out.println(Messages.REMOVE_MATCHES_QUESTION);
            playerRemoves = turnOfPlayer();
            matchesLeft -= playerRemoves;
            System.out.format(Messages.REMAINING_MATCHES, matchesLeft);

            // 如果剩余1根火柴，CPU必须拿走，玩家获胜
            if (matchesLeft <= 1) {
                System.out.println(Messages.WIN);
                return;
            }

            // CPU移除火柴
            // 至少剩余两根火柴，因为上面的获胜条件没有触发
            if (matchesLeft <= 4) {
                cpuRemoves = matchesLeft - 1;
            } else {
                cpuRemoves = 4 - playerRemoves;
            }
            System.out.format(Messages.CPU_TURN, cpuRemoves);
            matchesLeft -= cpuRemoves;

            // 如果剩余1根火柴，玩家必须拿走，玩家失败
            if (matchesLeft <= 1) {
                System.out.println(Messages.LOSE);
                return;
            }
        }
    }

    // 抛硬币决定谁先开始的方法
    private CoinSide flipCoin() {
        return RAND.nextBoolean() ? CoinSide.HEADS : CoinSide.TAILS;
    }

    // 玩家回合的方法
    private int turnOfPlayer() {
        while (true) {
            int playerRemoves = scan.nextInt();
            // 处理无效输入
            if ((playerRemoves > 3) || (playerRemoves <= 0)) {
                System.out.println(Messages.INVALID);
                continue;
            }
            return playerRemoves;
        }
    }

}

```