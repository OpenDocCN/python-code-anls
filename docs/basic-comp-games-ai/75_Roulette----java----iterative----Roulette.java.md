# `basic-computer-games\75_Roulette\java\iterative\Roulette.java`

```
import java.io.InputStream;
import java.io.PrintStream;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

public class Roulette {

    private static Set<Integer> RED_NUMBERS;

    static {
        // 初始化红色号码的集合
        RED_NUMBERS = Set.of(1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36);
    }

    private PrintStream out;
    private Scanner scanner;
    private int houseBalance, playerBalance;
    private Random random;

    public Roulette(PrintStream out, InputStream in) {
        // 初始化输出流、输入流和初始的庄家余额、玩家余额
        this.out = out;
        this.scanner = new Scanner(in);
        houseBalance = 100000;
        playerBalance = 1000;
        random = new Random();
    }

    public static void main(String[] args) {
        // 创建 Roulette 对象并开始游戏
        new Roulette(System.out, System.in).play();
    }
}
    // 输出游戏标题
    public void play() {
        out.println("                                ROULETTE");
        out.println("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        out.println("WELCOME TO THE ROULETTE TABLE\n");
        out.print("DO YOU WANT INSTRUCTIONS? ");
        // 如果用户输入的第一个字符不是'n'，则打印游戏说明
        if (scanner.nextLine().toLowerCase().charAt(0) != 'n') {
            printInstructions();
        }

        do {
            // 获取用户下注信息
            Bet[] bets = queryBets();

            out.print("SPINNING...\n\n");
            // 生成1到39之间的随机数
            int result = random.nextInt(1, 39);

            /*
            Equivalent to following line
            if(result == 37) {
                out.println("00");
            } else if(result == 38) {
                out.println("0");
            } else if(RED_NUMBERS.contains(result)) {
                out.println(result + " RED");
            } else {
                out.println(result + " BLACK");
            }
             */
            // 根据随机数的不同情况输出结果
            out.println(switch (result) {
                case 37 -> "00";
                case 38 -> "0";
                default -> result + (RED_NUMBERS.contains(result) ? " RED" : " BLACK");
            });

            // 计算下注结果
            betResults(bets, result);
            out.println();

            out.println("TOTALS:\tME\t\tYOU");
            // 格式化输出庄家和玩家的余额
            out.format("\t\t%5d\t%d\n", houseBalance, playerBalance);
        } while (playAgain());
        // 如果玩家余额小于等于0，则输出信息
        if (playerBalance <= 0) {
            out.println("THANKS FOR YOUR MONEY\nI'LL USE IT TO BUY A SOLID GOLD ROULETTE WHEEL");
        } else {
            // 打印支票
            printCheck();
        }
        out.println("COME BACK SOON!");
    }
    // 查询下注信息并返回下注数组
    private Bet[] queryBets() {
        // 初始化下注数量为-1
        int numBets = -1;
        // 当下注数量小于1时循环
        while (numBets < 1) {
            // 提示用户输入下注数量
            out.print("HOW MANY BETS? ");
            try {
                // 尝试将用户输入的数量转换为整数
                numBets = Integer.parseInt(scanner.nextLine());
            } catch (NumberFormatException ignored) {
                // 如果转换失败则忽略异常
            }
        }

        // 创建下注数组
        Bet[] bets = new Bet[numBets];

        // 遍历下注数组
        for (int i = 0; i < numBets; i++) {
            // 当下注为空时循环
            while (bets[i] == null) {
                try {
                    // 提示用户输入下注号码和金额
                    out.print("NUMBER" + (i + 1) + "? ");
                    String[] values = scanner.nextLine().split(",");
                    int betNumber = Integer.parseInt(values[0]);
                    int betValue = Integer.parseInt(values[1]);

                    // 检查是否重复下注
                    for (int j = 0; j < i; j++) {
                        if (bets[j].num == betNumber) {
                            out.println("YOU MADE THAT BET ONCE ALREADY,DUM-DUM");
                            betNumber = -1; //Since -1 is out of the range, this will throw it out at the end
                        }
                    }

                    // 检查下注号码和金额的有效性
                    if (betNumber > 0 && betNumber <= 50 && betValue >= 5 && betValue <= 500) {
                        // 创建新的下注对象
                        bets[i] = new Bet(betNumber,betValue);
                    }
                } catch (Exception ignored) {
                    // 捕获异常并忽略
                }
            }
        }
        // 返回下注数组
        return bets;
    }
    private void betResults(Bet[] bets, int num) {
        for (int i = 0; i < bets.length; i++) {
            Bet bet = bets[i];
            /*
            使用 switch 语句和三元运算符检查基于赌注值是否满足某个条件
            返回赌注金额应该乘以的系数，以获得结果值
             */
            int coefficient = switch (bet.num) {
                case 37 -> (num <= 12) ? 2 : -1;
                case 38 -> (num > 12 && num <= 24) ? 2 : -1;
                case 39 -> (num > 24 && num < 37) ? 2 : -1;
                case 40 -> (num < 37 && num % 3 == 1) ? 2 : -1;
                case 41 -> (num < 37 && num % 3 == 2) ? 2 : -1;
                case 42 -> (num < 37 && num % 3 == 0) ? 2 : -1;
                case 43 -> (num <= 18) ? 1 : -1;
                case 44 -> (num > 18 && num <= 36) ? 1 : -1;
                case 45 -> (num % 2 == 0) ? 1 : -1;
                case 46 -> (num % 2 == 1) ? 1 : -1;
                case 47 -> RED_NUMBERS.contains(num) ? 1 : -1;
                case 48 -> !RED_NUMBERS.contains(num) ? 1 : -1;
                case 49 -> (num == 37) ? 35 : -1;
                case 50 -> (num == 38) ? 35 : -1;
                default -> (bet.num < 49 && bet.num == num) ? 35 : -1;
            };

            int betResult = bet.amount * coefficient;

            if (betResult < 0) {
                out.println("YOU LOSE " + -betResult + " DOLLARS ON BET " + (i + 1));
            } else {
                out.println("YOU WIN " + betResult + " DOLLARS ON BET " + (i + 1));
            }

            playerBalance += betResult;
            houseBalance -= betResult;
        }
    }
    // 检查玩家是否要再玩一次游戏
    private boolean playAgain() {

        // 如果玩家余额小于等于0，输出信息并返回false
        if (playerBalance <= 0) {
            out.println("OOPS! YOU JUST SPENT YOUR LAST DOLLAR!");
            return false;
        } 
        // 如果庄家余额小于等于0，输出信息，重置玩家和庄家余额，并返回false
        else if (houseBalance <= 0) {
            out.println("YOU BROKE THE HOUSE!");
            playerBalance = 101000;
            houseBalance = 0;
            return false;
        } 
        // 否则，询问玩家是否要再玩一次，并返回输入结果是否为'y'
        else {
            out.println("PLAY AGAIN?");
            return scanner.nextLine().toLowerCase().charAt(0) == 'y';
        }
    }

    // 打印支票
    private void printCheck() {
        out.print("TO WHOM SHALL I MAKE THE CHECK? ");
        String name = scanner.nextLine();

        out.println();
        // 打印72个'-'
        for (int i = 0; i < 72; i++) {
            out.print("-");
        }
        out.println();

        // 打印"CHECK NO."和随机数
        for (int i = 0; i < 50; i++) {
            out.print(" ");
        }
        out.println("CHECK NO. " + random.nextInt(0, 100));

        // 打印当前日期和时间
        for (int i = 0; i < 40; i++) {
            out.print(" ");
        }
        out.println(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE));
        out.println();

        // 打印支票收款人和金额
        out.println("PAY TO THE ORDER OF -----" + name + "----- $" + (playerBalance));
        out.println();

        // 打印银行信息
        for (int i = 0; i < 40; i++) {
            out.print(" ");
        }
        out.println("THE MEMORY BANK OF NEW YORK");

        for (int i = 0; i < 40; i++) {
            out.print(" ");
        }
        out.println("THE COMPUTER");

        for (int i = 0; i < 40; i++) {
            out.print(" ");
        }
        out.println("----------X-----");

        // 打印72个'-'
        for (int i = 0; i < 72; i++) {
            out.print("-");
        }
        out.println();
    }

    // 下注类
    public class Bet {

        final int num, amount;

        // 构造函数，初始化下注号码和金额
        public Bet(int num, int amount) {
            this.num = num;
            this.amount = amount;
        }
    }
# 闭合前面的函数定义
```