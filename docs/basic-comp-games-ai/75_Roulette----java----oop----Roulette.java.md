# `basic-computer-games\75_Roulette\java\oop\Roulette.java`

```py
// 导入所需的类
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.time.format.FormatStyle;

// 创建名为 Roulette 的类
public class Roulette {
    // 主方法
    public static void main(String args[]) throws Exception {
        // 创建 Roulette 对象并调用 play 方法
        Roulette r = new Roulette();
        r.play();
    }

    // 声明私有变量
    private BufferedReader reader;
    private PrintStream writer;

    // 表示庄家拥有的金额
    private int house;
    // 表示玩家拥有的金额
    private int player;
    // 创建 Wheel 对象
    private Wheel wheel = new Wheel();

    // 构造方法
    public Roulette() {
        // 初始化输入输出流和金额
        reader = new BufferedReader(new InputStreamReader(System.in));
        writer = System.out;
        house = 100000;
        player = 1000;
    }

    // 用于测试/作弊模式 - 将随机数生成器设置为已知值
    private void setSeed(long l) {
        wheel.setSeed(l);
    }
}
    // 定义一个名为 play 的方法
    public void play() {
        try {
            // 调用 intro 方法，显示游戏介绍
            intro();
            // 输出欢迎信息，并询问是否需要游戏说明
            writer.println("WELCOME TO THE ROULETTE TABLE\n" +
                           "DO YOU WANT INSTRUCTIONS");
            // 读取用户输入的指令
            String instr = reader.readLine();
            // 如果用户输入的指令不是以字母"N"开头，则显示游戏说明
            if (!instr.toUpperCase().startsWith("N"))
                instructions();

            // 当下注并旋转轮盘的方法返回 true 时，继续游戏
            while (betAndSpin()) { // returns true if the game is to continue
            }

            // 如果玩家的金额小于等于0
            if (player <= 0) {
                // 玩家钱包空了，显示感谢信息
                writer.println("THANKS FOR YOUR MONEY.\nI'LL USE IT TO BUY A SOLID GOLD ROULETTE WHEEL");
            } else {
                // 玩家还有钱，打印支票
                writer.println("TO WHOM SHALL I MAKE THE CHECK");

                // 读取收款人信息
                String payee = reader.readLine();

                // 打印支票信息
                writer.println("-".repeat(72));
                tab(50); writer.println("CHECK NO. " + (new Random().nextInt(100) + 1));
                writer.println();
                tab(40); writer.println(LocalDate.now().format(DateTimeFormatter.ofLocalizedDate(FormatStyle.LONG)));
                writer.println("\n\nPAY TO THE ORDER OF-----" + payee + "-----$ " + player);
                writer.print("\n\n");
                tab(10); writer.println("THE MEMORY BANK OF NEW YORK\n");
                tab(40); writer.println("THE COMPUTER");
                tab(40); writer.println("----------X-----\n");
                writer.println("-".repeat(72));
                writer.println("COME BACK SOON!\n");
            }
        }
        // 捕获可能出现的 IO 异常
        catch (IOException e) {
            // 如果出现异常，打印错误信息
            System.err.println("System error:\n" + e);
        }
    }

    /* Write the starting introduction */
    // 定义一个名为 intro 的私有方法，用于显示游戏介绍
    private void intro() throws IOException {
        // 调用 tab 方法，设置输出格式，显示游戏标题
        tab(32); writer.println("ROULETTE");
        tab(15); writer.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n");
    }

    /* Display the game instructions */
    // 显示游戏说明的方法
    // 显示游戏的下注说明
    private void instructions() {
        // 创建包含游戏下注说明的字符串数组
        String[] instLines = new String[] {
            "THIS IS THE BETTING LAYOUT",
            "  (*=RED)",
            "" ,
            " 1*    2     3*",
            " 4     5*    6 ",
            " 7*    8     9*",
            "10    11    12*",
            "---------------",
            "13    14*   15 ",
            "16*   17    18*",
            "19*   20    21*",
            "22    23*   24 ",
            "---------------",
            "25*   26    27*",
            "28    29    30*",
            "31    32*   33 ",
            "34*   35    36*",
            "---------------",
            "    00    0    ",
            "" ,
            "TYPES OF BETS",
            ""  ,
            "THE NUMBERS 1 TO 36 SIGNIFY A STRAIGHT BET",
            "ON THAT NUMBER.",
            "THESE PAY OFF 35:1",
            ""  ,
            "THE 2:1 BETS ARE:",
            " 37) 1-12     40) FIRST COLUMN",
            " 38) 13-24    41) SECOND COLUMN",
            " 39) 25-36    42) THIRD COLUMN",
            ""  ,
            "THE EVEN MONEY BETS ARE:",
            " 43) 1-18     46) ODD",
            " 44) 19-36    47) RED",
            " 45) EVEN     48) BLACK",
            "",
            " 49)0 AND 50)00 PAY OFF 35:1",
            " NOTE: 0 AND 00 DO NOT COUNT UNDER ANY",
            "       BETS EXCEPT THEIR OWN.",
            "",
            "WHEN I ASK FOR EACH BET, TYPE THE NUMBER",
            "AND THE AMOUNT, SEPARATED BY A COMMA.",
            "FOR EXAMPLE: TO BET $500 ON BLACK, TYPE 48,500",
            "WHEN I ASK FOR A BET.",
            "",
            "THE MINIMUM BET IS $5, THE MAXIMUM IS $500.",
            "" };
        // 将游戏下注说明打印到输出流
        writer.println(String.join("\n", instLines));
    }

    /* Take a set of bets from the player, then spin the wheel and work out the winnings *
     * This returns true if the game is to continue afterwards
     */
    }

    // utility to print n spaces for formatting
    # 定义一个私有方法，用于在输出中打印 n 个空格
    private void tab(int n) {
        # 使用字符串的 repeat 方法，打印 n 个空格
        writer.print(" ".repeat(n));
    }
# 闭合前面的函数定义
```