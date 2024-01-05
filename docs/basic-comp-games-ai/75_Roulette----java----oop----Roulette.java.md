# `75_Roulette\java\oop\Roulette.java`

```
# 导入所需的包
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

public class Roulette {
    public static void main(String args[]) throws Exception {
        Roulette r = new Roulette();
        r.play();
    }

    # 创建一个用于读取输入的 BufferedReader 对象
    private BufferedReader reader;
    # 创建一个用于输出的 PrintStream 对象
    private PrintStream writer;
    private int house;      // 存储庄家的金额
    private int player;     // 存储玩家的金额
    private Wheel wheel = new Wheel();  // 创建一个轮盘对象

    public Roulette() {
        reader = new BufferedReader(new InputStreamReader(System.in));  // 创建一个从控制台读取输入的对象
        writer = System.out;  // 创建一个输出到控制台的对象
        house = 100000;  // 初始化庄家的金额为 100000
        player = 1000;   // 初始化玩家的金额为 1000
    }

    // for a test / cheat mode -- set the random number generator to a known value
    private void setSeed(long l) {
        wheel.setSeed(l);  // 设置随机数生成器的种子值
    }

    public void play() {
        try {
            intro();  // 调用介绍方法
            writer.println("WELCOME TO THE ROULETTE TABLE\n" +  // 输出欢迎信息
            // 提示用户是否需要游戏说明
            writer.println("DO YOU WANT INSTRUCTIONS");
            // 读取用户输入的指令
            String instr = reader.readLine();
            // 如果用户输入的指令不是以"N"开头，则显示游戏说明
            if (!instr.toUpperCase().startsWith("N"))
                instructions();

            // 当下注并旋转轮盘时，返回true表示游戏继续进行
            while (betAndSpin()) { 
            }

            // 如果玩家的钱小于等于0，则玩家已经用完了所有的钱
            if (player <= 0) {
                // 玩家用完了所有的钱
                writer.println("THANKS FOR YOUR MONEY.\nI'LL USE IT TO BUY A SOLID GOLD ROULETTE WHEEL");
            } else {
                // 玩家还有钱 -- 打印支票
                writer.println("TO WHOM SHALL I MAKE THE CHECK");

                // 读取收款人信息
                String payee = reader.readLine();

                // 打印支票信息
                writer.println("-".repeat(72));
                tab(50); writer.println("CHECK NO. " + (new Random().nextInt(100) + 1));
                writer.println();
                tab(40); writer.println(LocalDate.now().format(DateTimeFormatter.ofLocalizedDate(FormatStyle.LONG)));  // 打印当前日期
                writer.println("\n\nPAY TO THE ORDER OF-----" + payee + "-----$ " + player);  // 打印付款对象和金额
                writer.print("\n\n");  // 打印空行
                tab(10); writer.println("THE MEMORY BANK OF NEW YORK\n");  // 打印纽约的银行名称
                tab(40); writer.println("THE COMPUTER");  // 打印计算机名称
                tab(40); writer.println("----------X-----\n");  // 打印分隔线
                writer.println("-".repeat(72));  // 打印一条长横线
                writer.println("COME BACK SOON!\n");  // 打印提示信息
            }
        }
        catch (IOException e) {
            // this should not happen
            System.err.println("System error:\n" + e);  // 捕获并打印IO异常
        }
    }

    /* Write the starting introduction */
    private void intro() throws IOException {
        tab(32); writer.println("ROULETTE");  // 打印标题
        tab(15); writer.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n");  // 打印创意计算的地点
    }

    /* 显示游戏说明 */
    private void instructions() {
        // 创建包含游戏说明的字符串数组
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
# 创建一个多行字符串，包含了赌注类型的说明
bets_description = """
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
"""
            "",  # 空字符串
            " 49)0 AND 50)00 PAY OFF 35:1",  # 字符串
            " NOTE: 0 AND 00 DO NOT COUNT UNDER ANY",  # 字符串
            "       BETS EXCEPT THEIR OWN.",  # 字符串
            "",  # 空字符串
            "WHEN I ASK FOR EACH BET, TYPE THE NUMBER",  # 字符串
            "AND THE AMOUNT, SEPARATED BY A COMMA.",  # 字符串
            "FOR EXAMPLE: TO BET $500 ON BLACK, TYPE 48,500",  # 字符串
            "WHEN I ASK FOR A BET.",  # 字符串
            "",  # 空字符串
            "THE MINIMUM BET IS $5, THE MAXIMUM IS $500.",  # 字符串
            "" };  # 字符串数组
        writer.println(String.join("\n", instLines));  # 将字符串数组连接成一个字符串并打印出来
    }

    /* Take a set of bets from the player, then spin the wheel and work out the winnings *
     * This returns true if the game is to continue afterwards
     */
    private boolean betAndSpin() throws IOException {  # 定义一个私有方法，接收玩家的一组赌注，然后旋转轮盘并计算赢利，如果游戏继续返回true
        int betCount = 0;  # 初始化赌注数量为0
        while (betCount == 0) {   // 持续询问下注数量，直到得到有效答案
            try {
                writer.println("HOW MANY BETS");  // 向输出流写入请求下注数量的消息
                String howMany = reader.readLine();  // 从输入流读取下注数量
                betCount = Integer.parseInt(howMany.strip());  // 将输入的下注数量转换为整数

                if ((betCount < 1) || (betCount > 100)) betCount = 0; // 如果下注数量不在有效范围内，则设置为零并重新询问
            }
            catch (NumberFormatException e) {
                // 如果输入不是一个数字，则捕获异常
                writer.println("INPUT ERROR");  // 向输出流写入输入错误的消息
            }
        }

        HashSet<Integer> betsMade = new HashSet<>(); // 存储已经下注的目标，以便识别重复
        ArrayList<Bet> bets = new ArrayList<>();     // 存储本轮的所有下注

        while (bets.size() < betCount) {
            Bet bet = new Bet(0, 0);                 // 创建一个无效的下注对象占位
            while (!bet.isValid()) {                 // 持续询问，直到下注有效为止
                try {
                    writer.println("NUMBER " + (bets.size() + 1));  // 输出下注号码
                    String fields[] = reader.readLine().split(",");  // 读取输入的号码并以逗号分隔
                    if (fields.length == 2) {  // 如果输入的号码长度为2
                        bet = new Bet(Integer.parseInt(fields[0].strip()),  // 创建一个新的下注对象
                                      Integer.parseInt(fields[1].strip()));
                    }
                }
                catch (NumberFormatException e) {  // 捕获输入格式错误的异常
                    writer.println("INPUT ERROR");  // 输出输入错误信息
                }
            }

            // 检查是否已经有相同目标的下注
            if (betsMade.contains(bet.target)) {  // 如果已经有相同目标的下注
                writer.println("YOU MADE THAT BET ONCE ALREADY,DUM-DUM");  // 输出已经下注过的信息
            } else {
                betsMade.add(bet.target); // 记录已经下注的目标
                bets.add(bet);  // 添加下注到下注列表
        }

        writer.println("SPINNING\n\n");  # 打印"SPINNING"到输出流

        wheel.spin(); // this deliberately takes some random amount of time  # 调用轮盘对象的spin方法，这里故意花费一些随机的时间

        writer.println(wheel.value());  # 打印轮盘当前的值到输出流

        // go through the bets, and evaluate each one  # 遍历所有的赌注，并评估每一个

        int betNumber = 1;  # 初始化赌注编号为1
        for (Bet b : bets) {  # 遍历赌注列表
            int multiplier = b.winsOn(wheel);  # 调用赌注对象的winsOn方法，计算赌注的倍数
            if (multiplier == 0) {  # 如果倍数为0
                // lost the amount of the bet  # 输掉了赌注的金额
                writer.println("YOU LOSE " + b.amount + " DOLLARS ON BET " + betNumber);  # 打印输掉赌注金额的消息到输出流
                house += b.amount;  # 庄家增加赌注金额
                player -= b.amount;  # 玩家减少赌注金额
            } else {
                // won the amount of the bet, multiplied by the odds  # 赢得了赌注金额，乘以赔率
                int winnings = b.amount * multiplier;  // 计算赢得的奖金，赌注金额乘以赔率
                writer.println("YOU WIN " + winnings + " DOLLARS ON BET " + betNumber);  // 打印赢得的奖金和赌注编号
                house -= winnings;  // 减去赢得的奖金，更新庄家的金额
                player += winnings;  // 增加赢得的奖金，更新玩家的金额
            }
            ++betNumber;  // 更新赌注编号
        }

        writer.println("\nTOTALS:\tME\tYOU\n\t" + house + "\t" + player);  // 打印庄家和玩家的总金额

        if (player <= 0) {
            writer.println("OOPS! YOU JUST SPENT YOUR LAST DOLLAR");  // 打印玩家已经没有钱了
            return false;     // 玩家已经没有钱了，不再重复游戏
        }
        if (house <= 0) {
            writer.println("YOU BROKE THE HOUSE!");  // 打印玩家赢得了庄家的所有金额
            player = 101000;  // 玩家不能赢得比庄家初始金额更多的金额
            return false;     // 庄家已经没有钱了，不再重复游戏
        }
        // player still has money, and the house still has money, so ask the player
        // if they want to continue
        writer.println("AGAIN"); // 输出提示信息，要求玩家输入是否继续
        String doContinue = reader.readLine(); // 读取玩家输入的内容

        // repeat if the answer was not "n" or "no"
        return (!doContinue.toUpperCase().startsWith("N")); // 将玩家输入转换为大写并检查是否以"N"开头，如果不是则返回true，表示继续游戏
    }

    // utility to print n spaces for formatting
    private void tab(int n) {
        writer.print(" ".repeat(n)); // 打印n个空格，用于格式化输出
    }
}
```