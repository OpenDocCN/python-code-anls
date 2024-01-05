# `75_Roulette\java\iterative\Roulette.java`

```
import java.io.InputStream;  // 导入用于处理输入流的类
import java.io.PrintStream;  // 导入用于处理输出流的类
import java.time.LocalDateTime;  // 导入用于处理日期时间的类
import java.time.format.DateTimeFormatter;  // 导入用于格式化日期时间的类
import java.util.Random;  // 导入用于生成随机数的类
import java.util.Scanner;  // 导入用于接收用户输入的类
import java.util.Set;  // 导入用于存储不重复元素的集合类

public class Roulette {

    private static Set<Integer> RED_NUMBERS;  // 声明并初始化存储红色数字的集合

    static {
        RED_NUMBERS = Set.of(1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36);  // 初始化红色数字的集合
    }

    private PrintStream out;  // 声明用于输出的流
    private Scanner scanner;  // 声明用于接收输入的扫描器
    private int houseBalance, playerBalance;  // 声明庄家和玩家的余额
    private Random random;  // 声明用于生成随机数的对象
    public Roulette(PrintStream out, InputStream in) {
        this.out = out;  // 将传入的输出流赋值给类的输出流变量
        this.scanner = new Scanner(in);  // 使用传入的输入流创建一个 Scanner 对象
        houseBalance = 100000;  // 初始化庄家的余额为 100000
        playerBalance = 1000;  // 初始化玩家的余额为 1000
        random = new Random();  // 创建一个随机数生成器对象
    }

    public static void main(String[] args) {
        new Roulette(System.out, System.in).play();  // 创建一个 Roulette 对象并调用 play 方法
    }

    public void play() {
        out.println("                                ROULETTE");  // 输出赌轮游戏的标题
        out.println("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 输出游戏的创作者和地点信息
        out.println("WELCOME TO THE ROULETTE TABLE\n");  // 输出欢迎信息
        out.print("DO YOU WANT INSTRUCTIONS? ");  // 提示玩家是否需要游戏说明
        if (scanner.nextLine().toLowerCase().charAt(0) != 'n') {  // 如果玩家输入的第一个字符不是 'n'，则打印游戏说明
            printInstructions();
        }

        do {

            // 查询下注信息
            Bet[] bets = queryBets();

            // 输出提示信息
            out.print("SPINNING...\n\n");
            // 生成1到39之间的随机数
            int result = random.nextInt(1, 39);

            /*
            // 等同于以下代码
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
            */
            // 使用 switch 语句根据 result 的值进行不同的处理
            out.println(switch (result) {
                case 37 -> "00"; // 如果 result 为 37，则输出 "00"
                case 38 -> "0"; // 如果 result 为 38，则输出 "0"
                default -> result + (RED_NUMBERS.contains(result) ? " RED" : " BLACK"); // 其他情况下，如果 result 在 RED_NUMBERS 中，则输出 result + " RED"，否则输出 result + " BLACK"
            });

            // 调用 betResults 方法，传入 bets 和 result 作为参数
            betResults(bets, result);
            // 输出空行
            out.println();

            // 输出 "TOTALS:\tME\t\tYOU"
            out.println("TOTALS:\tME\t\tYOU");
            // 格式化输出 houseBalance 和 playerBalance 的值
            out.format("\t\t%5d\t%d\n", houseBalance, playerBalance);
        } while (playAgain()); // 循环，直到 playAgain() 返回 false
        if (playerBalance <= 0) { // 如果 playerBalance 小于等于 0
            // 输出 "THANKS FOR YOUR MONEY\nI'LL USE IT TO BUY A SOLID GOLD ROULETTE WHEEL"
            out.println("THANKS FOR YOUR MONEY\nI'LL USE IT TO BUY A SOLID GOLD ROULETTE WHEEL");
        } else {
            // 调用 printCheck 方法
            printCheck();
        }
        // 输出 "COME BACK SOON!"
        out.println("COME BACK SOON!");
    }
    public void printInstructions() {
        out.println();  // 打印空行
        out.println("THIS IS THE BETTING LAYOUT");  // 打印赌注布局的标题
        out.println("  (*=RED)");  // 打印红色标记的说明
        out.println();
        out.println(" 1*    2     3*");  // 打印赌注布局的第一行
        out.println(" 4     5*    6 ");  // 打印赌注布局的第二行
        out.println(" 7*    8     9*");  // 打印赌注布局的第三行
        out.println("10    11    12*");  // 打印赌注布局的第四行
        out.println("---------------");  // 打印分隔线
        out.println("13    14*   15 ");  // 打印赌注布局的第五行
        out.println("16*   17    18*");  // 打印赌注布局的第六行
        out.println("19*   20    21*");  // 打印赌注布局的第七行
        out.println("22    23*   24 ");  // 打印赌注布局的第八行
        out.println("---------------");  // 打印分隔线
        out.println("25*   26    27*");  // 打印赌注布局的第九行
        out.println("28    29    30*");  // 打印赌注布局的第十行
        out.println("31    32*   33 ");  // 打印赌注布局的第十一行
        out.println("34*   35    36*");  // 打印赌注布局的第十二行
    }
        # 打印分隔线
        out.println("---------------")
        out.println("    00    0    ")
        out.println()
        out.println("TYPES OF BETS")
        out.println()
        out.println("THE NUMBERS 1 TO 36 SIGNIFY A STRAIGHT BET")
        out.println("ON THAT NUMBER.")
        out.println("THESE PAY OFF 35:1")
        out.println()
        out.println("THE 2:1 BETS ARE:")
        out.println(" 37) 1-12     40) FIRST COLUMN")
        out.println(" 38) 13-24    41) SECOND COLUMN")
        out.println(" 39) 25-36    42) THIRD COLUMN")
        out.println()
        out.println("THE EVEN MONEY BETS ARE:")
        out.println(" 43) 1-18     46) ODD")
        out.println(" 44) 19-36    47) RED")
        out.println(" 45) EVEN     48) BLACK")
        out.println()
        out.println(" 49)0 AND 50)00 PAY OFF 35:1")
        // 打印提示信息，说明0和00不计入任何下注，除非是它们自己
        out.println(" NOTE: 0 AND 00 DO NOT COUNT UNDER ANY");
        // 打印提示信息，说明每个下注都需要输入数字和金额，用逗号分隔
        out.println("       BETS EXCEPT THEIR OWN.");
        out.println();
        // 打印提示信息，要求输入每个下注的数字和金额
        out.println("WHEN I ASK FOR EACH BET, TYPE THE NUMBER");
        out.println("AND THE AMOUNT, SEPARATED BY A COMMA.");
        // 打印示例，说明如何输入下注的格式
        out.println("FOR EXAMPLE: TO BET $500 ON BLACK, TYPE 48,500");
        out.println("WHEN I ASK FOR A BET.");
        out.println();
        // 打印提示信息，说明最小下注金额为$5，最大下注金额为$500
        out.println("THE MINIMUM BET IS $5, THE MAXIMUM IS $500.");
    }

    // 查询下注信息的方法
    private Bet[] queryBets() {
        // 初始化下注数量为-1，用于循环判断输入的下注数量是否合法
        int numBets = -1;
        // 当下注数量小于1时，循环询问下注数量
        while (numBets < 1) {
            out.print("HOW MANY BETS? ");
            try {
                // 尝试将输入的下注数量转换为整数
                numBets = Integer.parseInt(scanner.nextLine());
            } catch (NumberFormatException ignored) {
                // 捕获输入格式错误的异常，忽略并继续循环
            }
        }
        Bet[] bets = new Bet[numBets];  // 创建一个 Bet 类型的数组，用于存储赌注信息，数组长度为 numBets

        for (int i = 0; i < numBets; i++) {  // 循环遍历赌注数组
            while (bets[i] == null) {  // 当当前位置的赌注为空时执行以下操作
                try {  // 尝试执行以下代码
                    out.print("NUMBER" + (i + 1) + "? ");  // 输出提示信息，要求用户输入赌注号码
                    String[] values = scanner.nextLine().split(",");  // 读取用户输入的赌注号码和赌注金额，并以逗号分隔存储到数组 values 中
                    int betNumber = Integer.parseInt(values[0]);  // 将数组 values 中的第一个元素转换为整数类型，表示赌注号码
                    int betValue = Integer.parseInt(values[1]);  // 将数组 values 中的第二个元素转换为整数类型，表示赌注金额

                    for (int j = 0; j < i; j++) {  // 循环遍历已有的赌注
                        if (bets[j].num == betNumber) {  // 如果已有的赌注号码与当前输入的赌注号码相同
                            out.println("YOU MADE THAT BET ONCE ALREADY,DUM-DUM");  // 输出提示信息，表示已经下过相同的赌注
                            betNumber = -1; //Since -1 is out of the range, this will throw it out at the end
                        }
                    }

                    if (betNumber > 0 && betNumber <= 50 && betValue >= 5 && betValue <= 500) {  // 如果赌注号码在1到50之间，赌注金额在5到500之间
                        bets[i] = new Bet(betNumber,betValue);  // 创建一个新的 Bet 对象，表示当前的赌注信息
            }
        }
    }

    // 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
    def read_zip(fname):
        // 根据 ZIP 文件名读取其二进制，封装成字节流
        bio = BytesIO(open(fname, 'rb').read())
        // 使用字节流里面内容创建 ZIP 对象
        zip = zipfile.ZipFile(bio, 'r')
        // 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
        fdict = {n:zip.read(n) for n in zip.namelist()}
        // 关闭 ZIP 对象
        zip.close()
        // 返回结果字典
        return fdict
                case 41 -> (num < 37 && num % 3 == 2) ? 2 : -1;  // 如果num小于37且num除以3的余数为2，则返回2，否则返回-1
                case 42 -> (num < 37 && num % 3 == 0) ? 2 : -1;  // 如果num小于37且num除以3的余数为0，则返回2，否则返回-1
                case 43 -> (num <= 18) ? 1 : -1;  // 如果num小于等于18，则返回1，否则返回-1
                case 44 -> (num > 18 && num <= 36) ? 1 : -1;  // 如果num大于18且小于等于36，则返回1，否则返回-1
                case 45 -> (num % 2 == 0) ? 1 : -1;  // 如果num为偶数，则返回1，否则返回-1
                case 46 -> (num % 2 == 1) ? 1 : -1;  // 如果num为奇数，则返回1，否则返回-1
                case 47 -> RED_NUMBERS.contains(num) ? 1 : -1;  // 如果RED_NUMBERS集合包含num，则返回1，否则返回-1
                case 48 -> !RED_NUMBERS.contains(num) ? 1 : -1;  // 如果RED_NUMBERS集合不包含num，则返回1，否则返回-1
                case 49 -> (num == 37) ? 35 : -1;  // 如果num等于37，则返回35，否则返回-1
                case 50 -> (num == 38) ? 35 : -1;  // 如果num等于38，则返回35，否则返回-1
                default -> (bet.num < 49 && bet.num == num) ? 35 : -1;  // 如果bet.num小于49且等于num，则返回35，否则返回-1
            };

            int betResult = bet.amount * coefficient;  // 计算赌注结果

            if (betResult < 0) {  // 如果赌注结果小于0
                out.println("YOU LOSE " + -betResult + " DOLLARS ON BET " + (i + 1));  // 输出输掉的赌注金额和赌注编号
            } else {  // 否则
                out.println("YOU WIN " + betResult + " DOLLARS ON BET " + (i + 1));  // 输出赢得的赌注金额和赌注编号
            }
            playerBalance += betResult;  # 增加玩家的余额，根据赌注结果
            houseBalance -= betResult;   # 减少庄家的余额，根据赌注结果
        }
    }

    private boolean playAgain() {

        if (playerBalance <= 0) {  # 如果玩家余额小于等于0
            out.println("OOPS! YOU JUST SPENT YOUR LAST DOLLAR!");  # 输出信息告知玩家已经花光最后一美元
            return false;  # 返回false，表示不再玩游戏
        } else if (houseBalance <= 0) {  # 如果庄家余额小于等于0
            out.println("YOU BROKE THE HOUSE!");  # 输出信息告知玩家已经赢得了庄家的所有余额
            playerBalance = 101000;  # 重置玩家余额为101000
            houseBalance = 0;  # 重置庄家余额为0
            return false;  # 返回false，表示不再玩游戏
        } else {
            out.println("PLAY AGAIN?");  # 输出信息询问玩家是否再玩一次
            return scanner.nextLine().toLowerCase().charAt(0) == 'y';  # 返回玩家输入的是否为'y'的布尔值，表示是否再玩一次
        }
    } // 结束 printCheck 方法的定义

    private void printCheck() { // 定义一个名为 printCheck 的私有方法
        out.print("TO WHOM SHALL I MAKE THE CHECK? "); // 打印提示信息要求输入收款人姓名
        String name = scanner.nextLine(); // 从控制台读取用户输入的收款人姓名

        out.println(); // 打印空行

        for (int i = 0; i < 72; i++) { // 循环打印 72 个连字符
            out.print("-");
        }
        out.println(); // 打印换行

        for (int i = 0; i < 50; i++) { // 循环打印 50 个空格
            out.print(" ");
        }
        out.println("CHECK NO. " + random.nextInt(0, 100)); // 打印支票号码

        for (int i = 0; i < 40; i++) { // 循环打印 40 个空格
            out.print(" ");
        }
        # 打印当前日期时间，使用 ISO_LOCAL_DATE 格式
        out.println(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE));
        out.println();

        # 打印支票的收款人姓名和金额
        out.println("PAY TO THE ORDER OF -----" + name + "----- $" + (playerBalance));
        out.println();

        # 打印空格，用于格式化支票
        for (int i = 0; i < 40; i++) {
            out.print(" ");
        }
        out.println("THE MEMORY BANK OF NEW YORK");

        # 打印空格，用于格式化支票
        for (int i = 0; i < 40; i++) {
            out.print(" ");
        }
        out.println("THE COMPUTER");

        # 打印空格，用于格式化支票
        for (int i = 0; i < 40; i++) {
            out.print(" ");
        }
        out.println("----------X-----");
        for (int i = 0; i < 72; i++) {
            out.print("-");
        }
        out.println();
    }
```
这部分代码是一个 for 循环，用于打印 72 个连字符 "-"，然后换行。

```
    public class Bet {

        final int num, amount;

        public Bet(int num, int amount) {
            this.num = num;
            this.amount = amount;
        }
    }
```
这部分代码定义了一个名为 Bet 的类，该类有两个属性 num 和 amount，分别表示数字和金额。类中还有一个构造函数，用于初始化这两个属性。
```