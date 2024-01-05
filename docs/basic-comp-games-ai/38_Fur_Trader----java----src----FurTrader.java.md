# `38_Fur_Trader\java\src\FurTrader.java`

```
import java.util.ArrayList;  # 导入 ArrayList 类
import java.util.Arrays;  # 导入 Arrays 类
import java.util.Scanner;  # 导入 Scanner 类

/**
 * Game of Fur Trader
 * <p>
 * Based on the Basic game of Fur Trader here
 * https://github.com/coding-horror/basic-computer-games/blob/main/38%20Fur%20Trader/furtrader.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class FurTrader {

    public static final double START_SAVINGS_AMOUNT = 600.0;  # 设置初始储蓄金额为 600.0
    public static final int STARTING_FURS = 190;  # 设置初始毛皮数量为 190

    public static final int FORT_HOCHELAGA_MONTREAL = 1;  # 设置 FORT_HOCHELAGA_MONTREAL 的值为 1
    public static final int FORT_STADACONA_QUEBEC = 2;  # 设置 FORT_STADACONA_QUEBEC 的值为 2
    # 定义常量 FORT_NEW_YORK，值为 3
    public static final int FORT_NEW_YORK = 3;

    # 定义常量 MINK，值为 "MINK"
    public static final String MINK = "MINK";
    # 定义常量 MINK_ENTRY，值为 0
    public static final int MINK_ENTRY = 0;
    # 定义常量 BEAVER，值为 "BEAVER"
    public static final String BEAVER = "BEAVER";
    # 定义常量 BEAVER_ENTRY，值为 1
    public static final int BEAVER_ENTRY = 1;
    # 定义常量 ERMINE，值为 "ERMINE"
    public static final String ERMINE = "ERMINE";
    # 定义常量 ERMINE_ENTRY，值为 2
    public static final int ERMINE_ENTRY = 2;
    # 定义常量 FOX，值为 "FOX"
    public static final String FOX = "FOX";
    # 定义常量 FOX_ENTRY，值为 3
    public static final int FOX_ENTRY = 3;

    # 用于键盘输入的 Scanner 对象
    private final Scanner kbScanner;

    # 定义枚举类型 GAME_STATE，包括 STARTUP, INIT, TRADE_AT_FORT, TRADE_SUMMARY, TRADE_AGAIN
    private enum GAME_STATE {
        STARTUP,
        INIT,
        TRADE_AT_FORT,
        TRADE_SUMMARY,
        TRADE_AGAIN,
        GAME_OVER
    }
    // 结束游戏

    // 当前游戏状态
    private GAME_STATE gameState;

    // 存款
    private double savings;
    // 貂皮价格
    private double minkPrice;
    // 海狸皮价格
    private double beaverPrice;
    // 貂皮价格
    private double erminePrice;
    // 狐狸皮价格
    private double foxPrice;

    // 购买的皮毛列表
    private ArrayList<Pelt> pelts;

    // 游戏是否已经进行过一次
    private boolean playedOnce;

    // 构造函数
    public FurTrader() {
        // 创建键盘输入扫描器
        kbScanner = new Scanner(System.in);
        // 初始化游戏状态
        gameState = GAME_STATE.INIT;
        // 游戏是否已经进行过一次的标志
        playedOnce = false;
    }

    /**
     * Main game loop
     */
    public void play() {

        do {
            switch (gameState) {

                case INIT:
                    // 初始化游戏状态
                    savings = START_SAVINGS_AMOUNT;

                    // 只显示初始游戏标题一次
                    if (!playedOnce) {
                        playedOnce = true;
                        gameStartupMessage();
                    }

                    // 游戏介绍
                    intro();
                    // 如果玩家选择交易毛皮
                    if (yesEntered(displayTextAndGetInput("DO YOU WISH TO TRADE FURS? "))) {
                        // 打印玩家的存款和起始毛皮数量
                        System.out.println("YOU HAVE $" + formatNumber(savings) + " SAVINGS.");
                        System.out.println("AND " + STARTING_FURS + " FURS TO BEGIN THE EXPEDITION.");

                        // 创建一个新的毛皮数组
                        pelts = initPelts();
                        // 设置游戏状态为开始
                        gameState = GAME_STATE.STARTUP;
                    } else {
                        // 如果玩家选择不交易毛皮，设置游戏状态为游戏结束
                        gameState = GAME_STATE.GAME_OVER;
                    }

                    break;

                case STARTUP:

                    // 重置毛皮数量（所有类型）
                    resetPelts();

                    // 这是在处理所有毛皮后我们将前往的地方
                    gameState = GAME_STATE.TRADE_AT_FORT;
                    int totalPelts = 0; // 初始化总皮毛数量为0
                    // 遍历所有类型的皮毛
                    for (int i = 0; i < pelts.size(); i++) { // 循环遍历皮毛列表
                        Pelt pelt = pelts.get(i); // 获取当前索引处的皮毛对象
                        int number = getPeltCount(pelt.getName()); // 获取当前皮毛的数量
                        totalPelts += number; // 将当前皮毛数量累加到总皮毛数量中
                        if (totalPelts > STARTING_FURS) { // 如果总皮毛数量超过了初始皮毛数量
                            System.out.println("YOU MAY NOT HAVE THAT MANY FURS."); // 输出提示信息
                            System.out.println("DO NOT TRY TO CHEAT.  I CAN ADD."); // 输出提示信息
                            System.out.println("YOU MUST START AGAIN."); // 输出提示信息
                            System.out.println(); // 输出空行
                            // 重新开始游戏
                            gameState = GAME_STATE.INIT; // 将游戏状态设置为初始化状态
                            break; // 跳出循环
                        } else {
                            // 更新玩家输入的数量并保存回ArrayList
                            pelt.setPeltCount(number); // 设置当前皮毛的数量
                            pelts.set(i, pelt); // 更新ArrayList中的皮毛对象
                            // 玩家可能将他们的所有皮毛分配完
// 如果总皮毛数量等于初始皮毛数量，就不再继续询问是否添加更多皮毛
if (totalPelts == STARTING_FURS) {
    break;
}

// 只有在游戏状态不是启动状态时才进行交易部分的游戏
if (gameState != GAME_STATE.STARTUP) {
    // 在这里设置貂皮和海狸皮的默认价格，取决于在哪里交易这些皮毛，这些默认值可能会被其他值覆盖
    // 查看 tradeAt??? 方法以获取更多信息
    erminePrice = ((.15 * Math.random() + .95) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2);
    beaverPrice = ((.25 * Math.random() + 1.00) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2);
    System.out.println();
}
break;
                case TRADE_AT_FORT:  // 当交易在堡垒发生时执行以下操作

                    extendedTradingInfo();  // 调用extendedTradingInfo()函数，显示扩展交易信息
                    int answer = displayTextAndGetNumber("ANSWER 1, 2, OR 3. ");  // 显示文本并获取玩家输入的数字，将结果存储在answer变量中

                    System.out.println();  // 打印空行

                    // 现在显示他们即将交易的堡垒的详细信息，并给玩家选择是否继续的机会
                    // "No" 或 false 表示他们不想改变到另一个堡垒
                    if (!confirmFort(answer)) {  // 如果确认不交易在堡垒
                        switch (answer) {  // 根据玩家的选择进行不同的操作
                            case 1:  // 如果玩家选择1
                                tradeAtFortHochelagaMontreal();  // 进行Hochelaga Montreal堡垒的交易
                                gameState = GAME_STATE.TRADE_SUMMARY;  // 设置游戏状态为交易总结
                                break;  // 跳出switch语句
                            case 2:  // 如果玩家选择2
                                tradeAtFortStadaconaQuebec();  // 进行Stadacona Quebec堡垒的交易
                                gameState = GAME_STATE.TRADE_SUMMARY;  // 设置游戏状态为交易总结
                                break;  // 结束当前的 switch 语句块
                            case 3:  // 如果当前的 case 值为 3
                                // 检查玩家和队伍是否全部死亡
                                if (!tradeAtFortNewYork()) {
                                    gameState = GAME_STATE.GAME_OVER;  // 如果没有，将游戏状态设置为 GAME_OVER
                                } else {
                                    gameState = GAME_STATE.TRADE_SUMMARY;  // 如果是，将游戏状态设置为 TRADE_SUMMARY
                                }
                                break;  // 结束当前的 switch 语句块
                        }

                        break;  // 结束当前的 switch 语句块
                    }

                case TRADE_SUMMARY:  // 如果当前的游戏状态为 TRADE_SUMMARY

                    System.out.println();  // 打印一个空行
                    double beaverTotal = beaverPrice * pelts.get(BEAVER_ENTRY).getNumber();  // 计算海狸皮的总价值
                    System.out.print("YOUR BEAVER SOLD FOR $ " + formatNumber(beaverTotal));  // 打印玩家卖出海狸皮的总价值
                    // 计算狐狸皮的总价值
                    double foxTotal = foxPrice * pelts.get(FOX_ENTRY).getNumber();
                    // 打印狐狸皮的总价值
                    System.out.println(simulateTabs(5) + "YOUR FOX SOLD FOR $ " + formatNumber(foxTotal));

                    // 计算貂皮的总价值
                    double erMineTotal = erminePrice * pelts.get(ERMINE_ENTRY).getNumber();
                    // 打印貂皮的总价值
                    System.out.print("YOUR ERMINE SOLD FOR $ " + formatNumber(erMineTotal));

                    // 计算水貂皮的总价值
                    double minkTotal = minkPrice * pelts.get(MINK_ENTRY).getNumber();
                    // 打印水貂皮的总价值
                    System.out.println(simulateTabs(5) + "YOUR MINK SOLD FOR $ " + formatNumber(minkTotal));

                    // 计算总的收益并加到存款中
                    savings += beaverTotal + foxTotal + erMineTotal + minkTotal;
                    // 打印总的存款金额
                    System.out.println();
                    System.out.println("YOU NOW HAVE $" + formatNumber(savings) + " INCLUDING YOUR PREVIOUS SAVINGS");

                    // 设置游戏状态为再次交易
                    gameState = GAME_STATE.TRADE_AGAIN;
                    break;

                case TRADE_AGAIN:
                    // 如果玩家选择再次交易，则设置游戏状态为开始
                    if (yesEntered(displayTextAndGetInput("DO YOU WANT TO TRADE FURS NEXT YEAR? "))) {
                        gameState = GAME_STATE.STARTUP;
                    } else {
                        gameState = GAME_STATE.GAME_OVER;  // 设置游戏状态为游戏结束
                    }

            }
        } while (gameState != GAME_STATE.GAME_OVER);  // 当游戏状态不是游戏结束时，继续循环

    }

    /**
     * Create all pelt types with a count of zero
     *
     * @return Arraylist of initialised Pelt objects.
     */
    private ArrayList<Pelt> initPelts() {

        ArrayList<Pelt> tempPelts = new ArrayList<>();  // 创建一个空的 Pelt 对象列表
        tempPelts.add(new Pelt(MINK, 0));  // 向列表中添加一个类型为 MINK，数量为 0 的 Pelt 对象
        tempPelts.add(new Pelt(BEAVER, 0));  // 向列表中添加一个类型为 BEAVER，数量为 0 的 Pelt 对象
        tempPelts.add(new Pelt(ERMINE, 0));  // 向列表中添加一个类型为 ERMINE，数量为 0 的 Pelt 对象
        tempPelts.add(new Pelt(FOX, 0));  // 向列表中添加一个类型为 FOX，数量为 0 的 Pelt 对象
        return tempPelts;  // 返回初始化后的 Pelt 对象列表
    }

    /**
     * Display a message about trading at each fort and confirm if the player wants to trade
     * at ANOTHER fort
     *
     * @param fort the fort in question
     * @return true if YES was typed by player
     */
    private boolean confirmFort(int fort) {
        switch (fort) {
            case FORT_HOCHELAGA_MONTREAL:
                // 显示关于在每个要塞交易的消息
                System.out.println("YOU HAVE CHOSEN THE EASIEST ROUTE.  HOWEVER, THE FORT");
                System.out.println("IS FAR FROM ANY SEAPORT.  THE VALUE");
                System.out.println("YOU RECEIVE FOR YOUR FURS WILL BE LOW AND THE COST");
                System.out.println("OF SUPPLIES HIGHER THAN AT FORTS STADACONA OR NEW YORK.");
                break;
            case FORT_STADACONA_QUEBEC:
                // 显示关于在每个要塞交易的消息
                System.out.println("YOU HAVE CHOSEN A HARD ROUTE.  IT IS, IN COMPARSION,");
                System.out.println("HARDER THAN THE ROUTE TO HOCHELAGA BUT EASIER THAN");
                System.out.println("THE ROUTE TO NEW YORK.  YOU WILL RECEIVE AN AVERAGE VALUE");
                // 打印到纽约的路线信息，提示平均价值
                System.out.println("FOR YOUR FURS AND THE COST OF YOUR SUPPLIES WILL BE AVERAGE.");
                // 打印毛皮的价值和供应品的成本将是平均值
                break;
            case FORT_NEW_YORK:
                System.out.println("YOU HAVE CHOSEN THE MOST DIFFICULT ROUTE.  AT");
                // 打印选择了最困难的路线
                System.out.println("FORT NEW YORK YOU WILL RECEIVE THE HIGHEST VALUE");
                // 打印在纽约堡垒将获得最高价值
                System.out.println("FOR YOUR FURS.  THE COST OF YOUR SUPPLIES");
                // 打印毛皮的价值和供应品的成本
                System.out.println("WILL BE LOWER THAN AT ALL THE OTHER FORTS.");
                // 打印将低于所有其他堡垒的成本
                break;
        }

        System.out.println("DO YOU WANT TO TRADE AT ANOTHER FORT?");
        // 打印询问是否想在另一个堡垒交易
        return yesEntered(displayTextAndGetInput("ANSWER YES OR NO "));
        // 返回用户输入的是否想要交易的答案

    }

    /**
     * Trade at the safest fort - Fort Hochelaga
     * No chance of anything bad happening, so just calculate amount per pelt
     * and return
     */
    private void tradeAtFortHochelagaMontreal() {
        savings -= 160.0; // 减去160.0的费用
        System.out.println();
        System.out.println("SUPPLIES AT FORT HOCHELAGA COST $150.00."); // 打印输出费用信息
        System.out.println("YOUR TRAVEL EXPENSES TO HOCHELAGA WERE $10.00."); // 打印输出旅行费用信息
        minkPrice = ((.2 * Math.random() + .7) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2); // 计算价格
        erminePrice = ((.2 * Math.random() + .65) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2); // 计算价格
        beaverPrice = ((.2 * Math.random() + .75) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2); // 计算价格
        foxPrice = ((.2 * Math.random() + .8) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2); // 计算价格
    }


    private void tradeAtFortStadaconaQuebec() {
        savings -= 140.0; // 减去140.0的费用
        System.out.println();
        minkPrice = ((.2 * Math.random() + .85) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2); // 计算价格
        erminePrice = ((.2 * Math.random() + .8) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2); // 计算价格
        beaverPrice = ((.2 * Math.random() + .9) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2); // 计算价格
        // 生成一个1到10之间的随机数，表示旅行结果
        int tripResult = (int) (Math.random() * 10) + 1;
        // 如果旅行结果小于等于2
        if (tripResult <= 2) {
            // 在我们的ArrayList中找到海狸皮
            Pelt beaverPelt = pelts.get(BEAVER_ENTRY);
            // 海狸皮被偷了，所以更新为数量为零
            beaverPelt.lostPelts();
            // 在ArrayList中更新
            pelts.set(BEAVER_ENTRY, beaverPelt);
            System.out.println("YOUR BEAVER WERE TOO HEAVY TO CARRY ACROSS");
            System.out.println("THE PORTAGE.  YOU HAD TO LEAVE THE PELTS, BUT FOUND");
            System.out.println("THEM STOLEN WHEN YOU RETURNED.");
        } else if (tripResult <= 6) {
            System.out.println("YOU ARRIVED SAFELY AT FORT STADACONA.");
        } else if (tripResult <= 8) {
            System.out.println("YOUR CANOE UPSET IN THE LACHINE RAPIDS.  YOU");
            System.out.println("LOST ALL YOUR FURS.");
            // 清空所有皮毛
            resetPelts();
        } else if (tripResult <= 10) {
            // 输出狐狸皮未经过处理
            System.out.println("你的狐狸皮没有经过适当的处理。");
            System.out.println("没有人会购买它们。");
            // 由于原始基本程序中未计算狐狸毛皮，因此存在错误
            // 在我们的ArrayList中找到海狸皮
            Pelt foxPelt = pelts.get(FOX_ENTRY);
            // 皮毛被偷了，所以更新为零
            foxPelt.lostPelts();
            // 在ArrayList中更新它
            pelts.set(FOX_ENTRY, foxPelt);
        }

        System.out.println("在Stadacona堡垒的供应品费用为$125.00。");
        System.out.println("你前往Stadacona的旅行费用为$15.00。");
    }

    private boolean tradeAtFortNewYork() {

        boolean playerAlive = true;
        savings -= 105.0;
        // 输出空行
        System.out.println();
        // 计算獾皮的价格
        minkPrice = ((.2 * Math.random() + 1.05) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2);
        // 计算狐狸皮的价格
        foxPrice = ((.2 * Math.random() + 1.1) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2);

        // 在前往堡垒的旅途中发生了什么？
        int tripResult = (int) (Math.random() * 10) + 1;
        if (tripResult <= 2) {
            // 被易洛魁部落袭击
            System.out.println("YOU WERE ATTACKED BY A PARTY OF IROQUOIS.");
            System.out.println("ALL PEOPLE IN YOUR TRADING GROUP WERE");
            System.out.println("KILLED.  THIS ENDS THE GAME.");
            playerAlive = false;
        } else if (tripResult <= 6) {
            // 幸运抵达纽约堡
            System.out.println("YOU WERE LUCKY.  YOU ARRIVED SAFELY");
            System.out.println("AT FORT NEW YORK.");
        } else if (tripResult <= 8) {
            // 险遭易洛魁部落袭击，但不得不放弃所有毛皮
            System.out.println("YOU NARROWLY ESCAPED AN IROQUOIS RAIDING PARTY.");
            System.out.println("HOWEVER, YOU HAD TO LEAVE ALL YOUR FURS BEHIND.");
            // 清空所有毛皮
            resetPelts();
        } else if (tripResult <= 10) {
            beaverPrice /= 2;  // 将beaverPrice除以2，相当于将其减半
            minkPrice /= 2;  // 将minkPrice除以2，相当于将其减半
            System.out.println("YOUR MINK AND BEAVER WERE DAMAGED ON YOUR TRIP.");  // 打印旅行中水貂和海狸皮毛受损的消息
            System.out.println("YOU RECEIVE ONLY HALF THE CURRENT PRICE FOR THESE FURS.");  // 打印只能获得这些毛皮当前价格的一半的消息
        }

        if (playerAlive) {  // 如果玩家还活着
            System.out.println("SUPPLIES AT NEW YORK COST $80.00.");  // 打印纽约的供应品价格为$80.00
            System.out.println("YOUR TRAVEL EXPENSES TO NEW YORK WERE $25.00.");  // 打印你前往纽约的旅行费用为$25.00
        }

        return playerAlive;  // 返回玩家是否还活着的布尔值
    }

    /**
     * Reset pelt count for all Pelt types to zero.
     */
    private void resetPelts() {  // 重置所有毛皮类型的毛皮计数为零
        for (int i = 0; i < pelts.size(); i++) {  // 遍历毛皮列表
            Pelt pelt = pelts.get(i);  // 获取当前索引处的毛皮对象
            pelt.lostPelts();  # 调用pelt对象的lostPelts方法
            pelts.set(i, pelt);  # 将pelt对象设置到pelts列表的第i个位置
        }
    }

    /**
     * 返回一个包含用户输入的皮毛数量的pelt对象。
     *
     * @param peltName 皮毛的名称（类型）
     * @return 玩家分配的皮毛数量
     */
    private int getPeltCount(String peltName) {
        return displayTextAndGetNumber("HOW MANY " + peltName + " PELTS DO YOU HAVE? ");  # 显示提示信息并获取玩家输入的数字
    }

    private void extendedTradingInfo() {
        System.out.println("YOU MAY TRADE YOUR FURS AT FORT 1, FORT 2,");  # 打印交易信息
        System.out.println("OR FORT 3.  FORT 1 IS FORT HOCHELAGA (MONTREAL)");  # 打印交易信息
        System.out.println("AND IS UNDER THE PROTECTION OF THE FRENCH ARMY.");  # 打印交易信息
        System.out.println("FORT 2 IS FORT STADACONA (QUEBEC) AND IS UNDER THE");  # 打印交易信息
    // 输出游戏开始消息
    private void gameStartupMessage() {
        System.out.println(simulateTabs(31) + "FUR TRADER");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
    }

    // 输出游戏介绍
    private void intro() {
        System.out.println("YOU ARE THE LEADER OF A FRENCH FUR TRADING EXPEDITION IN ");
        System.out.println("1776 LEAVING THE LAKE ONTARIO AREA TO SELL FURS AND GET");
        System.out.println("SUPPLIES FOR THE NEXT YEAR.  YOU HAVE A CHOICE OF THREE");
        System.out.println("FORTS AT WHICH YOU MAY TRADE.  THE COST OF SUPPLIES");
        System.out.println("AND THE AMOUNT YOU RECEIVE FOR YOUR FURS WILL DEPEND");
    }
        System.out.println("ON THE FORT THAT YOU CHOOSE.");  // 在屏幕上打印消息
        System.out.println();  // 打印空行
    }

    /**
     * Format a double number to two decimal points for output.
     *
     * @param number double number
     * @return formatted number as a string
     */
    private String formatNumber(double number) {
        return String.format("%.2f", number);  // 将 double 数字格式化为两位小数点的字符串
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     * Converts input to an Integer
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
    */
    # 根据显示的文本获取用户输入的数字
    private int displayTextAndGetNumber(String text) {
        return Integer.parseInt(displayTextAndGetInput(text));
    }

    /*
     * 在屏幕上打印消息，然后从键盘接受输入。
     *
     * @param text 要在屏幕上显示的消息。
     * @return 玩家输入的内容。
     */
    # 在屏幕上显示消息并获取用户输入
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }

    /**
     * 模拟旧的基本语言中的tab(xx)命令，该命令将文本缩进xx个空格。
     *
     * @param spaces 需要缩进的空格数
    * @param text the string to be checked
    * @param values variable number of values to compare with
    * @return true if the string equals one of the values, otherwise false
    */
    private boolean stringIsAnyValue(String text, String... values) {
        for (String value : values) {
            if (text.equalsIgnoreCase(value)) {
                return true;
            }
        }
        return false;
    }
# 定义一个私有方法，用于检查源字符串是否等于传入的一系列字符串中的任意一个
# 比较时不区分大小写
# @param text   源字符串
# @param values 用于比较的一系列字符串
# @return 如果源字符串等于传入的任意一个字符串，则返回true
private boolean stringIsAnyValue(String text, String... values) {
    # 使用流处理传入的字符串数组，检查是否有任意一个字符串与源字符串相等（忽略大小写）
    return Arrays.stream(values).anyMatch(str -> str.equalsIgnoreCase(text));
}
```