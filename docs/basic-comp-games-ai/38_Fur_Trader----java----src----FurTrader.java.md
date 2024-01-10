# `basic-computer-games\38_Fur_Trader\java\src\FurTrader.java`

```
// 导入所需的类
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

/**
 * Fur Trader 游戏
 * <p>
 * 基于这里的基本Fur Trader游戏
 * https://github.com/coding-horror/basic-computer-games/blob/main/38%20Fur%20Trader/furtrader.bas
 * <p>
 * 注意：本意是在Java中创建1970年代Basic游戏的版本，没有引入新功能-没有添加额外的文本，错误检查等。
 */
public class FurTrader {

    // 初始储蓄金额
    public static final double START_SAVINGS_AMOUNT = 600.0;
    // 初始毛皮数量
    public static final int STARTING_FURS = 190;

    // 不同贸易点的标识
    public static final int FORT_HOCHELAGA_MONTREAL = 1;
    public static final int FORT_STADACONA_QUEBEC = 2;
    public static final int FORT_NEW_YORK = 3;

    // 不同类型毛皮的标识和名称
    public static final String MINK = "MINK";
    public static final int MINK_ENTRY = 0;
    public static final String BEAVER = "BEAVER";
    public static final int BEAVER_ENTRY = 1;
    public static final String ERMINE = "ERMINE";
    public static final int ERMINE_ENTRY = 2;
    public static final String FOX = "FOX";
    public static final int FOX_ENTRY = 3;

    // 用于键盘输入
    private final Scanner kbScanner;

    // 当前游戏状态
    private GAME_STATE gameState;

    // 当前储蓄金额和各种毛皮的价格
    private double savings;
    private double minkPrice;
    private double beaverPrice;
    private double erminePrice;
    private double foxPrice;

    // 毛皮列表
    private ArrayList<Pelt> pelts;

    // 游戏是否已经进行过一次
    private boolean playedOnce;

    // 构造函数，初始化键盘输入和游戏状态
    public FurTrader() {
        kbScanner = new Scanner(System.in);
        gameState = GAME_STATE.INIT;
        playedOnce = false;
    }

    /**
     * 主游戏循环
     */
    }

    /**
     * 创建所有类型的毛皮，并初始化数量为零
     *
     * @return 初始化后的Pelt对象的ArrayList
     */
    // 初始化皮毛列表，包括水貂、海狸、鼬和狐狸
    private ArrayList<Pelt> initPelts() {

        ArrayList<Pelt> tempPelts = new ArrayList<>();
        tempPelts.add(new Pelt(MINK, 0));
        tempPelts.add(new Pelt(BEAVER, 0));
        tempPelts.add(new Pelt(ERMINE, 0));
        tempPelts.add(new Pelt(FOX, 0));
        return tempPelts;
    }

    /**
     * 在每个要塞显示有关交易的消息，并确认玩家是否想在另一个要塞进行交易
     *
     * @param fort 要塞编号
     * @return 如果玩家输入了YES，则返回true
     */
    private boolean confirmFort(int fort) {
        switch (fort) {
            case FORT_HOCHELAGA_MONTREAL:
                System.out.println("YOU HAVE CHOSEN THE EASIEST ROUTE.  HOWEVER, THE FORT");
                System.out.println("IS FAR FROM ANY SEAPORT.  THE VALUE");
                System.out.println("YOU RECEIVE FOR YOUR FURS WILL BE LOW AND THE COST");
                System.out.println("OF SUPPLIES HIGHER THAN AT FORTS STADACONA OR NEW YORK.");
                break;
            case FORT_STADACONA_QUEBEC:
                System.out.println("YOU HAVE CHOSEN A HARD ROUTE.  IT IS, IN COMPARSION,");
                System.out.println("HARDER THAN THE ROUTE TO HOCHELAGA BUT EASIER THAN");
                System.out.println("THE ROUTE TO NEW YORK.  YOU WILL RECEIVE AN AVERAGE VALUE");
                System.out.println("FOR YOUR FURS AND THE COST OF YOUR SUPPLIES WILL BE AVERAGE.");
                break;
            case FORT_NEW_YORK:
                System.out.println("YOU HAVE CHOSEN THE MOST DIFFICULT ROUTE.  AT");
                System.out.println("FORT NEW YORK YOU WILL RECEIVE THE HIGHEST VALUE");
                System.out.println("FOR YOUR FURS.  THE COST OF YOUR SUPPLIES");
                System.out.println("WILL BE LOWER THAN AT ALL THE OTHER FORTS.");
                break;
        }

        System.out.println("DO YOU WANT TO TRADE AT ANOTHER FORT?");
        return yesEntered(displayTextAndGetInput("ANSWER YES OR NO "));
    }
    /**
     * 在最安全的堡垒 - 霍谢拉加堡交易
     * 没有任何不好的事情发生的可能，所以只需计算每张皮毛的金额并返回
     */
    private void tradeAtFortHochelagaMontreal() {
        // 从储蓄中减去160.0
        savings -= 160.0;
        System.out.println();
        System.out.println("SUPPLIES AT FORT HOCHELAGA COST $150.00.");
        System.out.println("YOUR TRAVEL EXPENSES TO HOCHELAGA WERE $10.00.");
        // 计算水貂的价格
        minkPrice = ((.2 * Math.random() + .7) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2);
        // 计算貂皮的价格
        erminePrice = ((.2 * Math.random() + .65) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2);
        // 计算海狸皮的价格
        beaverPrice = ((.2 * Math.random() + .75) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2);
        // 计算狐狸皮的价格
        foxPrice = ((.2 * Math.random() + .8) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2);
    }
    // 在纽约堡交易
    private boolean tradeAtFortNewYork() {
        // 玩家是否存活
        boolean playerAlive = true;
        // 花费 105.0
        savings -= 105.0;
        // 输出空行
        System.out.println();
        // 计算水貂价格
        minkPrice = ((.2 * Math.random() + 1.05) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2);
        // 计算狐狸价格
        foxPrice = ((.2 * Math.random() + 1.1) * (Math.pow(10, 2) + .5)) / Math.pow(10, 2);

        // 行程结果
        int tripResult = (int) (Math.random() * 10) + 1;
        if (tripResult <= 2) {
            // 被伊罗quois部队袭击
            System.out.println("YOU WERE ATTACKED BY A PARTY OF IROQUOIS.");
            System.out.println("ALL PEOPLE IN YOUR TRADING GROUP WERE");
            System.out.println("KILLED.  THIS ENDS THE GAME.");
            playerAlive = false;
        } else if (tripResult <= 6) {
            // 幸运到达纽约堡
            System.out.println("YOU WERE LUCKY.  YOU ARRIVED SAFELY");
            System.out.println("AT FORT NEW YORK.");
        } else if (tripResult <= 8) {
            // 逃脱伊罗quois袭击，但是丢失所有毛皮
            System.out.println("YOU NARROWLY ESCAPED AN IROQUOIS RAIDING PARTY.");
            System.out.println("HOWEVER, YOU HAD TO LEAVE ALL YOUR FURS BEHIND.");
            // 清空所有毛皮
            resetPelts();
        } else if (tripResult <= 10) {
            // 水貂和海狸价格减半
            beaverPrice /= 2;
            minkPrice /= 2;
            System.out.println("YOUR MINK AND BEAVER WERE DAMAGED ON YOUR TRIP.");
            System.out.println("YOU RECEIVE ONLY HALF THE CURRENT PRICE FOR THESE FURS.");
        }

        if (playerAlive) {
            // 输出纽约堡的供应品价格和旅行费用
            System.out.println("SUPPLIES AT NEW YORK COST $80.00.");
            System.out.println("YOUR TRAVEL EXPENSES TO NEW YORK WERE $25.00.");
        }

        return playerAlive;
    }

    /**
     * 重置所有毛皮类型的毛皮数量为零
     */
    private void resetPelts() {
        for (int i = 0; i < pelts.size(); i++) {
            // 获取毛皮对象
            Pelt pelt = pelts.get(i);
            // 重置毛皮数量
            pelt.lostPelts();
            // 更新毛皮对象
            pelts.set(i, pelt);
        }
    }
    /**
     * 返回一个包含用户输入的皮毛数量的pelt对象。
     *
     * @param peltName 皮毛的名称（类型）
     * @return 玩家分配的皮毛数量
     */
    private int getPeltCount(String peltName) {
        return displayTextAndGetNumber("HOW MANY " + peltName + " PELTS DO YOU HAVE? ");
    }

    private void extendedTradingInfo() {
        System.out.println("YOU MAY TRADE YOUR FURS AT FORT 1, FORT 2,");
        System.out.println("OR FORT 3.  FORT 1 IS FORT HOCHELAGA (MONTREAL)");
        System.out.println("AND IS UNDER THE PROTECTION OF THE FRENCH ARMY.");
        System.out.println("FORT 2 IS FORT STADACONA (QUEBEC) AND IS UNDER THE");
        System.out.println("PROTECTION OF THE FRENCH ARMY.  HOWEVER, YOU MUST");
        System.out.println("MAKE A PORTAGE AND CROSS THE LACHINE RAPIDS.");
        System.out.println("FORT 3 IS FORT NEW YORK AND IS UNDER DUTCH CONTROL.");
        System.out.println("YOU MUST CROSS THROUGH IROQUOIS LAND.");
        System.out.println();

    }

    private void gameStartupMessage() {
        System.out.println(simulateTabs(31) + "FUR TRADER");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
    }

    private void intro() {
        System.out.println("YOU ARE THE LEADER OF A FRENCH FUR TRADING EXPEDITION IN ");
        System.out.println("1776 LEAVING THE LAKE ONTARIO AREA TO SELL FURS AND GET");
        System.out.println("SUPPLIES FOR THE NEXT YEAR.  YOU HAVE A CHOICE OF THREE");
        System.out.println("FORTS AT WHICH YOU MAY TRADE.  THE COST OF SUPPLIES");
        System.out.println("AND THE AMOUNT YOU RECEIVE FOR YOUR FURS WILL DEPEND");
        System.out.println("ON THE FORT THAT YOU CHOOSE.");
        System.out.println();
    }

    /**
     * 格式化一个双精度数，保留两位小数以便输出。
     *
     * @param number 双精度数
     * @return 格式化后的数字字符串
     */
    // 格式化数字为两位小数的字符串
    private String formatNumber(double number) {
        return String.format("%.2f", number);
    }

    /*
     * 在屏幕上打印消息，然后从键盘接受输入。
     * 将输入转换为整数
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private int displayTextAndGetNumber(String text) {
        return Integer.parseInt(displayTextAndGetInput(text));
    }

    /*
     * 在屏幕上打印消息，然后从键盘接受输入。
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }

    /**
     * 模拟旧的基本 tab(xx) 命令，将文本缩进 xx 个空格。
     *
     * @param spaces 需要的空格数
     * @return 包含指定空格数的字符串
     */
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }

    /**
     * 检查玩家是否输入了 Y 或 YES 作为答案。
     *
     * @param text 从键盘输入的字符串
     * @return 如果输入了 Y 或 YES，则返回 true，否则返回 false
     */
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");
    }

    /**
     * 检查字符串是否等于一系列变量数量的值
     * 用于检查是否输入了 Y 或 YES 等
     * 比较不区分大小写。
     *
     * @param text    源字符串
     * @param values  要与源字符串进行比较的一系列值
     * @return 如果在传递的一系列字符串中找到了匹配，则返回 true
     */
    private boolean stringIsAnyValue(String text, String... values) {
        return Arrays.stream(values).anyMatch(str -> str.equalsIgnoreCase(text));
    }
# 闭合前面的函数定义
```