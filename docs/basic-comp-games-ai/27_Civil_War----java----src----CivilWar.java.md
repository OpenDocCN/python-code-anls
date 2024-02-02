# `basic-computer-games\27_Civil_War\java\src\CivilWar.java`

```py
// 导入所需的类
import java.io.PrintStream;
import java.util.InputMismatchException;
import java.util.List;
import java.util.Scanner;
import java.util.function.Function;
import java.util.function.Predicate;
// 导入静态方法，用于将流中的元素连接成字符串
import static java.util.stream.Collectors.joining;
// 导入静态方法，用于生成指定范围的整数流
import static java.util.stream.IntStream.range;

// 禁止拼写检查
@SuppressWarnings("SpellCheckingInspection")
public class CivilWar {

    // 输出流
    private final PrintStream out;
    // 历史数据列表
    private final List<HistoricalDatum> data;

    // 战斗结果
    private final BattleResults results;

    // 当前战斗状态
    private BattleState currentBattle;
    // 将军数量
    private int numGenerals;
    // 战斗编号
    private int battleNumber;
    // 是否需要战斗描述
    private boolean wantBattleDescriptions;
    // 战略数组
    private final int[] strategies;

    // 南方战略
    private int confedStrategy;
    // 北方战略
    private int unionStrategy;

    // 资源
    private final ArmyPair<ArmyResources> resources;
    // 总预期伤亡
    private final ArmyPair<Integer> totalExpectedCasualties;
    // 总伤亡
    private final ArmyPair<Integer> totalCasualties;
    // 收入
    private final ArmyPair<Integer> revenue;
    // 通货膨胀
    private final ArmyPair<Integer> inflation;
    // 总支出
    private final ArmyPair<Integer> totalExpenditure;
    // 总军队数量
    private final ArmyPair<Integer> totalTroops;

    // 南方伤亡过多
    private boolean excessiveConfederateLosses;
    // 北方伤亡过多
    private boolean excessiveUnionLosses;

    // 南方投降
    private boolean confedSurrender;
    // 北方投降
    private boolean unionSurrender;

    // 是/否提醒信息
    private final static String YES_NO_REMINDER = "(ANSWER YES OR NO)";
    // 是/否检查器
    private final static Predicate<String> YES_NO_CHECKER = i -> isYes(i) || isNo(i);

    /**
     * ORIGINAL GAME DESIGN: CRAM, GOODIE, HIBBARD LEXINGTON H.S.
     * MODIFICATIONS: G. PAUL, R. HESS (TIES), 1973
     */
    // 主方法
    public static void main(String[] args) {
        // 创建 CivilWar 对象
        var x = new CivilWar(System.out);
        // 显示游戏制作人员信息
        x.showCredits();

        // LET D=RND(-1) ???

        // 提示是否需要游戏说明
        System.out.print("DO YOU WANT INSTRUCTIONS? ");
        // 如果需要，显示游戏说明
        if (isYes(inputString(YES_NO_CHECKER, YES_NO_REMINDER))) {
            x.showHelp();
        }
        // 进入游戏循环
        x.gameLoop();
    }
}
    // 游戏循环函数
    private void gameLoop() {
        out.println();  // 输出空行
        out.println();  // 输出空行
        out.println();  // 输出空行
        out.print("ARE THERE TWO GENERALS PRESENT (ANSWER YES OR NO)? ");  // 提示用户输入是否有两位将军

        // 判断用户输入是否为“YES”，如果是则设置将军数量为2，否则设置为1，并输出相关信息
        if (isYes(inputString(YES_NO_CHECKER, YES_NO_REMINDER))) {
            this.numGenerals = 2;
        } else {
            this.numGenerals = 1;
            out.println();  // 输出空行
            out.println("YOU ARE THE CONFEDERACY.   GOOD LUCK!");  // 输出提示信息
            out.println();  // 输出空行
        }

        // 输出选择战斗的提示信息
        out.println("SELECT A BATTLE BY TYPING A NUMBER FROM 1 TO 14 ON");
        out.println("REQUEST.  TYPE ANY OTHER NUMBER TO END THE SIMULATION.");
        out.println("BUT '0' BRINGS BACK EXACT PREVIOUS BATTLE SITUATION");
        out.println("ALLOWING YOU TO REPLAY IT");
        out.println();  // 输出空行
        out.println("NOTE: A NEGATIVE FOOD$ ENTRY CAUSES THE PROGRAM TO ");
        out.println("USE THE ENTRIES FROM THE PREVIOUS BATTLE");
        out.println();  // 输出空行

        // 提示用户是否需要战斗描述
        out.print("AFTER REQUESTING A BATTLE, DO YOU WISH BATTLE DESCRIPTIONS (ANSWER YES OR NO)? ");
        this.wantBattleDescriptions = isYes(inputString(YES_NO_CHECKER, YES_NO_REMINDER);  // 根据用户输入设置是否需要战斗描述

        // 游戏循环
        while (true) {
            var battle = startBattle();  // 开始战斗
            if (battle == null) {  // 如果战斗为空，则跳出循环
                break;
            }

            this.currentBattle = battle;  // 设置当前战斗

            offensiveLogic(battle.data);  // 进攻逻辑

            calcLosses(battle);  // 计算损失

            reset();  // 重置

            // 如果南方投降，则输出相关信息；如果北方投降，则输出相关信息
            if (this.confedSurrender) {
                out.println("THE CONFEDERACY HAS SURRENDERED");
            } else if (unionSurrender) {  // FIXME Is this actually possible? 2850
                out.println("THE UNION HAS SURRENDERED.");
            }
        }

        complete();  // 完成
    }

    // 获取北方军队数量
    private int getUnionTroops(HistoricalDatum battle) {
        return (int) Math.floor(battle.troops.union * (1 + (totalExpectedCasualties.union - totalCasualties.union) / (totalTroops.union + 1.0)));  // 计算北方军队数量
    }
    // 根据历史数据中的战斗信息计算联盟军队的数量
    private int getConfedTroops(HistoricalDatum battle) {
        return (int) Math.floor(battle.troops.confederate * (1 + (totalExpectedCasualties.confederate - totalCasualties.confederate) / (totalTroops.confederate + 1.0)));
    }

    // 根据战斗状态和军队索引计算军队的士气
    private String moraleForArmy(BattleState battleState, int armyIdx) {
        var builder = new StringBuilder();

        ArmyResources currentResources;

        // 根据将军数量和军队索引确定当前资源和输出信息
        if (this.numGenerals == 1 || armyIdx == 0) {
            builder.append("CONFEDERATE ");
            currentResources = resources.confederate;
        } else {
            builder.append("UNION ");
            currentResources = resources.union;
        }

        // 计算士气值
        currentResources.morale = (2 * Math.pow(currentResources.food, 2) + Math.pow(currentResources.salaries, 2)) / Math.pow(battleState.F1, 2) + 1;
        // 根据士气值输出相应的士气状态
        if (currentResources.morale >= 10) {
            builder.append("MORALE IS HIGH");
        } else if (currentResources.morale >= 5) {
            builder.append("MORALE IS FAIR");
        } else {
            builder.append("MORALE IS POOR");
        }

        return builder.toString();
    }

    // 定义进攻状态的枚举类型
    private enum OffensiveStatus {
        DEFENSIVE("YOU ARE ON THE DEFENSIVE"), OFFENSIVE("YOU ARE ON THE OFFENSIVE"), BOTH_OFFENSIVE("BOTH SIDES ARE ON THE OFFENSIVE");

        private final String label;

        // 枚举类型的构造函数
        OffensiveStatus(String label) {
            this.label = label;
        }
    }
    // 执行进攻逻辑，接受历史数据作为参数
    private void offensiveLogic(HistoricalDatum battle) {
        // 输出提示信息
        out.print("CONFEDERATE GENERAL---");
        // 输出进攻状态标签
        out.println(battle.offensiveStatus.label);

        // 选择策略

        // 如果将军数量为2
        if (numGenerals == 2) {
            // 输出提示信息
            out.print("CONFEDERATE STRATEGY ? ");
        } else {
            // 输出提示信息
            out.print("YOUR STRATEGY ? ");
        }

        // 输入并设置南方策略
        confedStrategy = inputInt(i -> i >= 1 && i <= 5, i -> "STRATEGY " + i + " NOT ALLOWED.");
        // 如果南方策略为5
        if (confedStrategy == 5) {  // 1970
            // 设置南方投降标志为true
            confedSurrender = true;
        }

        // 如果将军数量为2
        if (numGenerals == 2) {
            // 输出提示信息
            out.print("UNION STRATEGY ? ");

            // 输入并设置北方策略
            unionStrategy = inputInt(i -> i >= 1 && i <= 5, i -> "STRATEGY " + i + " NOT ALLOWED.");
            // 如果北方策略为5
            if (unionStrategy == 5) {  // 1970
                // 设置北方投降标志为true
                unionSurrender = true;
            }
        } else {
            // 调用unionStrategy方法
            unionStrategy();
        }
    }

    // 2070  REM : SIMULATED LOSSES-NORTH
    // 模拟北方损失
    private UnionLosses simulateUnionLosses(HistoricalDatum battle) {
        // 计算损失
        var losses = (2.0 * battle.expectedCasualties.union / 5) * (1 + 1.0 / (2 * (Math.abs(unionStrategy - confedStrategy) + 1)));
        losses = losses * (1.28 + (5.0 * battle.troops.union / 6) / (resources.union.ammunition + 1));
        losses = Math.floor(losses * (1 + 1 / resources.union.morale) + 0.5);
        // 如果损失大于现有人数，重新计算损失
        var moraleFactor = 100 / resources.union.morale;

        if (Math.floor(losses + moraleFactor) >= getUnionTroops(battle)) {
            losses = Math.floor(13.0 * getUnionTroops(battle) / 20);
            moraleFactor = 7 * losses / 13;
            // 设置过多的北方损失标志为true
            excessiveUnionLosses = true;
        }

        // 返回模拟损失
        return new UnionLosses((int) losses, (int) Math.floor(moraleFactor));
    }

    // 2170: CALCULATE SIMULATED LOSSES
    }

    // 2790
    // 重置游戏状态
    private void reset() {
        // 重置过多的南方损失和过多的北方损失标志为false
        excessiveConfederateLosses = excessiveUnionLosses = false;

        // 输出分隔线
        out.println("---------------");
    }
    // 完成游戏，输出结果
    private void complete() {
        out.println();
        out.println();
        out.println();
        out.println();
        out.println();
        out.println();
        out.println("THE CONFEDERACY HAS WON " + results.confederate + " BATTLES AND LOST " + results.union);

        // 如果联盟策略为5，则输出南方胜利
        if (this.unionStrategy == 5) {
            out.println("THE CONFEDERACY HAS WON THE WAR");
        }

        // 如果南方策略为5或南方战败次数小于等于北方，则输出北方胜利
        if (this.confedStrategy == 5 || results.confederate <= results.union) {
            out.println("THE UNION HAS WON THE WAR");
        }

        out.println();

        // 输出战斗结果统计
        out.println("FOR THE " + results.getTotal() + " BATTLES FOUGHT (EXCLUDING RERUNS)");
        out.println("                       CONFEDERACY    UNION");
        out.println("HISTORICAL LOSSES      " + (int) Math.floor(totalExpectedCasualties.confederate + .5) + "          " + (int) Math.floor(totalExpectedCasualties.union + .5));
        out.println("SIMULATED LOSSES       " + (int) Math.floor(totalCasualties.confederate + .5) + "          " + (int) Math.floor(totalCasualties.union + .5));
        out.println();
        out.println("    % OF ORIGINAL      " + (int) Math.floor(100 * ((double) totalCasualties.confederate / totalExpectedCasualties.confederate) + .5) + "             " + (int) Math.floor(100 * ((double) totalCasualties.union / totalExpectedCasualties.union) + .5));

        // 如果将军数量为1，则输出南方使用策略的百分比
        if (this.numGenerals == 1) {
            out.println();
            out.println("UNION INTELLIGENCE SUGGESTS THAT THE SOUTH USED ");
            out.println("STRATEGIES 1, 2, 3, 4 IN THE FOLLOWING PERCENTAGES");
            out.println(this.strategies[0] + "," + this.strategies[1] + "," + this.strategies[2] + "," + this.strategies[3]);
        }
    }
    // 根据联盟和联邦的损失情况，确定胜利者
    private Winner findWinner(double confLosses, double unionLosses) {
        // 如果联盟和联邦都有过多损失，则判定为不确定胜利者
        if (this.excessiveConfederateLosses && this.excessiveUnionLosses) {
            return Winner.INDECISIVE;
        }

        // 如果联盟有过多损失，则判定为联邦胜利
        if (this.excessiveConfederateLosses) {
            return Winner.UNION;
        }

        // 如果联邦有过多损失或者联邦的损失少于联盟，则判定为联盟胜利
        if (this.excessiveUnionLosses || confLosses < unionLosses) {
            return Winner.CONFED;
        }

        // 如果联盟和联邦的损失相等，则判定为不确定胜利者
        if (confLosses == unionLosses) {
            return Winner.INDECISIVE;
        }

        // 默认情况下判定为联邦胜利
        return Winner.UNION;  // FIXME Really? 2400-2420 ?
    }

    // 枚举类型，表示胜利者
    private enum Winner {
        CONFED, UNION, INDECISIVE
    }

    // 联盟策略
    private void unionStrategy() {
        // 如果战斗次数为0，则需要输入联盟策略
        if (this.battleNumber == 0) {
            out.print("UNION STRATEGY ? ");
            var terminalInput = new Scanner(System.in);
            unionStrategy = terminalInput.nextInt();
            // 如果输入小于0，则提示重新输入
            if (unionStrategy < 0) {
                out.println("ENTER 1, 2, 3, OR 4 (USUALLY PREVIOUS UNION STRATEGY)");
                // FIXME Retry Y2 input !!!
            }

            // 如果输入小于5，则返回
            if (unionStrategy < 5) {  // 3155
                return;
            }
        }

        var S0 = 0;
        var r = 100 * Math.random();

        // 根据随机数和策略权重确定联盟策略
        for (unionStrategy = 1; unionStrategy <= 4; unionStrategy++) {
            S0 += this.strategies[unionStrategy - 1];
            // 如果随机数小于权重之和，则确定联盟策略
            if (r < S0) {
                break;
            }
        }
        // 输出确定的联盟策略
        out.println("UNION STRATEGY IS " + unionStrategy);
    }

    // 显示游戏制作人员信息
    private void showCredits() {
        out.println(" ".repeat(26) + "CIVIL WAR");
        out.println(" ".repeat(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        out.println();
        out.println();
        out.println();
    }
    // 更新策略，根据给定的策略值进行更新
    private void updateStrategies(int strategy) {
        // REM LEARN  PRESENT STRATEGY, START FORGETTING OLD ONES
        // REM - PRESENT STRATEGY OF SOUTH GAINS 3*S, OTHERS LOSE S
        // REM   PROBABILITY POINTS, UNLESS A STRATEGY FALLS BELOW 5%.

        // 定义变量 S 的值为 3，S0 的初始值为 0
        var S = 3;
        var S0 = 0;
        // 遍历策略数组
        for (int i = 0; i < 4; i++) {
            // 如果当前策略值小于等于 5，则跳过本次循环
            if (this.strategies[i] <= 5) {
                continue;
            }
            // 更新策略值
            this.strategies[i] -= S;
            // 累加 S0 的值
            S0 += S;
        }
        // 更新给定策略的值为 S0
        this.strategies[strategy - 1] += S0;
    }

    // 显示游戏帮助信息
    private void showHelp() {
        out.println();
        out.println();
        out.println();
        out.println();
        out.println("THIS IS A CIVIL WAR SIMULATION.");
        out.println("TO PLAY TYPE A RESPONSE WHEN THE COMPUTER ASKS.");
        out.println("REMEMBER THAT ALL FACTORS ARE INTERRELATED AND THAT YOUR");
        out.println("RESPONSES COULD CHANGE HISTORY. FACTS AND FIGURES USED ARE");
        out.println("BASED ON THE ACTUAL OCCURRENCE. MOST BATTLES TEND TO RESULT");
        out.println("AS THEY DID IN THE CIVIL WAR, BUT IT ALL DEPENDS ON YOU!!");
        out.println();
        out.println("THE OBJECT OF THE GAME IS TO WIN AS MANY BATTLES AS ");
        out.println("POSSIBLE.");
        out.println();
        out.println("YOUR CHOICES FOR DEFENSIVE STRATEGY ARE:");
        out.println("        (1) ARTILLERY ATTACK");
        out.println("        (2) FORTIFICATION AGAINST FRONTAL ATTACK");
        out.println("        (3) FORTIFICATION AGAINST FLANKING MANEUVERS");
        out.println("        (4) FALLING BACK");
        out.println(" YOUR CHOICES FOR OFFENSIVE STRATEGY ARE:");
        out.println("        (1) ARTILLERY ATTACK");
        out.println("        (2) FRONTAL ATTACK");
        out.println("        (3) FLANKING MANEUVERS");
        out.println("        (4) ENCIRCLEMENT");
        out.println("YOU MAY SURRENDER BY TYPING A '5' FOR YOUR STRATEGY.");
    }
    // 定义一个常量，表示整数的最大长度为6
    private static final int MAX_NUM_LENGTH = 6;

    // 将整数右对齐，并返回对齐后的字符串
    private String rightAlignInt(int number) {
        // 将整数转换为字符串
        var s = String.valueOf(number);
        // 返回右对齐后的字符串
        return " ".repeat(MAX_NUM_LENGTH - s.length()) + s;
    }

    // 将浮点数转换为整数后右对齐，并返回对齐后的字符串
    private String rightAlignInt(double number) {
        // 调用 rightAlignInt 方法将整数右对齐
        return rightAlignInt((int) Math.floor(number));
    }

    // 从控制台输入字符串，并进行验证，如果验证通过则返回输入的字符串，否则提示重新输入
    private static String inputString(Predicate<String> validator, String reminder) {
        while (true) {
            try {
                // 从控制台输入字符串
                var input = new Scanner(System.in).nextLine();
                // 验证输入的字符串是否符合要求
                if (validator.test(input)) {
                    return input;
                }
            } catch (InputMismatchException e) {
                // 捕获输入类型不匹配的异常，忽略并继续循环
                // Ignore
            }
            // 打印提示信息
            System.out.println(reminder);
        }
    }

    // 从控制台输入整数，并进行验证，如果验证通过则返回输入的整数，否则提示重新输入
    private static int inputInt(Predicate<Integer> validator, Function<Integer, String> reminder) {
        while (true) {
            try {
                // 从控制台输入整数
                var input = new Scanner(System.in).nextInt();
                // 验证输入的整数是否符合要求
                if (validator.test(input)) {
                    return input;
                }
                // 打印提示信息
                System.out.println(reminder.apply(input));
            } catch (InputMismatchException e) {
                // 捕获输入类型不匹配的异常，打印提示信息并继续循环
                System.out.println(reminder.apply(0));
            }
        }
    }

    // 判断输入的字符串是否为"Y"或"YES"，不区分大小写
    private static boolean isYes(String s) {
        // 如果输入字符串为空，则返回false
        if (s == null) {
            return false;
        }
        // 将输入字符串转换为大写
        var uppercase = s.toUpperCase();
        // 判断是否为"Y"或"YES"，不区分大小写
        return uppercase.equals("Y") || uppercase.equals("YES");
    }

    // 判断输入的字符串是否为"N"或"NO"，不区分大小写
    private static boolean isNo(String s) {
        // 如果输入字符串为空，则返回false
        if (s == null) {
            return false;
        }
        // 将输入字符串转换为大写
        var uppercase = s.toUpperCase();
        // 判断是否为"N"或"NO"，不区分大小写
        return uppercase.equals("N") || uppercase.equals("NO");
    }

    // 定义一个内部类 BattleState，包含一个 HistoricalDatum 对象和一个 double 类型的 F1 属性
    private static class BattleState {
        private final HistoricalDatum data;
        private double F1;

        // 构造方法，接受一个 HistoricalDatum 对象作为参数
        public BattleState(HistoricalDatum data) {
            this.data = data;
        }
    }
    # 定义一个泛型类，表示两个军队的对应关系
    private static class ArmyPair<T> {
        # 内部私有变量，表示南方联盟军队和北方联邦军队
        private T confederate;
        private T union;

        # 构造函数，初始化南方联盟军队和北方联邦军队
        public ArmyPair(T confederate, T union) {
            this.confederate = confederate;
            this.union = union;
        }
    }

    # 定义战斗结果类
    private static class BattleResults {
        # 内部私有变量，表示南方联盟军队、北方联邦军队和不确定的战斗结果
        private int confederate;
        private int union;
        private int indeterminate;

        # 获取总战斗结果
        public int getTotal() {
            return confederate + union + indeterminate;
        }
    }

    # 定义军队资源类
    private static class ArmyResources {
        # 内部私有变量，表示食物、工资、弹药和预算
        private int food;
        private int salaries;
        private int ammunition;
        private int budget;

        # TODO 是否真的需要在这里定义士气？
        private double morale;

        # 获取总资源
        public int getTotal() {
            return this.food + this.salaries + this.ammunition;
        }
    }

    # 定义历史数据记录类
    private record HistoricalDatum(String name, ArmyPair<Integer> troops,
                                   ArmyPair<Integer> expectedCasualties,
                                   OffensiveStatus offensiveStatus, String[] blurb) {
    }

    # 定义联邦军损失记录类
    private record UnionLosses(int losses, int desertions) {
    }
# 闭合前面的函数定义
```