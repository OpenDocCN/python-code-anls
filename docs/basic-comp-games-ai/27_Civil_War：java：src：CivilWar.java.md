# `d:/src/tocomm/basic-computer-games\27_Civil_War\java\src\CivilWar.java`

```
import java.io.PrintStream;  // 导入打印流类，用于输出结果
import java.util.InputMismatchException;  // 导入输入不匹配异常类，用于处理输入不匹配的情况
import java.util.List;  // 导入列表类，用于存储历史数据
import java.util.Scanner;  // 导入扫描器类，用于接收输入
import java.util.function.Function;  // 导入函数接口类，用于定义函数
import java.util.function.Predicate;  // 导入断言接口类，用于定义断言

import static java.util.stream.Collectors.joining;  // 静态导入流操作，用于将流中的元素连接成一个字符串
import static java.util.stream.IntStream.range;  // 静态导入流操作，用于生成指定范围的整数流

@SuppressWarnings("SpellCheckingInspection")  // 忽略拼写检查警告
public class CivilWar {

    private final PrintStream out;  // 声明打印流对象，用于输出结果
    private final List<HistoricalDatum> data;  // 声明历史数据列表，用于存储历史数据

    private final BattleResults results;  // 声明战斗结果对象，用于存储战斗结果

    private BattleState currentBattle;  // 声明当前战斗状态对象
    private int numGenerals;  // 声明将军数量变量
    private int battleNumber; // 保存战斗编号的变量
    private boolean wantBattleDescriptions; // 保存是否需要战斗描述的布尔变量
    private final int[] strategies; // 保存策略的整型数组

    private int confedStrategy; // 保存南方联盟的策略
    private int unionStrategy; // 保存北方联盟的策略

    private final ArmyPair<ArmyResources> resources; // 保存军队资源的军队对类
    private final ArmyPair<Integer> totalExpectedCasualties; // 保存总预期伤亡的军队对类
    private final ArmyPair<Integer> totalCasualties; // 保存总伤亡的军队对类
    private final ArmyPair<Integer> revenue; // 保存收入的军队对类
    private final ArmyPair<Integer> inflation; // 保存通货膨胀的军队对类
    private final ArmyPair<Integer> totalExpenditure; // 保存总支出的军队对类
    private final ArmyPair<Integer> totalTroops; // 保存总军队数量的军队对类

    private boolean excessiveConfederateLosses; // 保存南方联盟是否有过多损失的布尔变量
    private boolean excessiveUnionLosses; // 保存北方联盟是否有过多损失的布尔变量

    private boolean confedSurrender; // 保存南方联盟是否投降的布尔变量
    private boolean unionSurrender; // 保存北方联盟是否投降的布尔变量
    private final static String YES_NO_REMINDER = "(ANSWER YES OR NO)"; // 定义一个常量字符串，用于提醒用户输入“YES”或“NO”
    private final static Predicate<String> YES_NO_CHECKER = i -> isYes(i) || isNo(i); // 定义一个谓词，用于检查用户输入是否为“YES”或“NO”

    /**
     * ORIGINAL GAME DESIGN: CRAM, GOODIE, HIBBARD LEXINGTON H.S.
     * MODIFICATIONS: G. PAUL, R. HESS (TIES), 1973
     */
    public static void main(String[] args) {
        var x = new CivilWar(System.out); // 创建一个CivilWar对象，并传入System.out作为参数
        x.showCredits(); // 调用CivilWar对象的showCredits方法，显示游戏的制作人员信息

        // LET D=RND(-1) ???

        System.out.print("DO YOU WANT INSTRUCTIONS? "); // 打印提示信息，询问用户是否需要游戏说明

        if (isYes(inputString(YES_NO_CHECKER, YES_NO_REMINDER))) { // 调用inputString方法获取用户输入，并使用YES_NO_CHECKER和YES_NO_REMINDER进行验证和提示
            x.showHelp(); // 如果用户输入为“YES”，则调用CivilWar对象的showHelp方法，显示游戏说明
        }
    }
        x.gameLoop();
    }
```
这行代码调用了名为gameLoop的方法。

```
    private void gameLoop() {
```
这行代码定义了一个名为gameLoop的私有方法。

```
        out.println();
        out.println();
        out.println();
```
这三行代码分别打印了三个空行。

```
        out.print("ARE THERE TWO GENERALS PRESENT (ANSWER YES OR NO)? ");
```
这行代码打印了一个提示信息，要求用户输入是否有两位将军在场。

```
        if (isYes(inputString(YES_NO_CHECKER, YES_NO_REMINDER))) {
            this.numGenerals = 2;
        } else {
            this.numGenerals = 1;
            out.println();
            out.println("YOU ARE THE CONFEDERACY.   GOOD LUCK!");
            out.println();
        }
```
这段代码根据用户输入的是“YES”还是“NO”来确定将军的数量，并打印相应的信息。

```
        out.println("SELECT A BATTLE BY TYPING A NUMBER FROM 1 TO 14 ON");
        out.println("REQUEST.  TYPE ANY OTHER NUMBER TO END THE SIMULATION.");
```
这两行代码分别打印了选择战斗的提示信息。
        // 输出提示信息
        out.println("BUT '0' BRINGS BACK EXACT PREVIOUS BATTLE SITUATION");
        out.println("ALLOWING YOU TO REPLAY IT");
        out.println();
        out.println("NOTE: A NEGATIVE FOOD$ ENTRY CAUSES THE PROGRAM TO ");
        out.println("USE THE ENTRIES FROM THE PREVIOUS BATTLE");
        out.println();

        // 提示用户输入是否需要战斗描述
        out.print("AFTER REQUESTING A BATTLE, DO YOU WISH BATTLE DESCRIPTIONS (ANSWER YES OR NO)? ");

        // 根据用户输入设置是否需要战斗描述
        this.wantBattleDescriptions = isYes(inputString(YES_NO_CHECKER, YES_NO_REMINDER));

        // 循环进行战斗
        while (true) {
            // 开始一场战斗
            var battle = startBattle();
            // 如果没有战斗发生，跳出循环
            if (battle == null) {
                break;
            }

            // 设置当前战斗
            this.currentBattle = battle;

            // 进行进攻逻辑处理
            offensiveLogic(battle.data);
            calcLosses(battle);  // 调用calcLosses函数计算战斗损失

            reset();  // 调用reset函数重置状态

            if (this.confedSurrender) {  // 如果南方联盟投降
                out.println("THE CONFEDERACY HAS SURRENDERED");  // 输出南方联盟投降信息
            } else if (unionSurrender) {  // 如果北方联盟投降
                out.println("THE UNION HAS SURRENDERED.");  // 输出北方联盟投降信息
            }
        }

        complete();  // 完成战斗模拟
    }

    private BattleState startBattle() {  // 开始战斗函数
        out.println();  // 输出空行
        out.println();  // 输出空行
        out.println();  // 输出空行
        out.print("WHICH BATTLE DO YOU WISH TO SIMULATE ? ");  // 输出提示信息，询问要模拟哪场战斗
        // 从输入中获取战斗编号，要求大于等于1，或者等于0且当前战斗不为空
        var battleNumber = inputInt(i -> i >= 1 || (i == 0 && this.currentBattle != null), i -> "BATTLE " + i + " NOT ALLOWED.");

        // 如果战斗编号为0，打印当前战斗的名称并返回当前战斗
        if (battleNumber == 0) {
            out.println(this.currentBattle.data.name + " INSTANT REPLAY");
            return this.currentBattle;
        }

        // 如果战斗编号大于数据大小，返回null
        if (battleNumber > this.data.size()) {  // TYPE ANY OTHER NUMBER TO END THE SIMULATION
            return null;
        }

        // 将战斗编号赋值给this.battleNumber
        this.battleNumber = battleNumber;

        // 获取指定战斗编号对应的战斗数据
        var battle = this.data.get(this.battleNumber - 1);
        // 创建战斗状态对象
        var battleState = new BattleState(battle);

        // 将 excessiveConfederateLosses 设置为false
        excessiveConfederateLosses = false;
        // INFLATION CALC
        // 计算通货膨胀
        // REM - ONLY IN PRINTOUT IS CONFED INFLATION = I1+15%
        // 注意 - 仅在打印输出中，南方的通货膨胀=I1+15%
        inflation.confederate = 10 + (results.union - results.confederate) * 2;
        inflation.union = 10 + (results.confederate - results.union) * 2;

        // MONEY AVAILABLE
        // 可用资金

        // MEN AVAILABLE
        // 可用士兵数量
        battleState.F1 = 5 * battle.troops.confederate / 6.0;

        if (this.numGenerals == 2) {
            // 如果将军数量为2
            resources.union.budget = 100 * (int) Math.floor((battle.troops.union * (100.0 - inflation.union) / 2000) * (1 + (revenue.union - totalExpenditure.union) / (revenue.union + 1.0)) + .5);
        } else {
            resources.union.budget = 100 * (int) Math.floor(battle.troops.union * (100.0 - inflation.union) / 2000 + .5);
        }

        out.println();
        out.println();
        out.println();
        out.println();
```
在这段代码中，注释解释了每个语句的作用，包括计算通货膨胀、可用资金和可用士兵数量。同时，还提到了特定情况下的通货膨胀计算。
        out.println(); // 打印空行
        out.println("THIS IS THE BATTLE OF " + battle.name); // 打印战斗名称

        if (this.wantBattleDescriptions) { // 如果想要战斗描述
            for (var eachLine : battle.blurb) { // 遍历战斗描述的每一行
                out.println(eachLine); // 打印每一行的描述
            }
        }

        out.println(); // 打印空行
        out.println("          CONFEDERACY     UNION"); // 打印联盟名称
        out.println("MEN         " + getConfedTroops(battle) + "          " + getUnionTroops(battle)); // 打印联盟的士兵数量
        out.println("MONEY     $ " + resources.confederate.budget + "       $ " + resources.union.budget); // 打印联盟的预算
        out.println("INFLATION   " + (inflation.confederate + 15) + "%          " + inflation.union + "%"); // 打印通货膨胀率

        // ONLY IN PRINTOUT IS CONFED INFLATION = I1+15%
        // IF TWO GENERALS, INPUT CONFED. FIRST

        var terminalInput = new Scanner(System.in); // 创建一个用于接收终端输入的Scanner对象
        for (int i = 0; i < numGenerals; i++) {  // 循环遍历每个将军
            out.println();  // 输出空行

            ArmyResources currentResources;  // 声明一个军队资源对象

            if (this.numGenerals == 1 || i == 0) {  // 如果只有一个将军或者当前将军是第一个
                out.print("CONFEDERATE GENERAL --- ");  // 输出“CONFEDERATE GENERAL --- ”
                currentResources = resources.confederate;  // 将当前资源设置为南方联盟的资源
            } else {  // 否则
                out.print("UNION GENERAL --- ");  // 输出“UNION GENERAL --- ”
                currentResources = resources.union;  // 将当前资源设置为联邦的资源
            }

            var validInputs = false;  // 声明一个变量表示输入是否有效
            while (!validInputs) {  // 当输入无效时循环
                out.println("HOW MUCH DO YOU WISH TO SPEND FOR");  // 输出提示信息
                out.print("- FOOD...... ? ");  // 输出提示信息
                var food = terminalInput.nextInt();  // 读取输入的食物花费
                if (food == 0) {  // 如果食物花费为0
                    if (this.revenue.confederate != 0) {  // 如果南方联盟的收入不为0
# 输出提示信息，假设你想保持相同的分配
out.println("ASSUME YOU WANT TO KEEP SAME ALLOCATIONS")
out.println()

# 如果当前资源的食物小于0，则重新输入
if (food < 0):
    currentResources.food = food
else:
    # 输出提示信息，要求输入薪水
    out.print("- SALARIES.. ? ")
    # 从终端输入获取薪水
    currentResources.salaries = terminalInput.nextInt()

    # 输出提示信息，要求输入弹药
    out.print("- AMMUNITION ? ")
    # 从终端输入获取弹药数量
    currentResources.ammunition = terminalInput.nextInt()  # FIXME Retry if -ve

    # 如果当前资源的总和大于预算，则输出警告信息
    if (currentResources.getTotal() > currentResources.budget):
        out.println("THINK AGAIN! YOU HAVE ONLY $" + currentResources.budget)
    else:
        # 如果输入的资源合法，则将validInputs设置为True
        validInputs = True
        out.println();
        // 打印空行

        // 记录士气
        out.println(range(0, numGenerals).mapToObj(i -> moraleForArmy(battleState, i)).collect(joining(", ")));
        // 打印每个将军的士气值，并用逗号分隔

        out.println();
        // 打印空行

        return battleState;
        // 返回战斗状态对象
    }

    private int getUnionTroops(HistoricalDatum battle) {
        // 计算联盟军队的预期剩余部队数量
        return (int) Math.floor(battle.troops.union * (1 + (totalExpectedCasualties.union - totalCasualties.union) / (totalTroops.union + 1.0)));
    }

    private int getConfedTroops(HistoricalDatum battle) {
        // 计算联盟军队的预期剩余部队数量
        return (int) Math.floor(battle.troops.confederate * (1 + (totalExpectedCasualties.confederate - totalCasualties.confederate) / (totalTroops.confederate + 1.0)));
    }

    private String moraleForArmy(BattleState battleState, int armyIdx) {
        // 计算指定军队的士气值
        // 创建一个 StringBuilder 对象，用于构建字符串
        var builder = new StringBuilder();

        // 声明一个 ArmyResources 对象变量 currentResources
        ArmyResources currentResources;

        // 如果将军数量为1或者军队索引为0，则在字符串构建器中添加"CONFEDERATE "，并将 currentResources 设置为 resources.confederate
        if (this.numGenerals == 1 || armyIdx == 0) {
            builder.append("CONFEDERATE ");
            currentResources = resources.confederate;
        } 
        // 否则在字符串构建器中添加"UNION "，并将 currentResources 设置为 resources.union
        else {
            builder.append("UNION ");
            currentResources = resources.union;
        }

        // 计算士气
        currentResources.morale = (2 * Math.pow(currentResources.food, 2) + Math.pow(currentResources.salaries, 2)) / Math.pow(battleState.F1, 2) + 1;
        // 如果士气大于等于10，则在字符串构建器中添加"MORALE IS HIGH"
        if (currentResources.morale >= 10) {
            builder.append("MORALE IS HIGH");
        } 
        // 如果士气大于等于5，则在字符串构建器中添加"MORALE IS FAIR"
        else if (currentResources.morale >= 5) {
            builder.append("MORALE IS FAIR");
        } 
        // 否则在字符串构建器中添加"MORALE IS POOR"
        else {
            builder.append("MORALE IS POOR");
        }
    }

    return builder.toString();
}

private enum OffensiveStatus {
    DEFENSIVE("YOU ARE ON THE DEFENSIVE"), OFFENSIVE("YOU ARE ON THE OFFENSIVE"), BOTH_OFFENSIVE("BOTH SIDES ARE ON THE OFFENSIVE");

    private final String label;

    OffensiveStatus(String label) {
        this.label = label;
    }
}

private void offensiveLogic(HistoricalDatum battle) {
    out.print("CONFEDERATE GENERAL---");
    // ACTUAL OFF/DEF BATTLE SITUATION
    out.println(battle.offensiveStatus.label);
```

注释：
1. `}` - 结束 offensiveLogic 方法的代码块。
2. `return builder.toString();` - 返回一个字符串构建器的字符串表示。
3. `private enum OffensiveStatus {` - 定义一个名为 OffensiveStatus 的枚举类型。
4. `DEFENSIVE("YOU ARE ON THE DEFENSIVE"), OFFENSIVE("YOU ARE ON THE OFFENSIVE"), BOTH_OFFENSIVE("BOTH SIDES ARE ON THE OFFENSIVE");` - 定义了枚举类型 OffensiveStatus 的三个枚举常量。
5. `private final String label;` - 声明了一个私有的字符串变量 label。
6. `OffensiveStatus(String label) { this.label = label; }` - OffensiveStatus 的构造函数，用于初始化枚举常量的 label 变量。
7. `private void offensiveLogic(HistoricalDatum battle) {` - 定义了一个名为 offensiveLogic 的私有方法，接受一个 HistoricalDatum 类型的参数 battle。
8. `out.print("CONFEDERATE GENERAL---");` - 打印字符串 "CONFEDERATE GENERAL---"。
9. `out.println(battle.offensiveStatus.label);` - 打印 battle 对象的 offensiveStatus 属性的 label 值。
        // 选择策略

        // 如果将军数量为2，则打印"CONFEDERATE STRATEGY ? "，否则打印"YOUR STRATEGY ? "
        if (numGenerals == 2) {
            out.print("CONFEDERATE STRATEGY ? ");
        } else {
            out.print("YOUR STRATEGY ? ");
        }

        // 输入并设置confedStrategy，要求输入的值在1到5之间，否则提示"STRATEGY " + i + " NOT ALLOWED."
        confedStrategy = inputInt(i -> i >= 1 && i <= 5, i -> "STRATEGY " + i + " NOT ALLOWED.");
        // 如果confedStrategy为5，则设置confedSurrender为true
        if (confedStrategy == 5) {  // 1970
            confedSurrender = true;
        }

        // 如果将军数量为2，则打印"UNION STRATEGY ? "
        if (numGenerals == 2) {
            out.print("UNION STRATEGY ? ");

            // 输入并设置unionStrategy，要求输入的值在1到5之间，否则提示"STRATEGY " + i + " NOT ALLOWED."
            unionStrategy = inputInt(i -> i >= 1 && i <= 5, i -> "STRATEGY " + i + " NOT ALLOWED.");
            // 如果unionStrategy为5，则设置unionSurrender为true
            if (unionStrategy == 5) {  // 1970
                unionSurrender = true;
            }
        } else {
            unionStrategy();
        }
    }

    // 2070  REM : SIMULATED LOSSES-NORTH
    // 模拟北方联盟的损失
    private UnionLosses simulateUnionLosses(HistoricalDatum battle) {
        // 计算损失
        var losses = (2.0 * battle.expectedCasualties.union / 5) * (1 + 1.0 / (2 * (Math.abs(unionStrategy - confedStrategy) + 1)));
        // 根据部队数量和资源计算损失
        losses = losses * (1.28 + (5.0 * battle.troops.union / 6) / (resources.union.ammunition + 1));
        // 根据士气计算损失
        losses = Math.floor(losses * (1 + 1 / resources.union.morale) + 0.5);
        // 计算士气因子
        var moraleFactor = 100 / resources.union.morale;

        // 如果损失大于部队数量，则重新计算损失
        if (Math.floor(losses + moraleFactor) >= getUnionTroops(battle)) {
            losses = Math.floor(13.0 * getUnionTroops(battle) / 20);
            moraleFactor = 7 * losses / 13;
            excessiveUnionLosses = true;
        }

        // 返回计算得到的损失
        return new UnionLosses((int) losses, (int) Math.floor(moraleFactor));
    // 2170: 计算模拟损失
    private void calcLosses(BattleState battle) {
        // 2190
        out.println();
        out.println("            CONFEDERACY    UNION");

        // 计算 CONFEDERACY 损失
        var C5 = (2 * battle.data.expectedCasualties.confederate / 5) * (1 + 1.0 / (2 * (Math.abs(unionStrategy - confedStrategy) + 1)));
        C5 = (int) Math.floor(C5 * (1 + 1.0 / resources.confederate.morale) * (1.28 + battle.F1 / (resources.confederate.ammunition + 1.0)) + .5);
        var E = 100 / resources.confederate.morale;

        // 判断是否 CONFEDERACY 损失过大
        if (C5 + 100 / resources.confederate.morale >= battle.data.troops.confederate * (1 + (totalExpectedCasualties.confederate - totalCasualties.confederate) / (totalTroops.confederate + 1.0))) {
            C5 = (int) Math.floor(13.0 * battle.data.troops.confederate / 20 * (1 + (totalExpectedCasualties.union - totalCasualties.confederate) / (totalTroops.confederate + 1.0)));
            E = 7 * C5 / 13.0;
            excessiveConfederateLosses = true;
        }

        /////  2270
        final UnionLosses unionLosses;  // 声明一个名为unionLosses的final变量，用于存储联盟的损失

        if (this.numGenerals == 1) {  // 如果numGenerals等于1
        } else {
            unionLosses = simulateUnionLosses(battle.data);  // 调用simulateUnionLosses方法计算联盟的损失，并将结果存储在unionLosses变量中
        }

        out.println("CASUALTIES:  " + rightAlignInt(C5) + "        " + rightAlignInt(unionLosses.losses));  // 打印输出联盟的损失
        out.println("DESERTIONS:  " + rightAlignInt(E) + "        " + rightAlignInt(unionLosses.desertions));  // 打印输出联盟的叛逃人数
        out.println();

        if (numGenerals == 2) {  // 如果numGenerals等于2
            out.println("COMPARED TO THE ACTUAL CASUALTIES AT " + battle.data.name);  // 打印输出与实际伤亡情况的比较
            out.println("CONFEDERATE: " + (int) Math.floor(100 * (C5 / (double) battle.data.expectedCasualties.confederate) + 0.5) + " % OF THE ORIGINAL");  // 打印输出南方联盟的伤亡情况占实际伤亡情况的百分比
            out.println("UNION:       " + (int) Math.floor(100 * (unionLosses.losses / (double) battle.data.expectedCasualties.union) + 0.5) + " % OF THE ORIGINAL");  // 打印输出联盟的伤亡情况占实际伤亡情况的百分比

            out.println();

            // REM - 1 WHO WON
            var winner = findWinner(C5 + E, unionLosses.losses + unionLosses.desertions);  // 调用findWinner方法，计算并存储胜利方的结果
            switch (winner) {  # 使用 switch 语句根据 winner 的不同取值进行不同的操作
                case UNION -> {  # 当 winner 的取值为 UNION 时执行以下操作
                    out.println("THE UNION WINS " + battle.data.name);  # 打印出联邦获胜的信息和战斗名称
                    results.union++;  # 联邦获胜次数加一
                }
                case CONFED -> {  # 当 winner 的取值为 CONFED 时执行以下操作
                    out.println("THE CONFEDERACY WINS " + battle.data.name);  # 打印出联盟获胜的信息和战斗名称
                    results.confederate++;  # 联盟获胜次数加一
                }
                case INDECISIVE -> {  # 当 winner 的取值为 INDECISIVE 时执行以下操作
                    out.println("BATTLE OUTCOME UNRESOLVED");  # 打印出战斗结果未决的信息
                    results.indeterminate++;  # 未决战斗次数加一
                }
            }
        } else {
            out.println("YOUR CASUALTIES WERE " + Math.floor(100 * (C5 / (double) battle.data.expectedCasualties.confederate) + 0.5) + "% OF THE ACTUAL CASUALTIES AT " + battle.data.name);  # 打印出你的伤亡比例和战斗名称

            // FIND WHO WON  # 注释：找出谁赢了

            if (excessiveConfederateLosses) {  # 如果 excessiveConfederateLosses 为真
                out.println("YOU LOSE " + battle.data.name);  // 打印出战斗失败的信息和战斗名称

                if (this.battleNumber != 0) {  // 如果战斗次数不为0
                    results.union++;  // 结果中联邦联盟的胜利次数加一
                }
            } else {
                out.println("YOU WIN " + battle.data.name);  // 打印出战斗胜利的信息和战斗名称
                // 累积战斗因素，改变历史资源的可用性。如果是重播，则不更新。
                results.confederate++;  // 结果中南方联盟的胜利次数加一
            }
        }

        if (this.battleNumber != 0) {  // 如果战斗次数不为0
            totalCasualties.confederate += (int) (C5 + E);  // 南方联盟的总伤亡增加C5和E的整数部分
            totalCasualties.union += unionLosses.losses + unionLosses.desertions;  // 联邦联盟的总伤亡增加损失和叛变的数量
            totalExpectedCasualties.confederate += battle.data.expectedCasualties.confederate;  // 南方联盟的预期总伤亡增加当前战斗的预期南方联盟伤亡
            totalExpectedCasualties.union += battle.data.expectedCasualties.union;  // 联邦联盟的预期总伤亡增加当前战斗的预期联邦联盟伤亡
            totalExpenditure.confederate += resources.confederate.getTotal();  // 南方联盟的总支出增加当前资源的总数
            totalExpenditure.union += resources.union.getTotal();  // 联邦联盟的总支出增加当前资源的总数
            revenue.confederate += battle.data.troops.confederate * (100 - inflation.confederate) / 20;  // 南方联盟的总收入增加当前战斗的南方联盟部队数量乘以（100减去通货膨胀率）除以20
            # 计算联盟方的收入，根据战斗数据中的联盟方军队数量和通货膨胀率计算
            revenue.union += battle.data.troops.union * (100 - inflation.union) / 20;
            # 更新联盟方总军队数量，加上当前战斗数据中的联盟方军队数量
            totalTroops.confederate += battle.data.troops.confederate;
            # 更新联盟方总军队数量，加上当前战斗数据中的联盟方军队数量
            totalTroops.union += battle.data.troops.union;

            # 更新策略，传入当前的联盟方策略
            updateStrategies(this.confedStrategy);
        }
    }

    # 2790
    # 重置方法，将 excessiveConfederateLosses 和 excessiveUnionLosses 设置为 false
    private void reset() {
        excessiveConfederateLosses = excessiveUnionLosses = false;

        # 打印分隔线
        out.println("---------------");
    }

    # 2820  REM------FINISH OFF
    # 完成方法，打印多行空行
    private void complete() {
        out.println();
        out.println();
        out.println();
        // 输出空行
        out.println();
        // 输出空行
        out.println();
        // 输出空行
        out.println();
        // 输出“THE CONFEDERACY HAS WON x BATTLES AND LOST y”，其中 x 和 y 分别为 results.confederate 和 results.union 的值
        out.println("THE CONFEDERACY HAS WON " + results.confederate + " BATTLES AND LOST " + results.union);

        // 如果联盟策略为 5，则输出“THE CONFEDERACY HAS WON THE WAR”
        if (this.unionStrategy == 5) {
            out.println("THE CONFEDERACY HAS WON THE WAR");
        }

        // 如果联盟策略为 5 或者结果中南方联盟的胜利次数小于等于北方联盟的胜利次数，则输出“THE UNION HAS WON THE WAR”
        if (this.confedStrategy == 5 || results.confederate <= results.union) {
            out.println("THE UNION HAS WON THE WAR");
        }

        // 输出空行
        out.println();

        // 输出“FOR THE x BATTLES FOUGHT (EXCLUDING RERUNS)”，其中 x 为 results.getTotal() 的值
        out.println("FOR THE " + results.getTotal() + " BATTLES FOUGHT (EXCLUDING RERUNS)");
        // 输出表头
        out.println("                       CONFEDERACY    UNION");
        // 输出历史损失，分别为南方联盟和北方联盟的预期伤亡人数
        out.println("HISTORICAL LOSSES      " + (int) Math.floor(totalExpectedCasualties.confederate + .5) + "          " + (int) Math.floor(totalExpectedCasualties.union + .5));
        // 输出模拟损失的结果，包括联邦和联盟的损失人数
        out.println("SIMULATED LOSSES       " + (int) Math.floor(totalCasualties.confederate + .5) + "          " + (int) Math.floor(totalCasualties.union + .5));
        out.println();

        // 如果只有一个将军，输出联盟情报显示南方使用的策略百分比
        if (this.numGenerals == 1) {
            out.println();
            out.println("UNION INTELLIGENCE SUGGESTS THAT THE SOUTH USED ");
            out.println("STRATEGIES 1, 2, 3, 4 IN THE FOLLOWING PERCENTAGES");
            out.println(this.strategies[0] + "," + this.strategies[1] + "," + this.strategies[2] + "," + this.strategies[3]);
        }
    }

    // 根据联盟和联邦的损失情况判断胜利者
    private Winner findWinner(double confLosses, double unionLosses) {
        // 如果联盟和联邦都有过多损失，则判断为不明确胜利者
        if (this.excessiveConfederateLosses && this.excessiveUnionLosses) {
            return Winner.INDECISIVE;
        }

        // 如果联盟有过多损失，则判断为联邦胜利
        if (this.excessiveConfederateLosses) {
            return Winner.UNION;
        }
        if (this.excessiveUnionLosses || confLosses < unionLosses) {  # 如果联盟损失过多或者联盟损失小于联邦损失
            return Winner.CONFED;  # 返回赢家为联邦
        }

        if (confLosses == unionLosses) {  # 如果联邦损失等于联盟损失
            return Winner.INDECISIVE;  # 返回赢家为犹豫不决
        }

        return Winner.UNION;  // FIXME Really? 2400-2420 ?  # 返回赢家为联盟，但有疑问是否正确
    }

    private enum Winner {  # 定义赢家枚举类型
        CONFED, UNION, INDECISIVE
    }

    private void unionStrategy() {  # 定义联盟策略方法
        // 3130 ... so you can only input / override Union strategy on re-run??  # 3130 ... 所以你只能在重新运行时输入/覆盖联盟策略？？
        if (this.battleNumber == 0) {  # 如果战斗次数为0
            out.print("UNION STRATEGY ? ");  # 打印提示信息要求输入联盟策略
            var terminalInput = new Scanner(System.in);  # 创建一个用于从控制台输入的Scanner对象
            unionStrategy = terminalInput.nextInt();  // 从终端输入获取联盟策略
            if (unionStrategy < 0) {  // 如果联盟策略小于0
                out.println("ENTER 1, 2, 3, OR 4 (USUALLY PREVIOUS UNION STRATEGY)");  // 输出提示信息
                // FIXME Retry Y2 input !!!  // 输出错误信息
            }

            if (unionStrategy < 5) {  // 如果联盟策略小于5
                return;  // 返回
            }
        }

        var S0 = 0;  // 初始化S0为0
        var r = 100 * Math.random();  // 生成0到100之间的随机数

        for (unionStrategy = 1; unionStrategy <= 4; unionStrategy++) {  // 循环遍历联盟策略
            S0 += this.strategies[unionStrategy - 1];  // 将当前策略的权重加到S0上
            // IF ACTUAL STRATEGY INFO IS IN PROGRAM DATA STATEMENTS THEN R-100 IS EXTRA WEIGHT GIVEN TO THAT STATEGY.  // 如果实际策略信息在程序数据语句中，则R-100是额外给予该策略的权重
            if (r < S0) {  // 如果随机数小于S0
                break;  // 跳出循环
            }
        }
        // IF ACTUAL STRAT. IN,THEN HERE IS Y2= HIST. STRAT.
        // 如果实际战略存在，则这里是 Y2= 历史战略。
        out.println("UNION STRATEGY IS " + unionStrategy);
    }

    public CivilWar(PrintStream out) {
        this.out = out;

        this.results = new BattleResults();

        this.totalCasualties = new ArmyPair<>(0, 0);
        this.totalExpectedCasualties = new ArmyPair<>(0, 0);
        this.totalExpenditure = new ArmyPair<>(0, 0);
        this.totalTroops = new ArmyPair<>(0, 0);

        this.revenue = new ArmyPair<>(0, 0);
        this.inflation = new ArmyPair<>(0, 0);

        this.resources = new ArmyPair<>(new ArmyResources(), new ArmyResources());
        // 初始化联盟信息，设置四种可能的南方策略
        this.strategies = new int[]{25, 25, 25, 25};

        // 读取历史数据
        // 可以通过在适当的信息后插入数据语句并调整读取来添加更多的历史数据（如策略等）
        this.data = List.of(
        );
    }

    private void showCredits() {
        out.println(" ".repeat(26) + "CIVIL WAR");
        out.println(" ".repeat(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        out.println();
        out.println();
        out.println();
    }

    private void updateStrategies(int strategy) {
        // 更新策略
        // 记住当前策略，开始遗忘旧的策略
        // 当南方策略获得3*S时，其他联盟失去S
        // 设置初始概率点数为3，初始S0为0
        var S = 3;
        var S0 = 0;
        // 遍历4个策略
        for (int i = 0; i < 4; i++) {
            // 如果某个策略的概率点数低于等于5%，则跳过
            if (this.strategies[i] <= 5) {
                continue;
            }
            // 减去S点数，并将S点数累加到S0
            this.strategies[i] -= S;
            S0 += S;
        }
        // 将累积的S0点数加到指定策略上
        this.strategies[strategy - 1] += S0;
    }

    // 显示帮助信息
    private void showHelp() {
        out.println(); // 输出空行
        out.println(); // 输出空行
    }
        out.println(); // 打印空行
        out.println(); // 打印空行
        out.println("THIS IS A CIVIL WAR SIMULATION."); // 打印游戏标题
        out.println("TO PLAY TYPE A RESPONSE WHEN THE COMPUTER ASKS."); // 提示玩家如何进行游戏
        out.println("REMEMBER THAT ALL FACTORS ARE INTERRELATED AND THAT YOUR"); // 提示玩家注意所有因素之间的相互关系
        out.println("RESPONSES COULD CHANGE HISTORY. FACTS AND FIGURES USED ARE"); // 提示玩家他们的选择可能会改变历史
        out.println("BASED ON THE ACTUAL OCCURRENCE. MOST BATTLES TEND TO RESULT"); // 提示玩家游戏基于实际发生的事件
        out.println("AS THEY DID IN THE CIVIL WAR, BUT IT ALL DEPENDS ON YOU!!"); // 提示玩家游戏结果取决于他们的选择
        out.println(); // 打印空行
        out.println("THE OBJECT OF THE GAME IS TO WIN AS MANY BATTLES AS "); // 提示玩家游戏目标是赢得尽可能多的战斗
        out.println("POSSIBLE."); // 提示玩家游戏目标是赢得尽可能多的战斗
        out.println(); // 打印空行
        out.println("YOUR CHOICES FOR DEFENSIVE STRATEGY ARE:"); // 提示玩家防御策略选项
        out.println("        (1) ARTILLERY ATTACK"); // 提示玩家防御策略选项
        out.println("        (2) FORTIFICATION AGAINST FRONTAL ATTACK"); // 提示玩家防御策略选项
        out.println("        (3) FORTIFICATION AGAINST FLANKING MANEUVERS"); // 提示玩家防御策略选项
        out.println("        (4) FALLING BACK"); // 提示玩家防御策略选项
        out.println(" YOUR CHOICES FOR OFFENSIVE STRATEGY ARE:"); // 提示玩家进攻策略选项
        out.println("        (1) ARTILLERY ATTACK"); // 提示玩家进攻策略选项
        out.println("        (2) FRONTAL ATTACK"); // 提示玩家进攻策略选项
        // 输出提示信息：FLANKING MANEUVERS
        out.println("        (3) FLANKING MANEUVERS");
        // 输出提示信息：ENCIRCLEMENT
        out.println("        (4) ENCIRCLEMENT");
        // 输出提示信息：输入5可投降
        out.println("YOU MAY SURRENDER BY TYPING A '5' FOR YOUR STRATEGY.");
    }

    // 定义最大数字长度为6
    private static final int MAX_NUM_LENGTH = 6;

    // 将整数右对齐并转换为字符串
    private String rightAlignInt(int number) {
        // 将整数转换为字符串
        var s = String.valueOf(number);
        // 返回右对齐后的字符串
        return " ".repeat(MAX_NUM_LENGTH - s.length()) + s;
    }

    // 将浮点数右对齐并转换为字符串
    private String rightAlignInt(double number) {
        // 调用整数右对齐方法，将浮点数转换为整数后再右对齐
        return rightAlignInt((int) Math.floor(number));
    }

    // 输入字符串并进行验证
    private static String inputString(Predicate<String> validator, String reminder) {
        // 循环直到输入合法
        while (true) {
            try {
                // 从控制台获取输入
                var input = new Scanner(System.in).nextLine();
                if (validator.test(input)) {  # 检查输入是否符合给定的条件
                    return input;  # 如果输入符合条件，则返回输入值
                }
            } catch (InputMismatchException e) {  # 捕获输入不匹配异常
                // Ignore  # 忽略异常
            }
            System.out.println(reminder);  # 打印提醒信息
        }
    }

    private static int inputInt(Predicate<Integer> validator, Function<Integer, String> reminder) {  # 定义一个静态方法，接受一个整数验证器和一个提醒函数作为参数
        while (true) {  # 进入循环，持续接受输入
            try {  # 尝试执行以下代码
                var input = new Scanner(System.in).nextInt();  # 从标准输入中获取一个整数
                if (validator.test(input)) {  # 检查输入是否符合给定的条件
                    return input;  # 如果输入符合条件，则返回输入值
                }
                System.out.println(reminder.apply(input));  # 打印提醒信息
            } catch (InputMismatchException e) {  # 捕获输入不匹配异常
                System.out.println(reminder.apply(0));  # 打印提醒信息
    private static boolean isYes(String s) {
        // 检查输入字符串是否为 null，如果是则返回 false
        if (s == null) {
            return false;
        }
        // 将输入字符串转换为大写形式
        var uppercase = s.toUpperCase();
        // 检查转换后的字符串是否等于 "Y" 或 "YES"，如果是则返回 true，否则返回 false
        return uppercase.equals("Y") || uppercase.equals("YES");
    }

    private static boolean isNo(String s) {
        // 检查输入字符串是否为 null，如果是则返回 false
        if (s == null) {
            return false;
        }
        // 将输入字符串转换为大写形式
        var uppercase = s.toUpperCase();
        // 检查转换后的字符串是否等于 "N" 或 "NO"，如果是则返回 true，否则返回 false
        return uppercase.equals("N") || uppercase.equals("NO");
    }
        // 定义一个私有内部类 BattleState，包含一个 HistoricalDatum 类型的数据和一个 double 类型的 F1 值
        private static class BattleState {
            private final HistoricalDatum data; // 历史数据
            private double F1; // F1 值

            // 构造方法，接受一个 HistoricalDatum 类型的参数，并将其赋值给 data
            public BattleState(HistoricalDatum data) {
                this.data = data;
            }
        }

        // 定义一个私有内部类 ArmyPair，包含两个泛型类型的数据，分别为 confederate 和 union
        private static class ArmyPair<T> {
            private T confederate; // 南方军队
            private T union; // 北方军队

            // 构造方法，接受两个泛型类型的参数，并将其分别赋值给 confederate 和 union
            public ArmyPair(T confederate, T union) {
                this.confederate = confederate;
                this.union = union;
            }
        }

        // 定义一个私有内部类 BattleResults
        private int confederate;  // 定义私有整型变量 confederate
        private int union;  // 定义私有整型变量 union
        private int indeterminate;  // 定义私有整型变量 indeterminate

        public int getTotal() {  // 定义公有方法 getTotal，用于返回 confederate、union 和 indeterminate 三者之和
            return confederate + union + indeterminate;
        }
    }

    private static class ArmyResources {  // 定义私有静态内部类 ArmyResources
        private int food;  // 定义私有整型变量 food
        private int salaries;  // 定义私有整型变量 salaries
        private int ammunition;  // 定义私有整型变量 ammunition
        private int budget;  // 定义私有整型变量 budget

        private double morale;  // 定义私有双精度浮点型变量 morale  // TODO really here?

        public int getTotal() {  // 定义公有方法 getTotal，用于返回 food、salaries 和 ammunition 三者之和
            return this.food + this.salaries + this.ammunition;
        }
    }

    private record HistoricalDatum(String name, ArmyPair<Integer> troops,
                                   ArmyPair<Integer> expectedCasualties,
                                   OffensiveStatus offensiveStatus, String[] blurb) {
    }
    # 定义一个名为HistoricalDatum的记录类型，包含name、troops、expectedCasualties、offensiveStatus和blurb字段

    private record UnionLosses(int losses, int desertions) {
    }
    # 定义一个名为UnionLosses的记录类型，包含losses和desertions字段
}
```
这段代码是Java语言的记录类型定义，用于定义数据结构。
```