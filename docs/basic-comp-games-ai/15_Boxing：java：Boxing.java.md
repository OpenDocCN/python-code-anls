# `15_Boxing\java\Boxing.java`

```
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
# 循环3次，表示3个回合
for (int roundNro = 1; roundNro <= 3; roundNro++) {
    # 如果比赛结束，跳出循环
    if (session.isOver())
        break;

    # 重置比分
    session.resetPoints();
    # 打印回合开始信息
    console.print("\nROUND %d BEGINS...%n", roundNro);

    # 循环7次，表示每个回合内的7次主要出拳
    for (int majorPunches = 1; majorPunches <= 7; majorPunches++) {
        # 生成一个0到9的随机数
        long i = Basic.randomOf(10);

        # 如果随机数大于5，对手出拳并检查是否停止出拳
        if (i > 5) {
            boolean stopPunches = opponentPunch();
            # 如果需要停止出拳，跳出循环
            if (stopPunches ) break;
        } else {
            # 否则玩家出拳
            playerPunch();
        }
    }
    # 展示本回合的获胜者
    showRoundWinner(roundNro);
}
        }
        showWinner();
    }
```
这段代码是一个方法的结束和一个条件语句的结束，可能是一个游戏的逻辑控制部分。

```
    private boolean opponentPunch() {
```
这是一个名为opponentPunch的私有方法，返回一个布尔值。

```
        final Punch punch = Punch.random();
```
创建一个名为punch的Punch对象，并赋予一个随机值。

```
        if (punch == session.getOpponent().getBestPunch()) session.addOpponentPoints(2);
```
如果punch等于session.getOpponent().getBestPunch()，则session.addOpponentPoints(2)。

```
        if (punch == Punch.FULL_SWING) {
```
如果punch等于Punch.FULL_SWING，则执行以下代码块。

```
            console.print("%s TAKES A FULL SWING AND", session.getOpponent().getName());
```
打印出session.getOpponent().getName()进行全力挥拳的动作。

```
            long r6 = Basic.randomOf(60);
```
创建一个名为r6的长整型变量，并赋予一个0到59的随机值。

```
            if (session.getPlayer().hitVulnerability(Punch.FULL_SWING) || r6 < 30) {
```
如果session.getPlayer().hitVulnerability(Punch.FULL_SWING)为真，或者r6小于30，则执行以下代码块。

```
                console.print(" POW!!!!! HE HITS HIM RIGHT IN THE FACE!\n");
```
打印出" POW!!!!! HE HITS HIM RIGHT IN THE FACE!"。

```
                if (session.getPoints(session.getOpponent()) > 35) {
```
如果session.getPoints(session.getOpponent())大于35，则执行以下代码块。

```
                    session.setKnocked();
                    return true;
                }
```
调用session.setKnocked()方法，并返回true。
                session.addOpponentPoints(15);  # 如果拳击动作是钩拳或上勾拳，对手得到15分
            } else {
                console.print(" IT'S BLOCKED!\n");  # 如果拳击动作被阻挡，打印提示信息
            }
        }

        if (punch == Punch.HOOK  || punch == Punch.UPPERCUT) {  # 如果拳击动作是钩拳或上勾拳
            if (punch == Punch.HOOK) {  # 如果拳击动作是钩拳
                console.print("%s GETS %s IN THE JAW (OUCH!)\n", session.getOpponent().getName(), session.getPlayer().getName());  # 打印对手被击中下巴的提示信息

                session.addOpponentPoints(7);  # 对手得到7分
                console.print("....AND AGAIN!\n");  # 打印连续攻击的提示信息

                session.addOpponentPoints(5);  # 对手再次得到5分
                if (session.getPoints(session.getOpponent()) > 35) {  # 如果对手得分超过35分
                    session.setKnocked();  # 设置对手被击倒
                    return true;  # 返回true
                }
                console.print("\n");  # 打印换行符
            }
            # 打印玩家被上勾拳攻击的信息
            console.print("%s IS ATTACKED BY AN UPPERCUT (OH,OH)...\n", session.getPlayer().getName());
            # 生成一个0到200之间的随机数
            long q4 = Basic.randomOf(200);
            # 如果玩家被上勾拳攻击或者随机数小于等于75
            if (session.getPlayer().hitVulnerability(Punch.UPPERCUT) || q4 <= 75) {
                # 打印对手连接上勾拳的信息
                console.print("AND %s CONNECTS...\n", session.getOpponent().getName());
                # 给对手加8分
                session.addOpponentPoints(8);
            } else {
                # 打印对手阻挡并用钩拳攻击玩家的信息
                console.print(" BLOCKS AND HITS %s WITH A HOOK.\n", session.getOpponent().getName());
                # 给玩家加5分
                session.addPlayerPoints(5);
            }
        }
        else {
            # 打印对手出拳的信息
            console.print("%s JABS AND ", session.getOpponent().getName());
            # 生成一个0到7之间的随机数
            long z4 = Basic.randomOf(7);
            # 如果玩家被出拳攻击
            if (session.getPlayer().hitVulnerability(Punch.JAB))
                # 给对手加5分
                session.addOpponentPoints(5);
            # 如果随机数大于4
            else if (z4 > 4) {
                console.print(" BLOOD SPILLS !!!\n");  # 打印出玩家的拳击动作造成对手受伤的消息

                session.addOpponentPoints(5);  # 给对手增加5分
            } else {
                console.print("IT'S BLOCKED!\n");  # 打印出对手成功阻挡玩家的拳击动作的消息
            }
        }
        return true;  # 返回true表示玩家成功进行了拳击动作
    }

    private void playerPunch() {  # 定义玩家进行拳击动作的方法
        console.print("%s'S PUNCH? ", session.getPlayer().getName());  # 打印出玩家的名字和提示输入拳击动作的消息
        final Punch punch = Punch.fromCode(console.readInt());  # 从控制台读取玩家输入的拳击动作代码并转换成Punch对象

        if (punch == session.getPlayer().getBestPunch()) session.addPlayerPoints(2);  # 如果玩家选择的拳击动作是最佳拳击动作，则给玩家增加2分

        switch (punch) {  # 根据玩家选择的拳击动作进行不同的处理
            case FULL_SWING -> {  # 如果是全力挥拳
                console.print("%s SWINGS AND ", session.getPlayer().getName());  # 打印出玩家挥拳的消息
                if (session.getOpponent().getBestPunch() == Punch.JAB) {  # 如果对手的最佳拳击动作是快速戳击
                    console.print("HE CONNECTS!\n");  # 打印“他连接了！”
                    if (session.getPoints(session.getPlayer()) <= 35) session.addPlayerPoints(15);  # 如果玩家得分小于等于35，则给玩家加15分
                } else {
                    long x3 = Basic.randomOf(30);  # 生成一个30以内的随机数
                    if (x3 < 10) {
                        console.print("HE CONNECTS!\n");  # 打印“他连接了！”
                        if (session.getPoints(session.getPlayer()) <= 35) session.addPlayerPoints(15);  # 如果玩家得分小于等于35，则给玩家加15分
                    } else {
                        console.print("HE MISSES \n");  # 打印“他没击中”
                        if (session.getPoints(session.getPlayer()) != 1) console.print("\n\n");  # 如果玩家得分不等于1，则打印两个换行符
                    }
                }
            }
            case HOOK -> {
                console.print("\n%s GIVES THE HOOK... ", session.getPlayer().getName());  # 打印“某人发出了钩拳…”
                long h1 = Basic.randomOf(2);  # 生成一个0或1的随机数
                if (session.getOpponent().getBestPunch() == Punch.HOOK) {  # 如果对手的最佳拳是钩拳
                    session.addPlayerPoints(7);  # 给玩家加7分
                } else if (h1 == 1) {
                    console.print("BUT IT'S BLOCKED!!!!!!!!!!!!!\n");  // 打印出信息，表示拳击动作被对手挡住了
                } else {
                    console.print("CONNECTS...\n");  // 打印出信息，表示拳击动作成功击中对手

                    session.addPlayerPoints(7);  // 给玩家加7分
                }
            }
            case UPPERCUT -> {
                console.print("\n%s  TRIES AN UPPERCUT ", session.getPlayer().getName());  // 打印出信息，表示玩家尝试进行上勾拳
                long d5 = Basic.randomOf(100);  // 生成一个0到100之间的随机数
                if (session.getOpponent().getBestPunch() == Punch.UPPERCUT || d5 < 51) {  // 如果对手最擅长的拳击动作是上勾拳，或者随机数小于51
                    console.print("AND HE CONNECTS!\n");  // 打印出信息，表示上勾拳成功击中对手

                    session.addPlayerPoints(4);  // 给玩家加4分
                } else {
                    console.print("AND IT'S BLOCKED (LUCKY BLOCK!)\n");  // 打印出信息，表示上勾拳被对手挡住了
                }
            }
            default -> {
                console.print("%s JABS AT %s'S HEAD \n", session.getPlayer().getName(), session.getOpponent().getName());  // 打印出信息，表示玩家进行了一次快速的戳击动作
                # 如果对手的最佳拳击动作是JAB
                if (session.getOpponent().getBestPunch() == Punch.JAB) {
                    # 给玩家加3分
                    session.addPlayerPoints(3);
                } else {
                    # 生成一个0到7之间的随机数
                    long c = Basic.randomOf(8);
                    # 如果随机数小于4
                    if (c < 4) {
                        # 打印信息：被阻挡了
                        console.print("IT'S BLOCKED.\n");
                    } else {
                        # 给玩家加3分
                        session.addPlayerPoints(3);
                    }
                }
            }
        }
    }

    # 显示本轮的获胜者
    private void showRoundWinner(int roundNro) {
        # 如果玩家是本轮的获胜者
        if (session.isRoundWinner(session.getPlayer())) {
            # 打印信息：某某赢得了第几轮
            console.print("\n %s WINS ROUND %d\n", session.getPlayer().getName(), roundNro);
            # 将本轮的获胜者添加到回合获胜者列表中
            session.addRoundWind(session.getPlayer());
    } else {
        console.print("\n %s WINS ROUND %d\n", session.getOpponent().getName(), roundNro);
        session.addRoundWind(session.getOpponent());
    }
```
这段代码是一个条件语句，如果条件不满足，则执行大括号内的代码。在这里，如果条件不满足，就会打印出对手的名字和当前回合数，并调用session.addRoundWind()方法。

```
private void showWinner() {
    if (session.isGameWinner(session.getOpponent())) {
        console.print("%s WINS (NICE GOING, " + session.getOpponent().getName() + ").", session.getOpponent().getName());
    } else if (session.isGameWinner(session.getPlayer())) {
        console.print("%s AMAZINGLY WINS!!", session.getPlayer().getName());
    } else if (session.isPlayerKnocked()) {
        console.print("%s IS KNOCKED COLD AND %s IS THE WINNER AND CHAMP!", session.getPlayer().getName(), session.getOpponent().getName());
    } else {
        console.print("%s IS KNOCKED COLD AND %s IS THE WINNER AND CHAMP!", session.getOpponent().getName(), session.getPlayer().getName());
    }

    console.print("\n\nAND NOW GOODBYE FROM THE OLYMPIC ARENA.\n");
}
```
这段代码定义了一个名为showWinner()的方法。在方法内部，使用条件语句来判断游戏的胜者，并根据不同的情况打印出不同的消息。最后，打印出一条结束游戏的消息。
    // 加载玩家信息
    private void loadPlayers() {
        // 打印提示信息，要求输入对手的名字
        console.print("WHAT IS YOUR OPPONENT'S NAME? ");
        // 读取输入的对手名字
        final String opponentName = console.readLine();

        // 打印提示信息，要求输入玩家的名字
        console.print("INPUT YOUR MAN'S NAME? ");
        // 读取输入的玩家名字
        final String playerName = console.readLine();

        // 打印提示信息，列出不同的拳击动作选项
        console.print("DIFFERENT PUNCHES ARE: (1) FULL SWING; (2) HOOK; (3) UPPERCUT; (4) JAB.\n");
        console.print("WHAT IS YOUR MANS BEST? ");
        // 读取输入的玩家最擅长的拳击动作
        final int b = console.readInt();

        // 打印提示信息，要求输入对手的脆弱性
        console.print("WHAT IS HIS VULNERABILITY? ");
        // 读取输入的对手的脆弱性
        final int d = console.readInt();

        // 创建玩家对象，包括名字和拳击动作
        final Player player = new Player(playerName, Punch.fromCode(b), Punch.fromCode(d));
        // 创建对手玩家对象，只包括名字
        final Player opponent = new Player(opponentName);

        // 创建游戏会话对象，包括玩家和对手玩家
        session = new GameSession(player, opponent);
    }
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
```

需要注释的代码：

```java
    private void showIntro () {
        // 打印游戏介绍
        console.print("                                 BOXING\n");
        console.print("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n");
        console.print("BOXING OLYMPIC STYLE (3 ROUNDS -- 2 OUT OF 3 WINS)\n\n");
    }
}
```