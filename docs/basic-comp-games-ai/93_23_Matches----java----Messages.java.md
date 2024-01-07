# `basic-computer-games\93_23_Matches\java\Messages.java`

```

// 这是一个工具类，只包含静态成员。
// 工具类不应该被实例化。
private Messages() {
    throw new IllegalStateException("Utility class");
}

// 游戏介绍信息
public static final String INTRO = """
                                      23 MATCHES
                      CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY



         THIS IS A GAME CALLED '23 MATCHES'.

        WHEN IT IS YOUR TURN, YOU MAY TAKE ONE, TWO, OR THREE
        MATCHES. THE OBJECT OF THE GAME IS NOT TO HAVE TO TAKE
        THE LAST MATCH.

        LET'S FLIP A COIN TO SEE WHO GOES FIRST.
        IF IT COMES UP HEADS, I WILL WIN THE TOSS.
        """;

// 头部信息，如果是正面
public static final String HEADS = """
        HEADS! I WIN! HA! HA!
        PREPARE TO LOSE, MEATBALL-NOSE!!

        I TAKE 2 MATCHES
        """;

// 头部信息，如果是反面
public static final String TAILS = """
        TAILS! YOU GO FIRST.
        """;

// 剩余火柴数量信息
public static final String MATCHES_LEFT = """
        THE NUMBER OF MATCHES IS NOW %d

        YOUR TURN -- YOU MAY TAKE 1, 2 OR 3 MATCHES.
        """;

// 询问要移除多少火柴
public static final String REMOVE_MATCHES_QUESTION = "HOW MANY DO YOU WISH TO REMOVE? ";

// 剩余火柴数量信息
public static final String REMAINING_MATCHES = """
        THERE ARE NOW %d MATCHES REMAINING.
        """;

// 无效输入信息
public static final String INVALID = """
        VERY FUNNY! DUMMY!
        DO YOU WANT TO PLAY OR GOOF AROUND?
        NOW, HOW MANY MATCHES DO YOU WANT?
        """;

// 胜利信息
public static final String WIN = """
        YOU WON, FLOPPY EARS !
        THINK YOU'RE PRETTY SMART !
        LETS PLAY AGAIN AND I'LL BLOW YOUR SHOES OFF !!
        """;

// 电脑回合信息
public static final String CPU_TURN = """
        MY TURN ! I REMOVE %d MATCHES.
        """;

// 失败信息
public static final String LOSE = """
        YOU POOR BOOB! YOU TOOK THE LAST MATCH! I GOTCHA!!
        HA ! HA ! I BEAT YOU !!!

        GOOD BYE LOSER!
        """;

```