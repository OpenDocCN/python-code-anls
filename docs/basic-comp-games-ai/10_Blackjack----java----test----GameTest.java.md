# `10_Blackjack\java\test\GameTest.java`

```
# 导入 JUnit 的测试相关类
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
# 导入 JUnit 的断言相关类
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
# 导入 IO 相关类
import java.io.EOFException;
import java.io.StringReader;
import java.io.StringWriter;
import java.io.UncheckedIOException;
# 导入集合相关类
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
# 定义一个名为GameTest的公共类
public class GameTest {

    # 声明私有的StringReader和StringWriter对象
    private StringReader in;
    private StringWriter out;
    # 声明私有的Game对象
    private Game game;

    # 声明私有的StringBuilder对象playerActions和LinkedList对象cards
    private StringBuilder playerActions;
    private LinkedList<Card> cards;

    # 在每个测试方法执行前重置输入输出流和游戏对象
    @BeforeEach
    public void resetIo() {
        in = null;
        out = null;
        game = null;
        # 初始化playerActions为一个空的StringBuilder对象
        playerActions = new StringBuilder();
        # 初始化cards为一个空的LinkedList对象
        cards = new LinkedList<>();
    }

    # 定义一个名为playerGets的私有方法，接受value和suit两个参数
    private void playerGets(int value, Card.Suit suit) {
        cards.add(new Card(value, suit));  // 向牌组中添加一张新的牌，牌的数值为value，花色为suit
    }

    private void playerSays(String action) {
        playerActions.append(action).append(System.lineSeparator());  // 将玩家的动作添加到玩家动作记录中，并在每个动作之间添加换行符
    }

    private void initGame() {
        System.out.printf("Running game with input: %s\tand cards: %s\n",playerActions.toString(), cards);  // 打印游戏初始化信息，包括玩家动作记录和牌组情况
        in = new StringReader(playerActions.toString());  // 使用玩家动作记录创建一个字符串读取器
        out = new StringWriter();  // 创建一个字符串写入器
        UserIo userIo = new UserIo(in, out);  // 创建一个用户输入输出对象，用于模拟用户输入和输出
        Deck deck = new Deck((c) -> cards);  // 创建一副牌，使用lambda表达式将牌组传入
        game = new Game(deck, userIo);  // 创建一个游戏对象，传入牌组和用户输入输出对象
    }

    @AfterEach
    private void printOutput() {
        System.out.println(out.toString());  // 打印游戏输出信息
    }
@Test
public void shouldQuitOnCtrlD() {
    // 给定
    playerSays("\u2404"); // U+2404 is "End of Transmission" sent by CTRL+D (or CTRL+Z on Windows)
    initGame();

    // 当
    Exception e = assertThrows(UncheckedIOException.class, game::run);

    // 然后
    assertTrue(e.getCause() instanceof EOFException);
    assertEquals("!END OF INPUT", e.getMessage());
}

@Test
@DisplayName("collectInsurance() should not prompt on N")
public void collectInsuranceNo(){
    // 给定
    List<Player> players = Collections.singletonList(new Player(1));
        playerSays("N"); // 模拟玩家输入"N"
        initGame(); // 初始化游戏

        // When
        game.collectInsurance(players); // 调用collectInsurance方法，传入玩家列表参数

        // Then
        assertAll(
            () -> assertTrue(out.toString().contains("ANY INSURANCE")), // 断言输出结果包含"ANY INSURANCE"
            () -> assertFalse(out.toString().contains("INSURANCE BETS")) // 断言输出结果不包含"INSURANCE BETS"
        );
    }

    @Test
    @DisplayName("collectInsurance() should collect on Y")
    public void collectInsuranceYes(){
        // Given
        List<Player> players = Collections.singletonList(new Player(1)); // 创建一个只包含一个玩家的玩家列表
        players.get(0).setCurrentBet(100); // 设置玩家的当前赌注为100
        playerSays("Y"); // 模拟玩家输入"Y"
        playerSays("50");  // 调用 playerSays 方法，传入参数 "50"
        initGame();  // 调用 initGame 方法

        // When
        game.collectInsurance(players);  // 调用 game 对象的 collectInsurance 方法，传入 players 参数

        // Then
        assertAll(
            () -> assertTrue(out.toString().contains("ANY INSURANCE")),  // 断言 out 对象的字符串表示是否包含 "ANY INSURANCE"
            () -> assertTrue(out.toString().contains("INSURANCE BETS")),  // 断言 out 对象的字符串表示是否包含 "INSURANCE BETS"
            () -> assertEquals(50, players.get(0).getInsuranceBet())  // 断言 players 列表中第一个元素的 insuranceBet 属性是否等于 50
        );
    }

    @Test
    @DisplayName("collectInsurance() should not allow more than 50% of current bet")
    public void collectInsuranceYesTooMuch(){
        // Given
        List<Player> players = Collections.singletonList(new Player(1));  // 创建一个只包含一个元素的 players 列表
        players.get(0).setCurrentBet(100);  // 设置 players 列表中第一个元素的 currentBet 属性为 100
        playerSays("Y"); // 玩家输入"Y"
        playerSays("51"); // 玩家输入"51"
        playerSays("50"); // 玩家输入"50"
        initGame(); // 初始化游戏

        // 当
        game.collectInsurance(players); // 收集保险金

        // 然后
        assertAll(
            () -> assertEquals(50, players.get(0).getInsuranceBet()), // 断言玩家的保险金为50
            () -> assertTrue(out.toString().contains("# 1 ? # 1 ?")) // 断言输出包含指定字符串
        );
    }

    @Test
    @DisplayName("collectInsurance() should not allow negative bets")
    public void collectInsuranceYesNegative(){
        // 给定
        List<Player> players = Collections.singletonList(new Player(1)); // 创建一个玩家列表
        players.get(0).setCurrentBet(100);  // 设置第一个玩家的当前赌注为100
        playerSays("Y");  // 玩家选择投保
        playerSays("-1");  // 玩家输入-1
        playerSays("1");  // 玩家输入1
        initGame();  // 初始化游戏

        // 当
        game.collectInsurance(players);  // 收集玩家的保险赌注

        // 然后
        assertAll(
            () -> assertEquals(1, players.get(0).getInsuranceBet()),  // 断言第一个玩家的保险赌注为1
            () -> assertTrue(out.toString().contains("# 1 ? # 1 ?"))  // 断言输出包含指定字符串
        );
    }

    @Test
    @DisplayName("collectInsurance() should prompt all players")  // 测试收集保险赌注时应提示所有玩家
    public void collectInsuranceYesTwoPlayers(){
        // 给定
        List<Player> players = Arrays.asList(  // 创建一个包含两个Player对象的列表
            new Player(1),  // 创建一个id为1的Player对象
            new Player(2)   // 创建一个id为2的Player对象
        );
        players.get(0).setCurrentBet(100);  // 设置第一个Player的当前赌注为100
        players.get(1).setCurrentBet(100);  // 设置第二个Player的当前赌注为100

        playerSays("Y");  // 模拟玩家输入"Y"
        playerSays("50");  // 模拟玩家输入"50"
        playerSays("25");  // 模拟玩家输入"25"
        initGame();  // 初始化游戏

        // 当
        game.collectInsurance(players);  // 收集玩家的保险赌注

        // 然后
        assertAll(  // 断言所有条件都为真
            () -> assertEquals(50, players.get(0).getInsuranceBet()),  // 确保第一个玩家的保险赌注为50
            () -> assertEquals(25, players.get(1).getInsuranceBet()),  // 确保第二个玩家的保险赌注为25
            () -> assertTrue(out.toString().contains("# 1 ? # 2 ?"))  // 确保输出包含"# 1 ? # 2 ?"
    );
```
这是一个测试方法的结束标记。

```
    @Test
    @DisplayName("play() should end on STAY")
    public void playEndOnStay(){
```
这是一个测试方法的声明，用于测试play()方法在玩家选择停止时是否正确结束。

```
        // Given
        Player player = new Player(1);
        player.dealCard(new Card(3, Card.Suit.CLUBS));
        player.dealCard(new Card(2, Card.Suit.SPADES));
        playerSays("S"); // "I also like to live dangerously."
        initGame();
```
这部分是测试的准备工作，包括创建玩家对象，给玩家发牌，玩家选择停止，以及初始化游戏。

```
        // When
        game.play(player);
```
这是测试的执行部分，调用play()方法进行游戏。

```
        // Then
        assertTrue(out.toString().startsWith("PLAYER 1 ? TOTAL IS 5"));
```
这是测试的断言部分，验证游戏是否按预期进行。在这个例子中，断言验证游戏输出是否以"PLAYER 1 ? TOTAL IS 5"开头。
    @Test
    // 标记该方法为测试方法，用于测试 play() 方法是否能够允许玩家一直要牌直到爆牌
    @DisplayName("play() should allow HIT until BUST")
    public void playHitUntilBust() {
        // Given
        // 初始化玩家对象，发两张牌给玩家
        Player player = new Player(1);
        player.dealCard(new Card(10, Card.Suit.HEARTS));
        player.dealCard(new Card(10, Card.Suit.SPADES));

        // 玩家选择要牌，得到一张牌
        playerSays("H");
        playerGets(1, Card.Suit.SPADES); // 20
        // 玩家选择要牌，得到一张牌
        playerSays("H");
        playerGets(1, Card.Suit.HEARTS); // 21
        // 玩家选择要牌，得到一张牌
        playerSays("H");
        playerGets(1, Card.Suit.CLUBS); // 22 - D'oh!
        // 初始化游戏

        // When
        // 调用游戏的 play() 方法，测试玩家是否能够一直要牌直到爆牌
        game.play(player);

        // Then
        assertTrue(out.toString().contains("BUSTED"));
    }
```
这是一个测试用例，用于测试游戏中玩家是否能够在初始回合进行双倍下注。

```
    @Test
    @DisplayName("Should allow double down on initial turn")
    public void playDoubleDown(){
```
这是一个测试方法的声明，用于测试在初始回合是否允许玩家进行双倍下注。

```
        // Given
        Player player = new Player(1);
        player.setCurrentBet(100);
        player.dealCard(new Card(10, Card.Suit.HEARTS));
        player.dealCard(new Card(4, Card.Suit.SPADES));
```
在测试之前，设置了一个玩家对象，设置了玩家的当前下注金额，并发放了两张牌给玩家。

```
        playerSays("D");
        playerGets(7, Card.Suit.SPADES);
        initGame();
```
模拟玩家选择双倍下注，发放一张牌给玩家，初始化游戏。

```
        // When
        game.play(player);
```
执行游戏的玩家操作。

```
        // Then
```
测试游戏的结果。
        assertTrue(player.getCurrentBet() == 200);  # 确保玩家当前的赌注为200
        assertTrue(player.getHand().size() == 3);  # 确保玩家手中的牌数量为3
    }

    @Test
    @DisplayName("Should NOT allow double down after initial deal")
    public void playDoubleDownLate(){
        // Given
        Player player = new Player(1);  # 创建一个玩家对象
        player.setCurrentBet(100);  # 设置玩家的当前赌注为100
        player.dealCard(new Card(10, Card.Suit.HEARTS));  # 给玩家发一张牌，点数为10，花色为红心
        player.dealCard(new Card(2, Card.Suit.SPADES));  # 给玩家发一张牌，点数为2，花色为黑桃

        playerSays("H");  # 玩家选择命令为“H”（可能是“Hit”）
        playerGets(7, Card.Suit.SPADES);  # 玩家获得一张牌，点数为7，花色为黑桃
        playerSays("D");  # 玩家选择命令为“D”（可能是“Double Down”）
        playerSays("S");  # 玩家选择命令为“S”（可能是“Stand”）
        initGame();  # 初始化游戏

        // When
        game.play(player); // 调用游戏对象的play方法，传入玩家对象作为参数

        // Then
        assertTrue(out.toString().contains("TYPE H, OR S, PLEASE")); // 断言输出的字符串中包含特定的提示信息
    }

    @Test
    @DisplayName("play() should end on STAY after split")
    public void playSplitEndOnStay(){
        // Given
        Player player = new Player(1); // 创建一个玩家对象，初始资金为1
        player.dealCard(new Card(1, Card.Suit.CLUBS)); // 玩家获得一张梅花1的牌
        player.dealCard(new Card(1, Card.Suit.SPADES)); // 玩家获得一张黑桃1的牌

        playerSays("/"); // 玩家选择分牌
        playerGets(2, Card.Suit.SPADES); // 玩家获得一张黑桃2的牌，用于第一手牌
        playerSays("S"); // 玩家选择停牌
        playerGets(2, Card.Suit.SPADES); // 玩家获得一张黑桃2的牌，用于第二手牌
        playerSays("S"); // 玩家选择停牌
        initGame(); // 初始化游戏
        // 调用游戏对象的play方法，传入玩家对象作为参数
        game.play(player);

        // 断言输出流中包含指定的字符串
        assertTrue(out.toString().contains("FIRST HAND RECEIVES"));
        assertTrue(out.toString().contains("SECOND HAND RECEIVES"));
    }

    @Test
    @DisplayName("play() should allow HIT until BUST after split")
    public void playSplitHitUntilBust() {
        // 给定玩家对象
        Player player = new Player(1);
        // 为玩家发牌
        player.dealCard(new Card(10, Card.Suit.HEARTS));
        player.dealCard(new Card(10, Card.Suit.SPADES));

        // 玩家选择分牌
        playerSays("/");
        // 玩家获得一张牌
        playerGets(12, Card.Suit.SPADES); // First hand has 20
        // 玩家选择继续要牌
        playerSays("H");
        playerGets(12, Card.Suit.HEARTS); // First hand busted  // 玩家手牌得到12点，花色为红心；第一手牌爆牌
        playerGets(10, Card.Suit.HEARTS); // Second hand gets a 10  // 玩家第二手牌得到10点，花色为红心
        playerSays("S");  // 玩家选择停牌
        initGame();  // 初始化游戏

        // When
        game.play(player);  // 当游戏进行时，玩家进行游戏

        // Then
        assertTrue(out.toString().contains("BUSTED"));  // 确保输出包含“BUSTED”
    }

    @Test
    @DisplayName("play() should allow HIT on split hand until BUST")
    public void playSplitHitUntilBustHand2() {
        // Given
        Player player = new Player(1);  // 给定玩家编号为1
        player.dealCard(new Card(10, Card.Suit.HEARTS));  // 玩家得到一张10点的红心牌
        player.dealCard(new Card(10, Card.Suit.SPADES));  // 玩家得到一张10点的黑桃牌
        playerSays("/"); // 玩家下注
        playerGets(1, Card.Suit.CLUBS); // 玩家获得一张梅花牌
        playerSays("S"); // 玩家选择停止要牌
        playerGets(12, Card.Suit.SPADES); // 玩家获得一张黑桃牌
        playerSays("H"); // 玩家选择再要一张牌
        playerGets(12, Card.Suit.HEARTS); // 玩家获得一张红心牌，爆牌
        playerSays("H"); // 玩家选择再要一张牌
        initGame(); // 初始化游戏

        // 当
        game.play(player); // 游戏进行

        // 然后
        assertTrue(out.toString().contains("BUSTED")); // 断言输出包含"BUSTED"
    }

    @Test
    @DisplayName("play() should allow double down on split hands")
    public void playSplitDoubleDown(){
        // 给定
        // 创建一个名为player的玩家对象，编号为1
        Player player = new Player(1);
        // 设置玩家当前的赌注为100
        player.setCurrentBet(100);
        // 为玩家发一张点数为9，花色为红心的牌
        player.dealCard(new Card(9, Card.Suit.HEARTS));
        // 为玩家发一张点数为9，花色为黑桃的牌
        player.dealCard(new Card(9, Card.Suit.SPADES));

        // 玩家说“/”
        playerSays("/");
        // 玩家获得一张点数为5，花色为方块的牌，第一手牌总点数为14
        playerGets(5, Card.Suit.DIAMONDS);
        // 玩家说“D”
        playerSays("D");
        // 玩家获得一张点数为6，花色为红心的牌，第一手牌总点数为20
        playerGets(6, Card.Suit.HEARTS);
        // 玩家获得一张点数为7，花色为梅花的牌，第二手牌总点数为16
        playerGets(7, Card.Suit.CLUBS);
        // 玩家说“D”
        playerSays("D");
        // 玩家获得一张点数为4，花色为梅花的牌，第二手牌总点数为20
        playerGets(4, Card.Suit.CLUBS);
        // 初始化游戏

        // 当
        game.play(player);

        // 然后
        // 断言：玩家当前赌注应该加倍，为200
        assertAll(
            () -> assertEquals(200, player.getCurrentBet(), "Current bet should be doubled"),
            () -> assertEquals(200, player.getSplitBet(), "Split bet should be doubled"), // 检查分牌赌注是否翻倍
            () -> assertEquals(3, player.getHand(1).size(), "First hand should have exactly three cards"), // 检查第一手牌是否有三张牌
            () -> assertEquals(3, player.getHand(2).size(), "Second hand should have exactly three cards") // 检查第二手牌是否有三张牌
        );
    }

    @Test
    @DisplayName("play() should NOT allow re-splitting first split hand") // 测试play()方法是否不允许重新分牌第一手牌
    public void playSplitTwice(){
        // Given
        Player player = new Player(1); // 创建一个玩家对象
        player.setCurrentBet(100); // 设置玩家当前赌注为100
        player.dealCard(new Card(2, Card.Suit.HEARTS)); // 发给玩家一张2号红心牌
        player.dealCard(new Card(2, Card.Suit.SPADES)); // 发给玩家一张2号黑桃牌

        playerSays("/"); // 玩家选择分牌
        playerGets(13, Card.Suit.CLUBS); // 玩家获得一张13号梅花牌，放入第一手牌
        playerSays("/"); // 不允许重新分牌
        playerSays("S"); // 玩家选择停牌
        playerGets(13, Card.Suit.SPADES); // 玩家获得一张13号黑桃牌，放入第二手牌
        playerSays("S");  // 玩家输入"S"
        initGame();  // 初始化游戏

        // 当
        game.play(player);  // 进行游戏操作

        // 然后
        assertTrue(out.toString().contains("TYPE H, S OR D, PLEASE"));  // 断言输出包含指定的字符串
    }

    @Test
    @DisplayName("play() should NOT allow re-splitting second split hand")
    public void playSplitTwiceHand2(){
        // 给定
        Player player = new Player(1);  // 创建玩家对象
        player.setCurrentBet(100);  // 设置玩家当前赌注
        player.dealCard(new Card(10, Card.Suit.HEARTS));  // 给玩家发一张牌
        player.dealCard(new Card(10, Card.Suit.SPADES));  // 给玩家发一张牌

        playerSays("/");  // 玩家输入"/"
        playerGets(13, Card.Suit.CLUBS); // First hand  // 玩家获得一张点数为13，花色为梅花的牌，作为第一手牌
        playerSays("S");  // 玩家选择停牌
        playerGets(13, Card.Suit.SPADES); // Second hand  // 玩家获得一张点数为13，花色为黑桃的牌，作为第二手牌
        playerSays("/"); // Not allowed  // 玩家选择的操作不被允许
        playerSays("S");  // 玩家选择停牌
        initGame();  // 初始化游戏

        // When
        game.play(player);  // 当游戏进行时，玩家进行操作

        // Then
        assertTrue(out.toString().contains("TYPE H, S OR D, PLEASE"));  // 断言输出的字符串包含"TYPE H, S OR D, PLEASE"
    }

    @Test
    @DisplayName("evaluateRound() should total both hands when split")  // 测试用例名称
    public void evaluateRoundWithSplitHands(){  // 评估分牌后的回合
        // Given
        Player dealer = new Player(0); //Dealer  // 创建一个庄家玩家，编号为0
        dealer.dealCard(new Card(1, Card.Suit.HEARTS));  // 庄家发一张点数为1，花色为红心的牌给玩家
        dealer.dealCard(new Card(1, Card.Suit.SPADES)); // 庄家发一张黑桃1的牌给玩家

        Player player = new Player(1); // 创建一个玩家对象，编号为1
        player.recordRound(200); // 记录玩家的初始总数为200
        player.setCurrentBet(50); // 设置玩家当前的赌注为50
        player.dealCard(new Card(1, Card.Suit.HEARTS)); // 玩家获得一张红心1的牌
        player.dealCard(new Card(1, Card.Suit.SPADES)); // 玩家获得一张黑桃1的牌
        
        playerSays("/"); // 玩家说出动作"/"
        playerGets(13, Card.Suit.CLUBS); // 玩家获得一张梅花13的牌，表示第一手牌
        playerSays("S"); // 玩家说出动作"S"
        playerGets(13, Card.Suit.SPADES); // 玩家获得一张黑桃13的牌，表示第二手牌
        playerSays("S"); // 玩家说出动作"S"
        initGame(); // 初始化游戏

        // 当
        game.play(player); // 游戏进行玩家操作
        game.evaluateRound(Arrays.asList(player), dealer); // 游戏评估本轮结果

        // 然后
        assertAll(
            () -> assertTrue(out.toString().contains("PLAYER 1  WINS    100 TOTAL= 300")), // 断言输出结果包含特定文本
            () -> assertTrue(out.toString().contains("DEALER'S TOTAL= -100")) // 断言输出结果包含特定文本
        );
    }

    @Test
    @DisplayName("evaluateRound() should total add twice insurance bet") // 测试用例的名称
    public void evaluateRoundWithInsurance(){ // 测试方法
        // Given
        Player dealer = new Player(0); // 创建一个玩家对象，代表庄家
        dealer.dealCard(new Card(10, Card.Suit.HEARTS)); // 给庄家发一张牌
        dealer.dealCard(new Card(1, Card.Suit.SPADES)); // 给庄家发一张牌

        Player player = new Player(1); // 创建一个玩家对象
        player.setCurrentBet(50); // 设置玩家的当前赌注
        player.setInsuranceBet(10); // 设置玩家的保险赌注
        player.dealCard(new Card(2, Card.Suit.HEARTS)); // 给玩家发一张牌
        player.dealCard(new Card(1, Card.Suit.SPADES)); // 给玩家发一张牌
        initGame(); // 初始化游戏
        // 当
        game.evaluateRound(Arrays.asList(player), dealer);

        // 然后
        // 失去当前赌注（50），并赢得2*10，总共-30
        assertAll(
            () -> assertTrue(out.toString().contains("PLAYER 1 LOSES     30 TOTAL= -30")),
            () -> assertTrue(out.toString().contains("DEALER'S TOTAL= 30"))
        );
    }

    @Test
    @DisplayName("evaluateRound() should push with no total change")
    public void evaluateRoundWithPush(){
        // 给定
        Player dealer = new Player(0);
        dealer.dealCard(new Card(10, Card.Suit.HEARTS));
        dealer.dealCard(new Card(8, Card.Suit.SPADES)); 
        // 创建一个玩家对象，编号为1
        Player player = new Player(1);
        // 设置玩家的当前赌注为10
        player.setCurrentBet(10);
        // 为玩家发一张牌，点数为9，花色为红心
        player.dealCard(new Card(9, Card.Suit.HEARTS));
        // 为玩家发一张牌，点数为9，花色为黑桃
        player.dealCard(new Card(9, Card.Suit.SPADES));
        // 初始化游戏
        initGame();

        // 当（庄家和玩家都有19点）
        game.evaluateRound(Arrays.asList(player), dealer);

        // 然后
        assertAll(
            // 断言输出包含"PLAYER 1 PUSHES       TOTAL= 0"
            () -> assertTrue(out.toString().contains("PLAYER 1 PUSHES       TOTAL= 0")),
            // 断言输出包含"DEALER'S TOTAL= 0"
            () -> assertTrue(out.toString().contains("DEALER'S TOTAL= 0"))
        );
    }

    @Test
    @DisplayName("shouldPlayDealer() return false when players bust")
    public void shouldPlayDealerBust(){
        // 给定
        // 创建一个名为 player 的玩家对象，编号为 1
        Player player = new Player(1);
        // 为玩家发一张点数为 10，花色为 SPADES 的牌
        player.dealCard(new Card(10, Card.Suit.SPADES));
        // 为玩家发一张点数为 10，花色为 SPADES 的牌
        player.dealCard(new Card(10, Card.Suit.SPADES));
        // 玩家分牌
        player.split();
        // 为玩家第一手发一张点数为 5，花色为 SPADES 的牌
        player.dealCard(new Card(5, Card.Suit.SPADES));
        // 为玩家第一手发一张点数为 8，花色为 SPADES 的牌
        player.dealCard(new Card(8, Card.Suit.SPADES));//First hand Busted

        // 为玩家第二手发一张点数为 5，花色为 SPADES 的牌
        player.dealCard(new Card(5, Card.Suit.SPADES),2);
        // 为玩家第二手发一张点数为 8，花色为 SPADES 的牌
        player.dealCard(new Card(8, Card.Suit.SPADES),2);//Second hand Busted

        // 创建一个名为 playerTwo 的玩家对象，编号为 2
        Player playerTwo = new Player(2);
        // 为玩家Two发一张点数为 7，花色为 HEARTS 的牌
        playerTwo.dealCard(new Card(7, Card.Suit.HEARTS));
        // 为玩家Two发一张点数为 8，花色为 HEARTS 的牌
        playerTwo.dealCard(new Card(8, Card.Suit.HEARTS));
        // 为玩家Two发一张点数为 9，花色为 HEARTS 的牌
        playerTwo.dealCard(new Card(9, Card.Suit.HEARTS));
        // 初始化游戏

        // 当
        boolean result = game.shouldPlayDealer(Arrays.asList(player,playerTwo));

        // 然后
        assertFalse(result); // 断言结果为假

    }

    @Test
    @DisplayName("shouldPlayDealer() return false when players bust")
    public void ShouldPlayer(){
        // Given
        Player player = new Player(1); // 创建玩家对象
        player.dealCard(new Card(10, Card.Suit.SPADES)); // 玩家发一张牌
        player.dealCard(new Card(10, Card.Suit.SPADES)); // 玩家再发一张牌
        player.split(); // 玩家分牌
        player.dealCard(new Card(5, Card.Suit.SPADES)); // 玩家第一手再发一张牌
        player.dealCard(new Card(8, Card.Suit.SPADES)); // 玩家第一手再发一张牌，爆牌

        player.dealCard(new Card(5, Card.Suit.SPADES),2); // 玩家第二手再发一张牌
        player.dealCard(new Card(8, Card.Suit.SPADES),2); // 玩家第二手再发一张牌，爆牌

        Player playerTwo = new Player(2); // 创建第二个玩家对象
        playerTwo.dealCard(new Card(7, Card.Suit.HEARTS)); // 第二个玩家发一张牌
        playerTwo.dealCard(new Card(8, Card.Suit.HEARTS)); // 第二个玩家再发一张牌
        playerTwo.dealCard(new Card(9, Card.Suit.HEARTS)); // 给玩家二发一张牌，牌面为9，花色为红心
        initGame(); // 初始化游戏

        // 当
        boolean result = game.shouldPlayDealer(Arrays.asList(player,playerTwo)); // 调用游戏对象的shouldPlayDealer方法，传入玩家和玩家二的列表作为参数，将结果保存在result变量中

        // 然后
        assertFalse(result); // 断言result为false
    }

    @Test
    @DisplayName("shouldPlayDealer() return true when player has non-natural blackjack")
    public void shouldPlayDealerNonNaturalBlackjack(){
        // 给定
        Player player = new Player(1); // 创建一个ID为1的玩家对象
        player.dealCard(new Card(5, Card.Suit.SPADES)); // 给玩家发一张牌，牌面为5，花色为黑桃
        player.dealCard(new Card(6, Card.Suit.DIAMONDS)); // 给玩家发一张牌，牌面为6，花色为方块
        player.dealCard(new Card(10, Card.Suit.SPADES)); // 给玩家发一张牌，牌面为10，花色为黑桃

        initGame(); // 初始化游戏
        // 当
        boolean result = game.shouldPlayDealer(Arrays.asList(player));

        // 然后        
        assertTrue(result);
    }

    @Test
    @DisplayName("shouldPlayDealer() return true when player doesn't have blackjack")
    public void shouldPlayDealerNonBlackjack(){
        // 给定
        Player player = new Player(1);
        player.dealCard(new Card(10, Card.Suit.SPADES));
        player.dealCard(new Card(6, Card.Suit.DIAMONDS));
        initGame();

        // 当 
        boolean result = game.shouldPlayDealer(Arrays.asList(player));
// 确保 playDealer() 方法返回 true
assertTrue(result);
```

```
// 对 playDealerLessThanSeventeen() 方法进行测试
@Test
@DisplayName("playDealer() should DRAW on less than 17 intial deal")
public void playDealerLessThanSeventeen(){
    // 初始化一个玩家对象作为庄家
    Player dealer = new Player(0);
    // 给庄家发两张牌
    dealer.dealCard(new Card(10, Card.Suit.SPADES));
    dealer.dealCard(new Card(6, Card.Suit.DIAMONDS));
    // 给玩家发一张牌
    playerGets(11, Card.Suit.DIAMONDS);
    // 初始化游戏
    initGame();

    // 庄家进行操作
    game.playDealer(dealer);

    // 确保输出包含 "DRAWS"
    assertTrue(out.toString().contains("DRAWS"));
}
        assertTrue(out.toString().contains("BUSTED"));
    }

    @Test
    @DisplayName("playDealer() should stay on more than 17 intial deal")
    public void playDealerMoreThanSeventeen(){
        // 给定
        // 创建一个玩家对象作为庄家，初始点数为0
        Player dealer = new Player(0);
        // 给庄家发两张牌，一张黑桃10，一张方块8
        dealer.dealCard(new Card(10, Card.Suit.SPADES));
        dealer.dealCard(new Card(8, Card.Suit.DIAMONDS));
        // 初始化游戏

        // 当
        // 庄家进行游戏

        // 然后
        // 断言输出结果不包含"DRAWS"
        assertFalse(out.toString().contains("DRAWS"));
        // 断言输出结果不包含"BUSTED"
        assertFalse(out.toString().contains("BUSTED"));
        // 断言输出结果包含"---TOTAL IS"
        assertTrue(out.toString().contains("---TOTAL IS"));
    }
# 关闭 ZIP 对象
zip.close()
```