# `basic-computer-games\10_Blackjack\java\test\GameTest.java`

```
# 导入测试相关的包
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
# 导入断言相关的包
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
# 导入IO相关的包
import java.io.EOFException;
import java.io.StringReader;
import java.io.StringWriter;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

# 定义测试类
public class GameTest {

    # 定义测试所需的变量
    private StringReader in;
    private StringWriter out;
    private Game game;
    private StringBuilder playerActions;
    private LinkedList<Card> cards;

    # 在每个测试方法执行前重置IO
    @BeforeEach
    public void resetIo() {
        in = null;
        out = null;
        game = null;
        playerActions = new StringBuilder();
        cards = new LinkedList<>();
    }

    # 定义玩家获得牌的方法
    private void playerGets(int value, Card.Suit suit) {
        cards.add(new Card(value, suit));
    }

    # 定义玩家说的动作方法
    private void playerSays(String action) {
        playerActions.append(action).append(System.lineSeparator());
    }

    # 初始化游戏
    private void initGame() {
        # 打印游戏的输入和牌
        System.out.printf("Running game with input: %s\tand cards: %s\n",playerActions.toString(), cards);
        in = new StringReader(playerActions.toString());
        out = new StringWriter();
        UserIo userIo = new UserIo(in, out);
        Deck deck = new Deck((c) -> cards);
        game = new Game(deck, userIo);
    }

    # 在每个测试方法执行后打印输出
    @AfterEach
    private void printOutput() {
        System.out.println(out.toString());
    }

    # 定义测试方法
    @Test
    public void shouldQuitOnCtrlD() {
        // 当用户输入CTRL+D时，游戏应该退出
        // Given
        playerSays("\u2404"); // U+2404 is "End of Transmission" sent by CTRL+D (or CTRL+Z on Windows)
        initGame();

        // When
        // 断言游戏运行时会抛出UncheckedIOException异常
        Exception e = assertThrows(UncheckedIOException.class, game::run);

        // Then
        // 断言异常的原因是EOFException
        assertTrue(e.getCause() instanceof EOFException);
        // 断言异常消息为"!END OF INPUT"
        assertEquals("!END OF INPUT", e.getMessage());
    }

    @Test
    @DisplayName("collectInsurance() should not prompt on N")
    public void collectInsuranceNo(){
        // 当用户输入"N"时，collectInsurance()方法不应该提示
        // Given
        List<Player> players = Collections.singletonList(new Player(1));
        playerSays("N");
        initGame();

        // When
        // 调用collectInsurance()方法
        game.collectInsurance(players);

        // Then
        // 断言输出内容不包含"ANY INSURANCE"
        assertAll(
            () -> assertTrue(out.toString().contains("ANY INSURANCE")),
            () -> assertFalse(out.toString().contains("INSURANCE BETS"))
        );
    }

    @Test
    @DisplayName("collectInsurance() should collect on Y")
    public void collectInsuranceYes(){
        // 当用户输入"Y"时，collectInsurance()方法应该进行收集
        // Given
        List<Player> players = Collections.singletonList(new Player(1));
        players.get(0).setCurrentBet(100);
        playerSays("Y");
        playerSays("50");
        initGame();

        // When
        // 调用collectInsurance()方法
        game.collectInsurance(players);

        // Then
        // 断言输出内容包含"ANY INSURANCE"
        // 断言输出内容包含"INSURANCE BETS"
        // 断言玩家的保险赌注为50
        assertAll(
            () -> assertTrue(out.toString().contains("ANY INSURANCE")),
            () -> assertTrue(out.toString().contains("INSURANCE BETS")),
            () -> assertEquals(50, players.get(0).getInsuranceBet())
        );
    }

    @Test
    @DisplayName("collectInsurance() should not allow more than 50% of current bet")
    public void collectInsuranceYesTooMuch(){
        // 初始化玩家列表，只包含一个玩家，当前赌注为100
        List<Player> players = Collections.singletonList(new Player(1));
        players.get(0).setCurrentBet(100);
        // 玩家选择买保险
        playerSays("Y");
        // 玩家输入保险赌注为51
        playerSays("51");
        // 玩家输入保险赌注为50
        playerSays("50");
        // 初始化游戏
        initGame();

        // 执行游戏的买保险操作
        game.collectInsurance(players);

        // 断言：玩家的保险赌注应为50，输出结果包含指定字符串
        assertAll(
            () -> assertEquals(50, players.get(0).getInsuranceBet()),
            () -> assertTrue(out.toString().contains("# 1 ? # 1 ?"))
        );
    }

    @Test
    @DisplayName("collectInsurance() should not allow negative bets")
    public void collectInsuranceYesNegative(){
        // 初始化玩家列表，只包含一个玩家，当前赌注为100
        List<Player> players = Collections.singletonList(new Player(1));
        players.get(0).setCurrentBet(100);
        // 玩家选择买保险
        playerSays("Y");
        // 玩家输入负数保险赌注为-1
        playerSays("-1");
        // 玩家输入正数保险赌注为1
        playerSays("1");
        // 初始化游戏
        initGame();

        // 执行游戏的买保险操作
        game.collectInsurance(players);

        // 断言：玩家的保险赌注应为1，输出结果包含指定字符串
        assertAll(
            () -> assertEquals(1, players.get(0).getInsuranceBet()),
            () -> assertTrue(out.toString().contains("# 1 ? # 1 ?"))
        );
    }

    @Test
    @DisplayName("collectInsurance() should prompt all players")
    public void collectInsuranceYesTwoPlayers(){
        // 初始化玩家列表，包含两个玩家，当前赌注都为100
        List<Player> players = Arrays.asList(
            new Player(1),
            new Player(2)
        );
        players.get(0).setCurrentBet(100);
        players.get(1).setCurrentBet(100);

        // 玩家选择买保险
        playerSays("Y");
        // 第一个玩家输入保险赌注为50
        playerSays("50");
        // 第二个玩家输入保险赌注为25
        playerSays("25");
        // 初始化游戏
        initGame();

        // 执行游戏的买保险操作
        game.collectInsurance(players);

        // 断言：第一个玩家的保险赌注应为50，第二个玩家的保险赌注应为25，输出结果包含指定字符串
        assertAll(
            () -> assertEquals(50, players.get(0).getInsuranceBet()),
            () -> assertEquals(25, players.get(1).getInsuranceBet()),
            () -> assertTrue(out.toString().contains("# 1 ? # 2 ?"))
        );
    }

    @Test
    @DisplayName("play() should end on STAY")
    public void playEndOnStay(){
        // 初始化玩家对象，玩家编号为1
        Player player = new Player(1);
        // 为玩家发牌，分别为梅花3和黑桃2
        player.dealCard(new Card(3, Card.Suit.CLUBS));
        player.dealCard(new Card(2, Card.Suit.SPADES));
        // 玩家选择停牌
        playerSays("S"); // "I also like to live dangerously."
        // 初始化游戏
        initGame();

        // 玩家进行游戏
        game.play(player);

        // 断言输出结果以"PLAYER 1 ? TOTAL IS 5"开头
        assertTrue(out.toString().startsWith("PLAYER 1 ? TOTAL IS 5"));
    }

    @Test
    @DisplayName("play() should allow HIT until BUST")
    public void playHitUntilBust() {
        // 初始化玩家对象，玩家编号为1
        Player player = new Player(1);
        // 为玩家发牌，分别为红心10和黑桃10
        player.dealCard(new Card(10, Card.Suit.HEARTS));
        player.dealCard(new Card(10, Card.Suit.SPADES));

        // 玩家选择要牌
        playerSays("H");
        // 玩家获得一张黑桃牌，总点数为20
        playerGets(1, Card.Suit.SPADES); // 20
        // 玩家选择要牌
        playerSays("H");
        // 玩家获得一张红心牌，总点数为21
        playerGets(1, Card.Suit.HEARTS); // 21
        // 玩家选择要牌
        playerSays("H");
        // 玩家获得一张梅花牌，总点数为22，爆牌
        playerGets(1, Card.Suit.CLUBS); // 22 - D'oh!
        // 初始化游戏
        initGame();

        // 玩家进行游戏
        game.play(player);

        // 断言输出结果包含"BUSTED"
        assertTrue(out.toString().contains("BUSTED"));
    }

    @Test
    @DisplayName("Should allow double down on initial turn")
    public void playDoubleDown(){
        // 初始化玩家对象，玩家编号为1
        Player player = new Player(1);
        // 设置玩家当前下注额为100
        player.setCurrentBet(100);
        // 为玩家发牌，分别为红心10和黑桃4
        player.dealCard(new Card(10, Card.Suit.HEARTS));
        player.dealCard(new Card(4, Card.Suit.SPADES));

        // 玩家选择双倍下注
        playerSays("D");
        // 玩家获得一张黑桃牌
        playerGets(7, Card.Suit.SPADES);
        // 初始化游戏
        initGame();

        // 玩家进行游戏
        game.play(player);

        // 断言玩家当前下注额为200
        assertTrue(player.getCurrentBet() == 200);
        // 断言玩家手牌数量为3
        assertTrue(player.getHand().size() == 3);
    }

    @Test
    @DisplayName("Should NOT allow double down after initial deal")
    public void playDoubleDownLate(){
        // 初始化玩家对象，设置当前赌注为100
        Player player = new Player(1);
        player.setCurrentBet(100);
        // 为玩家发牌
        player.dealCard(new Card(10, Card.Suit.HEARTS));
        player.dealCard(new Card(2, Card.Suit.SPADES));

        // 玩家选择命令为"H"
        playerSays("H");
        // 玩家获得一张牌，点数为7，花色为黑桃
        playerGets(7, Card.Suit.SPADES);
        // 玩家选择命令为"D"
        playerSays("D");
        // 玩家选择命令为"S"
        playerSays("S");
        // 初始化游戏

        // 玩家进行游戏
        game.play(player);

        // 断言输出结果包含"TYPE H, OR S, PLEASE"
        assertTrue(out.toString().contains("TYPE H, OR S, PLEASE"));
    }

    @Test
    @DisplayName("play() should end on STAY after split")
    public void playSplitEndOnStay(){
        // 初始化玩家对象
        Player player = new Player(1);
        // 为玩家发牌
        player.dealCard(new Card(1, Card.Suit.CLUBS));
        player.dealCard(new Card(1, Card.Suit.SPADES));

        // 玩家选择命令为"/"
        playerSays("/");
        // 玩家获得一张牌，点数为2，花色为黑桃（第一手牌）
        playerGets(2, Card.Suit.SPADES); // First hand
        // 玩家选择命令为"S"
        playerSays("S");
        // 玩家获得一张牌，点数为2，花色为黑桃（第二手牌）
        playerGets(2, Card.Suit.SPADES); // Second hand
        // 玩家选择命令为"S"
        playerSays("S");
        // 初始化游戏

        // 玩家进行游戏
        game.play(player);

        // 断言输出结果包含"FIRST HAND RECEIVES"和"SECOND HAND RECEIVES"
        assertTrue(out.toString().contains("FIRST HAND RECEIVES"));
        assertTrue(out.toString().contains("SECOND HAND RECEIVES"));
    }

    @Test
    @DisplayName("play() should allow HIT until BUST after split")
    public void playSplitHitUntilBust() {
        // 初始化玩家对象
        Player player = new Player(1);
        // 为玩家发牌
        player.dealCard(new Card(10, Card.Suit.HEARTS));
        player.dealCard(new Card(10, Card.Suit.SPADES));

        // 玩家选择命令为"/"
        playerSays("/");
        // 玩家获得一张牌，点数为12，花色为黑桃（第一手牌）
        playerGets(12, Card.Suit.SPADES); // First hand has 20
        // 玩家选择命令为"H"
        playerSays("H");
        // 玩家获得一张牌，点数为12，花色为红桃（第一手牌爆牌）
        playerGets(12, Card.Suit.HEARTS); // First hand busted
        // 玩家获得一张牌，点数为10，花色为红桃（第二手牌）
        playerGets(10, Card.Suit.HEARTS); // Second hand gets a 10
        // 玩家选择命令为"S"
        playerSays("S");
        // 初始化游戏

        // 玩家进行游戏
        game.play(player);

        // 断言输出结果包含"BUSTED"
        assertTrue(out.toString().contains("BUSTED"));
    }

    @Test
    @DisplayName("play() should allow HIT on split hand until BUST")
    public void playSplitHitUntilBustHand2() {
        // 定义一个方法，模拟玩家进行分牌后继续要牌直到爆牌的情况
        // Given
        // 创建一个玩家对象
        Player player = new Player(1);
        // 给玩家发两张牌，分别是10号红心和10号黑桃
        player.dealCard(new Card(10, Card.Suit.HEARTS));
        player.dealCard(new Card(10, Card.Suit.SPADES));

        // 模拟玩家操作
        playerSays("/");
        playerGets(1, Card.Suit.CLUBS); // 第一手牌是21点
        playerSays("S");
        playerGets(12, Card.Suit.SPADES); // 第二手牌是20点
        playerSays("H");
        playerGets(12, Card.Suit.HEARTS); // 爆牌
        playerSays("H");
        initGame();

        // When
        // 调用游戏进行玩家操作
        game.play(player);

        // Then
        // 断言输出结果包含"BUSTED"
        assertTrue(out.toString().contains("BUSTED"));
    }

    @Test
    @DisplayName("play() should allow double down on split hands")
    public void playSplitDoubleDown(){
        // 定义一个测试用例，测试玩家在分牌后是否可以加倍下注
        // Given
        // 创建一个玩家对象
        Player player = new Player(1);
        // 设置玩家当前下注金额为100
        player.setCurrentBet(100);
        // 给玩家发两张牌，分别是9号红心和9号黑桃
        player.dealCard(new Card(9, Card.Suit.HEARTS));
        player.dealCard(new Card(9, Card.Suit.SPADES));

        // 模拟玩家操作
        playerSays("/");
        playerGets(5, Card.Suit.DIAMONDS); // 第一手牌是14点
        playerSays("D");
        playerGets(6, Card.Suit.HEARTS); // 第一手牌是20点
        playerGets(7, Card.Suit.CLUBS); // 第二手牌是16点
        playerSays("D");
        playerGets(4, Card.Suit.CLUBS); // 第二手牌是20点
        initGame();

        // When
        // 调用游戏进行玩家操作
        game.play(player);

        // Then
        // 使用断言同时验证多个条件
        assertAll(
            () -> assertEquals(200, player.getCurrentBet(), "当前下注金额应该加倍"),
            () -> assertEquals(200, player.getSplitBet(), "分牌后的下注金额应该加倍"),
            () -> assertEquals(3, player.getHand(1).size(), "第一手牌应该有三张牌"),
            () -> assertEquals(3, player.getHand(2).size(), "第二手牌应该有三张牌")
        );
    }

    @Test
    @DisplayName("play() should NOT allow re-splitting first split hand")
    public void playSplitTwice(){
        // 初始化玩家对象
        Player player = new Player(1);
        // 设置当前赌注为100
        player.setCurrentBet(100);
        // 发两张牌给玩家
        player.dealCard(new Card(2, Card.Suit.HEARTS));
        player.dealCard(new Card(2, Card.Suit.SPADES);

        // 玩家选择分牌
        playerSays("/");
        // 玩家获得13点，花色为梅花，表示第一手牌
        playerGets(13, Card.Suit.CLUBS); // First hand
        // 玩家再次选择分牌，不允许
        playerSays("/"); // Not allowed
        // 玩家选择停牌
        playerSays("S");
        // 玩家获得13点，花色为黑桃，表示第二手牌
        playerGets(13, Card.Suit.SPADES); // Second hand
        // 初始化游戏
        initGame();

        // 游戏进行
        game.play(player);

        // 断言输出包含"TYPE H, S OR D, PLEASE"
        assertTrue(out.toString().contains("TYPE H, S OR D, PLEASE"));
    }

    @Test
    @DisplayName("play() should NOT allow re-splitting second split hand")
    public void playSplitTwiceHand2(){
        // 初始化玩家对象
        Player player = new Player(1);
        // 设置当前赌注为100
        player.setCurrentBet(100);
        // 发两张牌给玩家
        player.dealCard(new Card(10, Card.Suit.HEARTS));
        player.dealCard(new Card(10, Card.Suit.SPADES));

        // 玩家选择分牌
        playerSays("/");
        // 玩家获得13点，花色为梅花，表示第一手牌
        playerGets(13, Card.Suit.CLUBS); // First hand
        // 玩家选择停牌
        playerSays("S");
        // 玩家获得13点，花色为黑桃，表示第二手牌
        playerGets(13, Card.Suit.SPADES); // Second hand
        // 玩家再次选择分牌，不允许
        playerSays("/"); // Not allowed
        // 玩家选择停牌
        playerSays("S");
        // 初始化游戏
        initGame();

        // 游戏进行
        game.play(player);

        // 断言输出包含"TYPE H, S OR D, PLEASE"
        assertTrue(out.toString().contains("TYPE H, S OR D, PLEASE"));
    }

    @Test
    @DisplayName("evaluateRound() should total both hands when split")
    public void evaluateRoundWithSplitHands(){
        // 定义一个名为 evaluateRoundWithSplitHands 的方法，用于评估玩家分牌后的局面
        // Given
        Player dealer = new Player(0); // 创建一个代表庄家的对象
        dealer.dealCard(new Card(1, Card.Suit.HEARTS)); // 给庄家发一张红心1
        dealer.dealCard(new Card(1, Card.Suit.SPADES)); // 给庄家发一张黑桃1

        Player player = new Player(1); // 创建一个代表玩家的对象
        player.recordRound(200);// 设置玩家的初始总数为200
        player.setCurrentBet(50); // 设置玩家的当前赌注为50
        player.dealCard(new Card(1, Card.Suit.HEARTS)); // 给玩家发一张红心1
        player.dealCard(new Card(1, Card.Suit.SPADES)); // 给玩家发一张黑桃1
        
        playerSays("/"); // 玩家选择分牌
        playerGets(13, Card.Suit.CLUBS); // 玩家获得一张梅花13，作为第一手牌
        playerSays("S"); // 玩家选择停牌
        playerGets(13, Card.Suit.SPADES); // 玩家获得一张黑桃13，作为第二手牌
        playerSays("S"); // 玩家选择停牌
        initGame(); // 初始化游戏

        // When
        game.play(player); // 玩家进行游戏
        game.evaluateRound(Arrays.asList(player), dealer); // 评估局面，传入玩家和庄家的列表

        // Then
        assertAll(
            () -> assertTrue(out.toString().contains("PLAYER 1  WINS    100 TOTAL= 300")), // 断言玩家1赢得100，总数为300
            () -> assertTrue(out.toString().contains("DEALER'S TOTAL= -100")) // 断言庄家的总数为-100
        );
    }

    @Test
    @DisplayName("evaluateRound() should total add twice insurance bet")
    public void evaluateRoundWithInsurance(){
        // Given
        Player dealer = new Player(0); // 创建一个代表庄家的对象
        dealer.dealCard(new Card(10, Card.Suit.HEARTS)); // 给庄家发一张红心10
        dealer.dealCard(new Card(1, Card.Suit.SPADES)); // 给庄家发一张黑桃1

        Player player = new Player(1); // 创建一个代表玩家的对象
        player.setCurrentBet(50); // 设置玩家的当前赌注为50
        player.setInsuranceBet(10); // 设置玩家的保险赌注为10
        player.dealCard(new Card(2, Card.Suit.HEARTS)); // 给玩家发一张红心2
        player.dealCard(new Card(1, Card.Suit.SPADES)); // 给玩家发一张黑桃1
        initGame(); // 初始化游戏

        // When
        game.evaluateRound(Arrays.asList(player), dealer); // 评估局面，传入玩家和庄家的列表

        // Then
        // Loses current bet (50) and wins 2*10 for total -30
        assertAll(
            () -> assertTrue(out.toString().contains("PLAYER 1 LOSES     30 TOTAL= -30")), // 断言玩家1输了30，总数为-30
            () -> assertTrue(out.toString().contains("DEALER'S TOTAL= 30")) // 断言庄家的总数为30
        );
    }

    @Test
    @DisplayName("evaluateRound() should push with no total change")
    public void evaluateRoundWithPush(){
        // 初始化一个庄家对象
        Player dealer = new Player(0);
        // 给庄家发一张牌（10号红心）
        dealer.dealCard(new Card(10, Card.Suit.HEARTS));
        // 给庄家发一张牌（8号黑桃）
        dealer.dealCard(new Card(8, Card.Suit.SPADES)); 

        // 初始化一个玩家对象
        Player player = new Player(1);
        // 设置玩家当前的赌注为10
        player.setCurrentBet(10);
        // 给玩家发一张牌（9号红心）
        player.dealCard(new Card(9, Card.Suit.HEARTS));
        // 给玩家发一张牌（9号黑桃）
        player.dealCard(new Card(9, Card.Suit.SPADES));
        // 初始化游戏
        initGame();

        // 当（庄家和玩家都有19点）
        game.evaluateRound(Arrays.asList(player), dealer);

        // 然后        
        assertAll(
            () -> assertTrue(out.toString().contains("PLAYER 1 PUSHES       TOTAL= 0")),
            () -> assertTrue(out.toString().contains("DEALER'S TOTAL= 0"))
        );
    }

    @Test
    @DisplayName("shouldPlayDealer() return false when players bust")
    public void shouldPlayDealerBust(){
        // 初始化一个玩家对象
        Player player = new Player(1);
        // 给玩家发一张牌（10号黑桃）
        player.dealCard(new Card(10, Card.Suit.SPADES));
        // 给玩家发一张牌（10号黑桃）
        player.dealCard(new Card(10, Card.Suit.SPADES));
        // 分牌
        player.split();
        // 给第一手牌发一张牌（5号黑桃）
        player.dealCard(new Card(5, Card.Suit.SPADES));
        // 给第一手牌发一张牌（8号黑桃）（第一手牌爆牌）
        player.dealCard(new Card(8, Card.Suit.SPADES));//First hand Busted

        // 给第二手牌发一张牌（5号黑桃）
        player.dealCard(new Card(5, Card.Suit.SPADES),2);
        // 给第二手牌发一张牌（8号黑桃）（第二手牌爆牌）
        player.dealCard(new Card(8, Card.Suit.SPADES),2);//Second hand Busted

        // 初始化一个第二个玩家对象
        Player playerTwo = new Player(2);
        // 给第二个玩家发一张牌（7号红心）
        playerTwo.dealCard(new Card(7, Card.Suit.HEARTS));
        // 给第二个玩家发一张牌（8号红心）
        playerTwo.dealCard(new Card(8, Card.Suit.HEARTS));
        // 给第二个玩家发一张牌（9号红心）
        playerTwo.dealCard(new Card(9, Card.Suit.HEARTS));
        // 初始化游戏
        initGame();

        // 当 
        boolean result = game.shouldPlayDealer(Arrays.asList(player,playerTwo));

        // 然后        
        assertFalse(result);
    }

    @Test
    @DisplayName("shouldPlayDealer() return false when players bust")
    public void ShouldPlayer(){
        // 初始化玩家对象
        Player player = new Player(1);
        // 为玩家发牌
        player.dealCard(new Card(10, Card.Suit.SPADES));
        player.dealCard(new Card(10, Card.Suit.SPADES));
        // 玩家分牌
        player.split();
        // 为第一手牌发牌
        player.dealCard(new Card(5, Card.Suit.SPADES));
        player.dealCard(new Card(8, Card.Suit.SPADES));//First hand Busted
        // 为第二手牌发牌
        player.dealCard(new Card(5, Card.Suit.SPADES),2);
        player.dealCard(new Card(8, Card.Suit.SPADES),2);//Second hand Busted

        // 初始化第二个玩家对象
        Player playerTwo = new Player(2);
        // 为第二个玩家发牌
        playerTwo.dealCard(new Card(7, Card.Suit.HEARTS));
        playerTwo.dealCard(new Card(8, Card.Suit.HEARTS));
        playerTwo.dealCard(new Card(9, Card.Suit.HEARTS));
        // 初始化游戏
        initGame();

        // 调用 shouldPlayDealer 方法判断是否需要玩家继续游戏
        boolean result = game.shouldPlayDealer(Arrays.asList(player,playerTwo));

        // 断言结果为 false
        assertFalse(result);
    }

    @Test
    @DisplayName("shouldPlayDealer() return true when player has non-natural blackjack")
    public void shouldPlayDealerNonNaturalBlackjack(){
        // 初始化玩家对象
        Player player = new Player(1);
        // 为玩家发牌
        player.dealCard(new Card(5, Card.Suit.SPADES));
        player.dealCard(new Card(6, Card.Suit.DIAMONDS));
        player.dealCard(new Card(10, Card.Suit.SPADES));

        // 初始化游戏
        initGame();

        // 调用 shouldPlayDealer 方法判断是否需要玩家继续游戏
        boolean result = game.shouldPlayDealer(Arrays.asList(player));

        // 断言结果为 true
        assertTrue(result);
    }

    @Test
    @DisplayName("shouldPlayDealer() return true when player doesn't have blackjack")
    public void shouldPlayDealerNonBlackjack(){
        // 初始化玩家对象
        Player player = new Player(1);
        // 为玩家发牌
        player.dealCard(new Card(10, Card.Suit.SPADES));
        player.dealCard(new Card(6, Card.Suit.DIAMONDS);
        // 初始化游戏
        initGame();

        // 调用 shouldPlayDealer 方法判断是否需要玩家继续游戏
        boolean result = game.shouldPlayDealer(Arrays.asList(player));

        // 断言结果为 true
        assertTrue(result);
    }


    @Test
    @DisplayName("playDealer() should DRAW on less than 17 intial deal")
    public void playDealerLessThanSeventeen(){
        // 给定初始条件，创建一个玩家对象作为庄家
        Player dealer = new Player(0);
        // 给庄家发两张牌，一张黑桃10，一张方块6
        dealer.dealCard(new Card(10, Card.Suit.SPADES));
        dealer.dealCard(new Card(6, Card.Suit.DIAMONDS));
        // 玩家获得一张牌，点数为11，花色为方块
        playerGets(11, Card.Suit.DIAMONDS);
        // 初始化游戏
        initGame();

        // 当
        // 庄家进行游戏
        game.playDealer(dealer);

        // 然后
        // 断言输出包含 "DRAWS"
        assertTrue(out.toString().contains("DRAWS"));
        // 断言输出包含 "BUSTED"
        assertTrue(out.toString().contains("BUSTED"));
    }

    @Test
    @DisplayName("playDealer() should stay on more than 17 intial deal")
    public void playDealerMoreThanSeventeen(){
        // 给定初始条件，创建一个玩家对象作为庄家
        Player dealer = new Player(0);
        // 给庄家发两张牌，一张黑桃10，一张方块8
        dealer.dealCard(new Card(10, Card.Suit.SPADES));
        dealer.dealCard(new Card(8, Card.Suit.DIAMONDS));
        // 初始化游戏
        initGame();

        // 当
        // 庄家进行游戏
        game.playDealer(dealer);

        // 然后
        // 断言输出不包含 "DRAWS"
        assertFalse(out.toString().contains("DRAWS"));
        // 断言输出不包含 "BUSTED"
        assertFalse(out.toString().contains("BUSTED"));
        // 断言输出包含 "---TOTAL IS"
        assertTrue(out.toString().contains("---TOTAL IS"));
    }
# 闭合前面的函数定义
```