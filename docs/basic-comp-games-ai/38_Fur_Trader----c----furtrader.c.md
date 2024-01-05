# `38_Fur_Trader\c\furtrader.c`

```
/*
 * Ported from furtrader.bas to ANSI C (C99) by krt@krt.com.au
 * 从 furtrader.bas 转换为 ANSI C (C99) 由 krt@krt.com.au 完成
 *
 * compile with:
 *    gcc -g -Wall -Werror furtrader.c -o furtrader
 * 使用以下命令编译：
 *    gcc -g -Wall -Werror furtrader.c -o furtrader
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


/* Constants */
#define FUR_TYPE_COUNT    4
#define FUR_MINK          0
#define FUR_BEAVER        1
#define FUR_ERMINE        2
#define FUR_FOX           3
/* 定义常量 */
#define FUR_TYPE_COUNT    4  // 毛皮类型数量
#define FUR_MINK          0  // 貂
#define FUR_BEAVER        1  // 海狸
#define FUR_ERMINE        2  // 白鼬
#define FUR_FOX           3  // 狐狸
#define MAX_FURS        190  // 定义最大毛皮数量为190
const char *FUR_NAMES[FUR_TYPE_COUNT] = { "MINK", "BEAVER", "ERMINE", "FOX" };  // 定义毛皮类型的名称数组

#define FORT_TYPE_COUNT 3  // 定义要塞类型的数量为3
#define FORT_MONTREAL   1  // 定义蒙特利尔要塞的编号为1
#define FORT_QUEBEC     2  // 定义魁北克要塞的编号为2
#define FORT_NEWYORK    3  // 定义纽约要塞的编号为3
const char *FORT_NAMES[FORT_TYPE_COUNT] = { "HOCHELAGA (MONTREAL)", "STADACONA (QUEBEC)", "NEW YORK" };  // 定义要塞名称数组

/* Print the words at the specified column */
void printAtColumn( int column, const char *words )  // 定义一个函数，将指定列的单词打印出来
{
    int i;
    for ( i=0; i<column; i++ )  // 循环打印空格，使单词在指定列
        printf( " " );
    printf( "%s\n", words );  // 打印单词
}
/* 用于输出带有换行符的一行文字的简单函数 */
void print( const char *words )
{
    printf( "%s\n", words );
}

/* 展示玩家游戏的介绍信息 */
void showIntroduction()
{
    print( "YOU ARE THE LEADER OF A FRENCH FUR TRADING EXPEDITION IN " );
    print( "1776 LEAVING THE LAKE ONTARIO AREA TO SELL FURS AND GET" );
    print( "SUPPLIES FOR THE NEXT YEAR.  YOU HAVE A CHOICE OF THREE" );
    print( "FORTS AT WHICH YOU MAY TRADE.  THE COST OF SUPPLIES" );
    print( "AND THE AMOUNT YOU RECEIVE FOR YOUR FURS WILL DEPEND" );
    print( "ON THE FORT THAT YOU CHOOSE." );
    print( "" );  // 输出空行
}
 * Prompt the user for input.
 * When input is given, try to convert it to an integer
 * return the integer converted value or 0 on error
 */
int getNumericInput()
{
    int  result = -1;  // 初始化结果为-1
    char buffer[64];   /* somewhere to store user input */  // 用于存储用户输入的地方
    char *endstr;

    while ( result == -1 )  // 当结果为-1时循环
    {
        printf( ">> " );                                 /* prompt the user */  // 提示用户输入
        fgets( buffer, sizeof( buffer ), stdin );        /* read from the console into the buffer */  // 从控制台读取输入到缓冲区
        result = (int)strtol( buffer, &endstr, 10 );     /* only simple error checking */  // 将输入的字符串转换为整数，进行简单的错误检查

        if ( endstr == buffer )                          /* was the string -> integer ok? */  // 检查字符串是否成功转换为整数
            result = -1;  // 如果转换失败，将结果设为-1
    }
    return result;
}
```
这段代码是一个函数的结尾，返回函数的结果。

```
/*
 * Prompt the user for YES/NO input.
 * When input is given, try to work out if it's YES, Yes, yes, Y, etc.
 * And convert to a single upper-case letter
 * Returns a character of 'Y' or 'N'.
 */
char getYesOrNo()
{
    char result = '!';
    char buffer[64];   /* somewhere to store user input */

    while ( !( result == 'Y' || result == 'N' ) )       /* While the answer was not Yes or No */
    {
        print( "ANSWER YES OR NO" );
        printf( ">> " );
```
这段代码是一个函数的定义，函数名为getYesOrNo，返回类型为char。函数内部包含一个while循环，当用户输入不是Y或N时，会一直提示用户输入YES或NO。

```python
bio = BytesIO(open(fname, 'rb').read())
```
这行代码创建了一个BytesIO对象，用于封装以二进制形式读取的文件内容。

```python
zip = zipfile.ZipFile(bio, 'r')
```
这行代码使用BytesIO对象里面的内容创建了一个ZIP对象，用于读取ZIP文件中的内容。

```python
fdict = {n:zip.read(n) for n in zip.namelist()}
```
这行代码遍历ZIP对象中所包含的文件名，然后读取文件数据，最后组成文件名到数据的字典。

```python
zip.close()
```
这行代码关闭了ZIP对象，释放资源。

```python
return fdict
```
这行代码返回了结果字典。
        fgets( buffer, sizeof( buffer ), stdin );            /* 从控制台读取内容到缓冲区 */
        if ( buffer[0] == 'Y' || buffer[0] == 'y' )         // 如果缓冲区的第一个字符是 'Y' 或 'y'
            result = 'Y';                                   // 将结果设置为 'Y'
        else if ( buffer[0] == 'N' || buffer[0] == 'n' )    // 如果缓冲区的第一个字符是 'N' 或 'n'
            result = 'N';                                   // 将结果设置为 'N'
    }

    return result;                                          // 返回结果
}



/*
 * 展示玩家选择的堡垒，获取他们的输入，如果输入是有效的选择（1,2,3），则返回它，否则继续提示用户。
 */
int getFortChoice()
{
    int result = 0;                                         // 初始化结果为 0
    while ( result == 0 )
    {
        # 打印交易毛皮的提示信息
        print( "" );
        print( "YOU MAY TRADE YOUR FURS AT FORT 1, FORT 2," );
        print( "OR FORT 3.  FORT 1 IS FORT HOCHELAGA (MONTREAL)" );
        print( "AND IS UNDER THE PROTECTION OF THE FRENCH ARMY." );
        print( "FORT 2 IS FORT STADACONA (QUEBEC) AND IS UNDER THE" );
        print( "PROTECTION OF THE FRENCH ARMY.  HOWEVER, YOU MUST" );
        print( "MAKE A PORTAGE AND CROSS THE LACHINE RAPIDS." );
        print( "FORT 3 IS FORT NEW YORK AND IS UNDER DUTCH CONTROL." );
        print( "YOU MUST CROSS THROUGH IROQUOIS LAND." );
        print( "ANSWER 1, 2, OR 3." ;

        result = getNumericInput();   /* get input from the player */  # 从玩家获取输入
    }

    return result;  # 返回结果
}
/*
 * Print the description for the fort
 */
void showFortComment( int which_fort )
{
    // Print an empty line
    print( "" );
    // Check if the chosen fort is FORT_MONTREAL
    if ( which_fort == FORT_MONTREAL )
    {
        // Print the description for FORT_MONTREAL
        print( "YOU HAVE CHOSEN THE EASIEST ROUTE.  HOWEVER, THE FORT" );
        print( "IS FAR FROM ANY SEAPORT.  THE VALUE" );
        print( "YOU RECEIVE FOR YOUR FURS WILL BE LOW AND THE COST" );
        print( "OF SUPPLIES HIGHER THAN AT FORTS STADACONA OR NEW YORK." );
    }
    // Check if the chosen fort is FORT_QUEBEC
    else if ( which_fort == FORT_QUEBEC )
    {
        // Print the description for FORT_QUEBEC
        print( "YOU HAVE CHOSEN A HARD ROUTE.  IT IS, IN COMPARSION," );
        print( "HARDER THAN THE ROUTE TO HOCHELAGA BUT EASIER THAN" );
        print( "THE ROUTE TO NEW YORK.  YOU WILL RECEIVE AN AVERAGE VALUE" );
        print( "FOR YOUR FURS AND THE COST OF YOUR SUPPLIES WILL BE AVERAGE." );
    }
    else if ( which_fort == FORT_NEWYORK )
    {
        print( "YOU HAVE CHOSEN THE MOST DIFFICULT ROUTE.  AT" );
        print( "FORT NEW YORK YOU WILL RECEIVE THE HIGHEST VALUE" );
        print( "FOR YOUR FURS.  THE COST OF YOUR SUPPLIES" );
        print( "WILL BE LOWER THAN AT ALL THE OTHER FORTS." );
    }
    else
    {
        printf( "Internal error #1, fort %d does not exist\n", which_fort );
        exit( 1 );  /* you have a bug */
    }
    print( "" );
}
```

这段代码是 C 语言的代码，不是 Python 代码，因此无法为其添加注释。
# 获取毛皮购买数量的函数
def getFursPurchase( int *furs ):
    # 初始化变量 i
    int i;

    # 打印提示信息
    printf( "YOUR %d FURS ARE DISTRIBUTED AMONG THE FOLLOWING\n", FUR_TYPE_COUNT );
    # 打印提示信息
    print( "KINDS OF PELTS: MINK, BEAVER, ERMINE AND FOX." );
    # 打印空行
    print( "" );

    # 循环遍历毛皮类型
    for ( i=0; i<FUR_TYPE_COUNT; i++ ):
        # 打印提示信息，获取用户输入的毛皮数量
        printf( "HOW MANY %s DO YOU HAVE\n", FUR_NAMES[i] );
        furs[i] = getNumericInput();

/*
 * (Re)Set the player's inventory to zero
 */
void zeroInventory( int *player_fur_count )
{
    int i;  // 声明一个整型变量 i
    for ( i=0; i<FUR_TYPE_COUNT; i++ )  // 循环遍历 player_fur_count 数组
    {
        player_fur_count[i] = 0;  // 将 player_fur_count 数组中的元素全部置为 0
    }
}


/*
 * Tally the player's inventory
 */
int sumInventory( int *player_fur_count )
{
    int result = 0;  // 声明一个整型变量 result，并初始化为 0
    int i;  // 声明一个整型变量 i
    for ( i=0; i<FUR_TYPE_COUNT; i++ )  // 循环遍历 player_fur_count 数组
    {
        result += player_fur_count[i];  // 将 player_fur_count 数组中的元素累加到 result 中
    }

    return result;
}
```
这段代码是一个函数的结束，返回result变量的值。

```
/*
 * Return a random number between a & b
 * Ref: https://stackoverflow.com/a/686376/1730895
 */
float randomAB(float a, float b)
{
    return ((b - a) * ((float)rand() / (float)RAND_MAX)) + a;
}
```
这段代码是一个函数，用于返回a和b之间的随机数。它使用了rand()函数生成一个0到RAND_MAX之间的随机整数，然后将其转换为0到1之间的浮点数，最后将其映射到a和b之间的范围内。

```
/* Random floating point number between 0 and 1 */
float randFloat()
{
    return randomAB( 0, 1 );
}
```
这段代码是一个函数，用于返回0到1之间的随机浮点数。它调用了前面定义的randomAB函数，将a和b分别设置为0和1。
/* 定义游戏状态的常量，用于在主游戏循环中进行切换 */
#define STATE_STARTING      1
#define STATE_CHOOSING_FORT 2
#define STATE_TRAVELLING    3
#define STATE_TRADING       4

int main( void )
{
    /* 用于存储玩家状态的变量 */
    float player_funds = 0;                              /* 玩家没有钱 */
    int   player_furs[FUR_TYPE_COUNT]  = { 0, 0, 0, 0 }; /* 玩家没有毛皮 */

    /* 玩家输入的变量 */
    char  yes_or_no;
    int   event_picker;
    int   which_fort;

    /* 游戏进行的阶段 */
    int   game_state = STATE_STARTING;
}
    /* 商品价格 */
    float mink_price   = -1;  // 貂皮价格
    float beaver_price = -1;  // 海狸皮价格
    float ermine_price = -1;  // 白鼬皮价格
    float fox_price    = -1;  // 狐狸皮价格（有时会取得“最后”价格，可能是一个 bug）

    float mink_value;  // 貂皮价值
    float beaver_value;  // 海狸皮价值
    float ermine_value;  // 白鼬皮价值
    float fox_value;  // 狐狸皮价值（用于计算销售结果）

    srand( time( NULL ) );  // 种子随机数生成器

    printAtColumn( 31, "FUR TRADER" );  // 在指定列打印标题
    printAtColumn( 15, "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY" );  // 在指定列打印信息
    printAtColumn( 15, "(Ported to ANSI-C Oct 2012 krt@krt.com.au)" );  // 在指定列打印信息
    print( "\n\n\n" );  // 打印换行
    # Loop forever until the player asks to quit
    while True:
        if game_state == STATE_STARTING:
            showIntroduction()

            player_funds = 600  # Initial player start money
            zeroInventory(player_furs)  # Player fur inventory

            print("DO YOU WISH TO TRADE FURS?")
            yes_or_no = getYesOrNo()
            if yes_or_no == 'N':
                exit(0)  # STOP
            game_state = STATE_TRADING

        elif game_state == STATE_TRADING:
            print("")
            # 打印玩家的储蓄金额
            printf( "YOU HAVE $ %1.2f IN SAVINGS\n", player_funds );
            # 打印玩家拥有的毛皮数量
            printf( "AND %d FURS TO BEGIN THE EXPEDITION\n", MAX_FURS );
            # 获取玩家购买的毛皮数量
            getFursPurchase( player_furs );

            # 如果玩家拥有的毛皮数量超过最大允许数量
            if ( sumInventory( player_furs ) > MAX_FURS )
            {
                # 打印错误信息
                print( "" );
                print( "YOU MAY NOT HAVE THAT MANY FURS." );
                print( "DO NOT TRY TO CHEAT.  I CAN ADD." );
                print( "YOU MUST START AGAIN." );
                print( "" );
                # 将游戏状态设置为重新开始
                game_state = STATE_STARTING;   /* T/N: Wow, harsh. */
            }
            # 如果玩家拥有的毛皮数量没有超过最大允许数量
            else
            {
                # 将游戏状态设置为选择要前往的堡垒
                game_state = STATE_CHOOSING_FORT;
            }
        }

        # 如果游戏状态为选择要前往的堡垒
        else if ( game_state == STATE_CHOOSING_FORT )
        {
            # 获取玩家选择的要交易的堡垒
            which_fort = getFortChoice();
            # 显示所选堡垒的评论
            showFortComment( which_fort );
            # 打印询问玩家是否想在另一个堡垒进行交易
            print( "DO YOU WANT TO TRADE AT ANOTHER FORT?" );
            # 获取玩家的是或否选择
            yes_or_no = getYesOrNo();
            # 如果玩家选择否，则将游戏状态设置为STATE_TRAVELLING
            if ( yes_or_no == 'N' )
                game_state = STATE_TRAVELLING;
        }

        else if ( game_state == STATE_TRAVELLING )
        {
            # 打印空行
            print( "" );
            # 如果选择的堡垒是FORT_MONTREAL
            if ( which_fort == FORT_MONTREAL )
            {
                # 计算不同物品的价格
                mink_price   = ( ( 0.2 * randFloat() + 0.70 ) * 100 + 0.5 ) / 100;
                ermine_price = ( ( 0.2 * randFloat() + 0.65 ) * 100 + 0.5 ) / 100;
                beaver_price = ( ( 0.2 * randFloat() + 0.75 ) * 100 + 0.5 ) / 100;
                fox_price    = ( ( 0.2 * randFloat() + 0.80 ) * 100 + 0.5 ) / 100;

                # 打印在FORT HOCHELAGA的供应品的价格
                print( "SUPPLIES AT FORT HOCHELAGA COST $150.00." );
                print( "YOUR TRAVEL EXPENSES TO HOCHELAGA WERE $10.00." );
                # 减去旅行费用
                player_funds -= 160;
            }

            else if ( which_fort == FORT_QUEBEC )
            {
                # 计算不同皮草的价格
                mink_price   = ( ( 0.30 * randFloat() + 0.85 ) * 100 + 0.5 ) / 100;
                ermine_price = ( ( 0.15 * randFloat() + 0.80 ) * 100 + 0.5 ) / 100;
                beaver_price = ( ( 0.20 * randFloat() + 0.90 ) * 100 + 0.5 ) / 100;
                fox_price    = ( ( 0.25 * randFloat() + 1.10 ) * 100 + 0.5 ) / 100;
                event_picker = ( 10 * randFloat() ) + 1;

                if ( event_picker <= 2 )
                {
                    # 输出提示信息
                    print( "YOUR BEAVER WERE TOO HEAVY TO CARRY ACROSS" );
                    print( "THE PORTAGE.  YOU HAD TO LEAVE THE PELTS, BUT FOUND" );
                    print( "THEM STOLEN WHEN YOU RETURNED." );
                    # 将玩家的海狸皮数量设为0
                    player_furs[ FUR_BEAVER ] = 0;
                }
                else if ( event_picker <= 6 )
                {
                    print( "YOU ARRIVED SAFELY AT FORT STADACONA." );  # 如果事件选择器小于等于3，打印“你安全抵达圣达科纳堡。”
                }
                else if ( event_picker <= 8 )
                {
                    print( "YOUR CANOE UPSET IN THE LACHINE RAPIDS.  YOU" );  # 如果事件选择器小于等于8，打印“你的独木舟在拉钦急流中翻了。你失去了所有的毛皮。”
                    print( "LOST ALL YOUR FURS." );
                    zeroInventory( player_furs );  # 调用zeroInventory函数，传入player_furs参数
                }
                else if ( event_picker <= 10 )
                {
                    print( "YOUR FOX PELTS WERE NOT CURED PROPERLY." );  # 如果事件选择器小于等于10，打印“你的狐狸皮没有被正确处理。没有人会买它们。”
                    print( "NO ONE WILL BUY THEM." );
                    player_furs[ FUR_FOX ] = 0;  # 将player_furs中的FUR_FOX索引位置的值设为0
                }
                else
                {
                    printf( "Internal Error #3, Out-of-bounds event_picker %d\n", event_picker );  # 如果事件选择器超出范围，打印错误信息
                    exit( 1 );  /* you have a bug */  # 退出程序并打印错误信息
                }
                print( "" );  # 打印空行
                print( "SUPPLIES AT FORT STADACONA COST $125.00." );  # 打印指定字符串
                print( "YOUR TRAVEL EXPENSES TO STADACONA WERE $15.00." );  # 打印指定字符串
                player_funds -= 140;  # 玩家资金减去140
            }

            else if ( which_fort == FORT_NEWYORK )  # 如果which_fort等于FORT_NEWYORK
            {
                mink_price   = ( ( 0.15 * randFloat() + 1.05 ) * 100 + 0.5 ) / 100;  # 计算mink_price的值
                ermine_price = ( ( 0.15 * randFloat() + 0.95 ) * 100 + 0.5 ) / 100;  # 计算ermine_price的值
                beaver_price = ( ( 0.25 * randFloat() + 1.00 ) * 100 + 0.5 ) / 100;  # 计算beaver_price的值
                if ( fox_price < 0 )  # 如果fox_price小于0
                {
                    /* Original Bug?  There is no Fox price generated for New York,
                       it will use any previous "D1" price.
                       So if there was no previous value, make one up */
                    fox_price = ( ( 0.25 * randFloat() + 1.05 ) * 100 + 0.5 ) / 100; /* not in orginal code */  # 计算fox_price的值
                }
                event_picker = ( 10 * randFloat() ) + 1;  # 计算event_picker的值
                if ( event_picker <= 2 ):  # 如果事件选择器的值小于等于2
                {
                    print( "YOU WERE ATTACKED BY A PARTY OF IROQUOIS." );  # 打印“你被伊罗quois部队攻击了。”
                    print( "ALL PEOPLE IN YOUR TRADING GROUP WERE" );  # 打印“你交易组中的所有人都被”
                    print( "KILLED.  THIS ENDS THE GAME." );  # 打印“杀死了。游戏结束。”
                    exit( 0 );  # 退出程序
                }
                else if ( event_picker <= 6 ):  # 否则如果事件选择器的值小于等于6
                {
                    print( "YOU WERE LUCKY.  YOU ARRIVED SAFELY" );  # 打印“你很幸运。你安全到达了”
                    print( "AT FORT NEW YORK." );  # 打印“在纽约新堡。”
                }
                else if ( event_picker <= 8 ):  # 否则如果事件选择器的值小于等于8
                {
                    print( "YOU NARROWLY ESCAPED AN IROQUOIS RAIDING PARTY." );  # 打印“你勉强逃脱了伊罗quois的袭击。”
                    print( "HOWEVER, YOU HAD TO LEAVE ALL YOUR FURS BEHIND." );  # 打印“然而，你不得不把所有的毛皮都留下。”
                    zeroInventory( player_furs );  # 调用zeroInventory函数，传入player_furs参数
                }
                else if ( event_picker <= 10 ):  # 否则如果事件选择器的值小于等于10
                {
                    # 将水貂价格除以2
                    mink_price /= 2;
                    # 将狐狸价格除以2
                    fox_price  /= 2;
                    # 打印旅行中水貂和海狸受损的消息
                    print( "YOUR MINK AND BEAVER WERE DAMAGED ON YOUR TRIP." );
                    # 打印只能获得当前价格一半的消息
                    print( "YOU RECEIVE ONLY HALF THE CURRENT PRICE FOR THESE FURS." );
                }
                else
                {
                    # 打印内部错误消息
                    print( "Internal Error #4, Out-of-bounds event_picker %d\n" );
                    # 退出程序，因为出现了bug
                    exit( 1 );  /* you have a bug */
                }

                # 打印空行
                print( "" );
                # 打印纽约的供应品价格
                print( "SUPPLIES AT NEW YORK COST $85.00." );
                # 打印前往纽约的旅行费用
                print( "YOUR TRAVEL EXPENSES TO NEW YORK WERE $25.00." );
                # 玩家资金减去105
                player_funds -= 105;
            }

            else
            {
                printf( "Internal error #2, fort %d does not exist\n", which_fort );
                exit( 1 );  # you have a bug
            }

            # 计算销售额
            beaver_value = beaver_price * player_furs[ FUR_BEAVER ];
            fox_value    = fox_price    * player_furs[ FUR_FOX ];
            ermine_value = ermine_price * player_furs[ FUR_ERMINE ];
            mink_value   = mink_price   * player_furs[ FUR_MINK ];

            print( "" );
            printf( "YOUR BEAVER SOLD FOR $%6.2f\n", beaver_value );
            printf( "YOUR FOX SOLD FOR    $%6.2f\n", fox_value );
            printf( "YOUR ERMINE SOLD FOR $%6.2f\n", ermine_value );
            printf( "YOUR MINK SOLD FOR   $%6.2f\n", mink_value );

            player_funds += beaver_value + fox_value + ermine_value + mink_value;

            print( "" );
            printf( "YOU NOW HAVE $ %1.2f INCLUDING YOUR PREVIOUS SAVINGS\n", player_funds );
# 打印空行
print( "" );
# 打印询问是否想要在下一年交易毛皮
print( "DO YOU WANT TO TRADE FURS NEXT YEAR?" );
# 获取用户输入的是或否
yes_or_no = getYesOrNo();
# 如果用户输入的是 'N'，则退出程序
if ( yes_or_no == 'N' )
    exit( 0 );             /* STOP */
# 如果用户输入的是其他值，则将游戏状态设置为交易状态
else
    game_state = STATE_TRADING;

# 结束 if 语句块
}

# 结束 while 循环
}

# 返回 0，表示程序正常退出
return 0; /* exit OK */
```