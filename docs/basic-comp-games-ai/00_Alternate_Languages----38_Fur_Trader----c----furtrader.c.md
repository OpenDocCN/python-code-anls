# `basic-computer-games\00_Alternate_Languages\38_Fur_Trader\c\furtrader.c`

```py
/*
 * 将 furtrader.bas 移植到 ANSI C (C99) 语言，作者为 krt@krt.com.au
 *
 * 使用以下命令编译：
 *    gcc -g -Wall -Werror furtrader.c -o furtrader
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


/* 常量 */
#define FUR_TYPE_COUNT    4
#define FUR_MINK          0
#define FUR_BEAVER        1
#define FUR_ERMINE        2
#define FUR_FOX           3
#define MAX_FURS        190
const char *FUR_NAMES[FUR_TYPE_COUNT] = { "MINK", "BEAVER", "ERMINE", "FOX" };

#define FORT_TYPE_COUNT 3
#define FORT_MONTREAL   1
#define FORT_QUEBEC     2
#define FORT_NEWYORK    3
const char *FORT_NAMES[FORT_TYPE_COUNT] = { "HOCHELAGA (MONTREAL)", "STADACONA (QUEBEC)", "NEW YORK" };



/* 在指定列打印单词 */
void printAtColumn( int column, const char *words )
{
    int i;
    for ( i=0; i<column; i++ )
        printf( " " );
    printf( "%s\n", words );
}

/* 输出带有换行符的行 */
void print( const char *words )
{
    printf( "%s\n", words );
}

/* 显示玩家介绍信息 */
void showIntroduction()
{
    print( "YOU ARE THE LEADER OF A FRENCH FUR TRADING EXPEDITION IN " );
    print( "1776 LEAVING THE LAKE ONTARIO AREA TO SELL FURS AND GET" );
    print( "SUPPLIES FOR THE NEXT YEAR.  YOU HAVE A CHOICE OF THREE" );
    print( "FORTS AT WHICH YOU MAY TRADE.  THE COST OF SUPPLIES" );
    print( "AND THE AMOUNT YOU RECEIVE FOR YOUR FURS WILL DEPEND" );
    print( "ON THE FORT THAT YOU CHOOSE." );
    print( "" );
}


/*
 * 提示用户输入。
 * 当输入给定时，尝试将其转换为整数
 * 返回转换后的整数值，出错时返回 0
 */
int getNumericInput()
{
    int  result = -1;
    char buffer[64];   /* 用于存储用户输入的地方 */
    char *endstr;

    while ( result == -1 )
    {
        printf( ">> " );                                 /* 提示用户输入 */
        fgets( buffer, sizeof( buffer ), stdin );        /* 从控制台读取内容到缓冲区 */
        result = (int)strtol( buffer, &endstr, 10 );     /* 将缓冲区内容转换为整数，进行简单的错误检查 */

        if ( endstr == buffer )                          /* 检查字符串是否成功转换为整数 */
            result = -1;
    }

    return result;
}
/*
 * Prompt the user for YES/NO input.
 * When input is given, try to work out if it's YES, Yes, yes, Y, etc.
 * And convert to a single upper-case letter
 * Returns a character of 'Y' or 'N'.
 */
char getYesOrNo()
{
    char result = '!';  /* initialize the result with a non-valid value */
    char buffer[64];   /* somewhere to store user input */

    while ( !( result == 'Y' || result == 'N' ) )       /* While the answer was not Yes or No */
    {
        print( "ANSWER YES OR NO" );  /* prompt the user to answer Yes or No */
        printf( ">> " );  /* print the input prompt */

        fgets( buffer, sizeof( buffer ), stdin );            /* read from the console into the buffer */
        if ( buffer[0] == 'Y' || buffer[0] == 'y' )  /* check if the input is Y or y */
            result = 'Y';  /* set the result to 'Y' */
        else if ( buffer[0] == 'N' || buffer[0] == 'n' )  /* check if the input is N or n */
            result = 'N';  /* set the result to 'N' */
    }

    return result;  /* return the result */
}

/*
 * Show the player the choices of Fort, get their input, if the
 * input is a valid choice (1,2,3) return it, otherwise keep
 * prompting the user.
 */
int getFortChoice()
{
    int result = 0;  /* initialize the result with 0 */

    while ( result == 0 )  /* While the result is not a valid choice */
    {
        print( "" );  /* print an empty line */
        print( "YOU MAY TRADE YOUR FURS AT FORT 1, FORT 2," );  /* print the options for the player */
        print( "OR FORT 3.  FORT 1 IS FORT HOCHELAGA (MONTREAL)" );
        print( "AND IS UNDER THE PROTECTION OF THE FRENCH ARMY." );
        print( "FORT 2 IS FORT STADACONA (QUEBEC) AND IS UNDER THE" );
        print( "PROTECTION OF THE FRENCH ARMY.  HOWEVER, YOU MUST" );
        print( "MAKE A PORTAGE AND CROSS THE LACHINE RAPIDS." );
        print( "FORT 3 IS FORT NEW YORK AND IS UNDER DUTCH CONTROL." );
        print( "YOU MUST CROSS THROUGH IROQUOIS LAND." );
        print( "ANSWER 1, 2, OR 3." ;

        result = getNumericInput();   /* get input from the player */
    }

    return result;  /* return the result */
}

/*
 * Print the description for the fort
 */
void showFortComment( int which_fort )
{
    print( "" );  /* print an empty line */
    if ( which_fort == FORT_MONTREAL )  /* check if the fort is Montreal */
    {
        # 打印消息，提示选择了最容易的路线，但是离海港很远，毛皮的价值会很低，而且供应品的成本会比圣达科纳堡或纽约堡高
        print( "YOU HAVE CHOSEN THE EASIEST ROUTE.  HOWEVER, THE FORT" );
        print( "IS FAR FROM ANY SEAPORT.  THE VALUE" );
        print( "YOU RECEIVE FOR YOUR FURS WILL BE LOW AND THE COST" );
        print( "OF SUPPLIES HIGHER THAN AT FORTS STADACONA OR NEW YORK." );
    }
    else if ( which_fort == FORT_QUEBEC )
    {
        # 打印消息，提示选择了较难的路线，比与霍切拉加的路线更难，但比去纽约的路线更容易，毛皮的价值和供应品的成本都是平均水平
        print( "YOU HAVE CHOSEN A HARD ROUTE.  IT IS, IN COMPARSION," );
        print( "HARDER THAN THE ROUTE TO HOCHELAGA BUT EASIER THAN" );
        print( "THE ROUTE TO NEW YORK.  YOU WILL RECEIVE AN AVERAGE VALUE" );
        print( "FOR YOUR FURS AND THE COST OF YOUR SUPPLIES WILL BE AVERAGE." );
    }
    else if ( which_fort == FORT_NEWYORK )
    {
        # 打印消息，提示选择了最困难的路线，在纽约堡可以获得最高价值的毛皮，而且供应品的成本会比其他堡低
        print( "YOU HAVE CHOSEN THE MOST DIFFICULT ROUTE.  AT" );
        print( "FORT NEW YORK YOU WILL RECEIVE THE HIGHEST VALUE" );
        print( "FOR YOUR FURS.  THE COST OF YOUR SUPPLIES" );
        print( "WILL BE LOWER THAN AT ALL THE OTHER FORTS." );
    }
    else
    {
        # 打印错误消息，提示选择的堡不存在，并退出程序
        printf( "Internal error #1, fort %d does not exist\n", which_fort );
        exit( 1 );  /* you have a bug */
    }
    # 打印空行
    print( "" );
/*
 * Prompt the player for how many of each fur type they want.
 * Accept numeric inputs, re-prompting on incorrect input values
 */
void getFursPurchase( int *furs )
{
    int i;

    printf( "YOUR %d FURS ARE DISTRIBUTED AMONG THE FOLLOWING\n", FUR_TYPE_COUNT );
    print( "KINDS OF PELTS: MINK, BEAVER, ERMINE AND FOX." );
    print( "" );

    for ( i=0; i<FUR_TYPE_COUNT; i++ )
    {
        printf( "HOW MANY %s DO YOU HAVE\n", FUR_NAMES[i] );
        furs[i] = getNumericInput();
    }
}


/*
 * (Re)Set the player's inventory to zero
 */
void zeroInventory( int *player_fur_count )
{
    int i;
    for ( i=0; i<FUR_TYPE_COUNT; i++ )
    {
        player_fur_count[i] = 0;
    }
}


/*
 * Tally the player's inventory
 */
int sumInventory( int *player_fur_count )
{
    int result = 0;
    int i;
    for ( i=0; i<FUR_TYPE_COUNT; i++ )
    {
        result += player_fur_count[i];
    }

    return result;
}


/*
 * Return a random number between a & b
 * Ref: https://stackoverflow.com/a/686376/1730895
 */
float randomAB(float a, float b)
{
    return ((b - a) * ((float)rand() / (float)RAND_MAX)) + a;
}
/* Random floating point number between 0 and 1 */
float randFloat()
{
    return randomAB( 0, 1 );
}


/* States to allow switching in main game-loop */
#define STATE_STARTING      1
#define STATE_CHOOSING_FORT 2
#define STATE_TRAVELLING    3
#define STATE_TRADING       4

int main( void )
{
    /* variables for storing player's status */
    float player_funds = 0;                              /* no money */
    int   player_furs[FUR_TYPE_COUNT]  = { 0, 0, 0, 0 }; /* no furs */

    /* player input holders */
    char  yes_or_no;
    int   event_picker;
    int   which_fort;

    /* what part of the game is in play */
    int   game_state = STATE_STARTING;

    /* commodity prices */
    float mink_price   = -1;
    float beaver_price = -1;
    float ermine_price = -1;
    float fox_price    = -1;  /* 定义狐狸价格变量，初始化为-1，有时可能会取到“最后”价格（可能是一个bug） */

    float mink_value;  /* 定义貂皮价值变量 */
    float beaver_value;  /* 定义海狸皮价值变量 */
    float ermine_value;  /* 定义鼬皮价值变量 */
    float fox_value;      /* 定义狐狸皮价值变量，用于计算销售结果 */


    srand( time( NULL ) );  /* 用当前时间作为种子，初始化随机数生成器 */

    printAtColumn( 31, "FUR TRADER" );  /* 在第31列打印“FUR TRADER” */
    printAtColumn( 15, "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY" );  /* 在第15列打印“CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY” */
    printAtColumn( 15, "(Ported to ANSI-C Oct 2012 krt@krt.com.au)" );  /* 在第15列打印“(Ported to ANSI-C Oct 2012 krt@krt.com.au)” */
    print( "\n\n\n" );  /* 打印三个换行符 */


    /* 永远循环，直到玩家要求退出 */
    while ( 1 )
    }

    return 0; /* 退出，正常结束 */
# 闭合前面的函数定义
```