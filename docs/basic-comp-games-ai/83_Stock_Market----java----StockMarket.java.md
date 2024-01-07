# `basic-computer-games\83_Stock_Market\java\StockMarket.java`

```

// 导入所需的 Java 类
import java.util.ArrayList;
import java.util.InputMismatchException;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

/**
 * 股票市场模拟
 *
 * 一些原始程序变量的文档和它们在这个程序中的等价物：
 * A-MRKT TRND SLP;             marketTrendSlope
 * B5-BRKRGE FEE;               brokerageFee
 * C-TTL CSH ASSTS;             cashAssets
 * C5-TTL CSH ASSTS (TEMP);     tmpCashAssets
 * C(I)-CHNG IN STK VAL;        changeStockValue
 * D-TTL ASSTS;                 assets
 * E1,E2-LRG CHNG MISC;         largeChange1, largeChange2
 * I1,I2-STCKS W LRG CHNG;      randomStockIndex1, randomStockIndex2
 * N1,N2-LRG CHNG DAY CNTS;     largeChangeNumberDays1, largeChangeNumberDays2
 * P5-TTL DAYS PRCHSS;          totalDaysPurchases
 * P(I)-PRTFL CNTNTS;           portfolioContents
 * Q9-NEW CYCL?;                newCycle
 * S4-SGN OF A;                 slopeSign
 * S5-TTL DYS SLS;              totalDaysSales
 * S(I)-VALUE/SHR;              stockValue
 * T-TTL STCK ASSTS;            totalStockAssets
 * T5-TTL VAL OF TRNSCTNS;      totalValueOfTransactions
 * W3-LRG CHNG;                 bigChange
 * X1-SMLL CHNG(<$1);           smallChange
 * Z4,Z5,Z6-NYSE AVE.;          tmpNyseAverage, nyseAverage, nyseAverageChange
 * Z(I)-TRNSCT                  transactionQuantity
 *
 * 新价格 = 旧价格 + (趋势 x 旧价格) + (小的随机价格变动) + (可能的大价格变动)
 *
 * 由 Aldrin Misquitta (@aldrinm) 从 BASIC 转换为 Java
 */

```