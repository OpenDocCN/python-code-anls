# `basic-computer-games\83_Stock_Market\python\Stock_Market.py`

```

# 导入 random 模块和 typing 模块中的 Any, Dict, List 类型
import random
from typing import Any, Dict, List

# 打印游戏说明
def print_instruction() -> None:
    print(
        """
        此程序模拟股票市场。您将获得$10,000，并可以买入或卖出股票。股票价格将随机生成，因此此模型并不完全代表交易所的真实情况。将打印出可用股票、它们的价格以及您投资组合中的股票数量的表格。在此之后，将打印出每支股票的缩写和一个问号。在这里，您可以指示进行交易。要买入股票，请输入+NNN，要卖出股票，请输入-NNN，其中NNN是股票数量。所有交易将收取1%的佣金。请注意，如果一支股票的价值降至零，它可能会再次反弹至正值。您有$10,000用于投资。请对所有输入使用整数。（注意：要对市场有所了解，请至少运行10天）
          ------------祝您好运！------------
        """
    )

# 主函数
def main() -> None:
    print("\t\t      股票市场")
    help = input("\n是否需要说明书(YES或NO)? ")

    # 打印说明书
    if help.lower() == "yes":
        print_instruction()

    # 初始化游戏
    Game = Stock_Market()

    # 进行第一天交易
    Game.print_first_day()
    new_holdings = Game.take_inputs()
    Game.update_holdings(new_holdings)
    Game.update_cash_assets(new_holdings)
    print("\n------------交易日结束--------------\n")

    response = 1
    while response == 1:

        # 模拟一天
        Game.update_prices()
        Game.print_trading_day()
        Game.print_exchange_average()
        Game.update_stock_assets()
        Game.print_assets()

        response = int(input("\n是否继续交易 (是-输入1, 否-输入0)? "))
        if response == 0:
            break

        new_holdings = Game.take_inputs()
        Game.update_holdings(new_holdings)
        Game.update_cash_assets(new_holdings)
        print("\n------------交易日结束--------------\n")

    print("\n希望您玩得开心！！！！")
    input()

# 如果当前脚本为主程序，则执行主函数
if __name__ == "__main__":
    main()

```