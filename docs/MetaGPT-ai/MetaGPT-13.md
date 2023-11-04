# MetaGPT源码解析 13

# `tests/metagpt/actions/test_action.py`

这段代码是一个Python脚本，用于测试"test_action.py"文件夹下的脚本。该脚本使用了一个名为"Action"的类，该类来自"metagpt.actions"模块，它提供了对PRD（产品需求文档）的写入和测试操作。

具体来说，该脚本包含以下两个函数：

1. `test_action_repr()`函数：该函数的作用是打印出给定的动作列表（包含动作类和动作名称）的字符串表示。该函数使用了一个名为`assert`的语句，用于验证给定的动作列表是否与预期的列表匹配。在这个例子中，它验证了给定的动作列表是否包含名为"WriteTest"的动作，因此如果列表中不包含这个名字，该函数将会引发一个异常。
2. `test_action.py`函数：该函数的作用是定义了给定的动作列表。在该函数中，使用了`Action`类，并调用了其三个构造函数，分别创建了一个名为"WriteTest"的动作和一个名为"WritePRD"的动作，还调用了"WriteTest"的`add_metag波澜"`方法。这些方法都在定义中使用了`Action`类中的常用方法，用于将给定的元数据添加到PRD中。

通过分析这段代码，我们可以了解到该脚本测试了一个名为"test_action.py"的文件，该文件定义了一个名为"test_action"的函数，该函数实现了将给定的元数据添加到PRD中的操作。在该函数中，使用了来自"metagpt.actions"模块的"Action"类，以及定义了一系列方法，包括"write_product_requirement"和"write_test"。此外，还定义了一个名为"test_action_repr"的函数，用于打印给定的动作列表的字符串表示。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : test_action.py
"""
from metagpt.actions import Action, WritePRD, WriteTest


def test_action_repr():
    actions = [Action(), WriteTest(), WritePRD()]
    assert "WriteTest" in str(actions)

```

# `tests/metagpt/actions/test_action_output.py`

This appears to be a list of tasks or requirements for a web game. It is not clear from the information provided what the application does or what the tasks are for the game. It is possible that the tasks are listed for a game that is built using the Flask web framework and the Python programming language, and that the game is responsible for managing the game logic and user interface.



```py
#!/usr/bin/env python
# coding: utf-8
"""
@Time    : 2023/7/11 10:49
@Author  : chengmaoyu
@File    : test_action_output
"""
from typing import List, Tuple

from metagpt.actions import ActionOutput

t_dict = {"Required Python third-party packages": "\"\"\"\nflask==1.1.2\npygame==2.0.1\n\"\"\"\n",
          "Required Other language third-party packages": "\"\"\"\nNo third-party packages required for other languages.\n\"\"\"\n",
          "Full API spec": "\"\"\"\nopenapi: 3.0.0\ninfo:\n  title: Web Snake Game API\n  version: 1.0.0\npaths:\n  /game:\n    get:\n      summary: Get the current game state\n      responses:\n        '200':\n          description: A JSON object of the game state\n    post:\n      summary: Send a command to the game\n      requestBody:\n        required: true\n        content:\n          application/json:\n            schema:\n              type: object\n              properties:\n                command:\n                  type: string\n      responses:\n        '200':\n          description: A JSON object of the updated game state\n\"\"\"\n",
          "Logic Analysis": [
              ["app.py", "Main entry point for the Flask application. Handles HTTP requests and responses."],
              ["game.py", "Contains the Game and Snake classes. Handles the game logic."],
              ["static/js/script.js", "Handles user interactions and updates the game UI."],
              ["static/css/styles.css", "Defines the styles for the game UI."],
              ["templates/index.html", "The main page of the web application. Displays the game UI."]],
          "Task list": ["game.py", "app.py", "static/css/styles.css", "static/js/script.js", "templates/index.html"],
          "Shared Knowledge": "\"\"\"\n'game.py' contains the Game and Snake classes which are responsible for the game logic. The Game class uses an instance of the Snake class.\n\n'app.py' is the main entry point for the Flask application. It creates an instance of the Game class and handles HTTP requests and responses.\n\n'static/js/script.js' is responsible for handling user interactions and updating the game UI based on the game state returned by 'app.py'.\n\n'static/css/styles.css' defines the styles for the game UI.\n\n'templates/index.html' is the main page of the web application. It displays the game UI and loads 'static/js/script.js' and 'static/css/styles.css'.\n\"\"\"\n",
          "Anything UNCLEAR": "We need clarification on how the high score should be stored. Should it persist across sessions (stored in a database or a file) or should it reset every time the game is restarted? Also, should the game speed increase as the snake grows, or should it remain constant throughout the game?"}

```

这段代码定义了一个名为 "WRITE\_TASKS\_OUTPUT\_MAPPING" 的字典，它包含了多个键值对，每个键都是一个字符串，然后是一个或多个需要安装的 Python 或其他编程语言的第三方包。这些键值对用于将依赖项映射到代码中需要使用的库。

接着定义了一个函数 "test\_create\_model\_class"，该函数接收一个字符串参数 "test\_class"，然后使用 "WRITE\_TASKS\_OUTPUT\_MAPPING" 字典中定义的键值对，创建一个名为 "test\_class" 的类实例。

最后在函数内部对实例进行了一些属性检查，如实例的名称是否为 "test\_class"，以及检查 "test\_class" 是否实现了 "ActionOutput.create\_model\_class" 接口。


```py
WRITE_TASKS_OUTPUT_MAPPING = {
    "Required Python third-party packages": (str, ...),
    "Required Other language third-party packages": (str, ...),
    "Full API spec": (str, ...),
    "Logic Analysis": (List[Tuple[str, str]], ...),
    "Task list": (List[str], ...),
    "Shared Knowledge": (str, ...),
    "Anything UNCLEAR": (str, ...),
}


def test_create_model_class():
    test_class = ActionOutput.create_model_class("test_class", WRITE_TASKS_OUTPUT_MAPPING)
    assert test_class.__name__ == "test_class"


```

这段代码定义了一个名为 `test_create_model_class_with_mapping` 的函数，它的作用是测试创建一个名为 "test_class_1" 的模型类，并将 `WRITE_TASKS_OUTPUT_MAPPING` 参数映射到模型的类中。

具体来说，这个函数首先使用 `ActionOutput.create_model_class` 方法创建了一个名为 "test_class_1" 的模型类，并将 `WRITE_TASKS_OUTPUT_MAPPING` 参数映射到该模型的类中。接着，使用函数内部的 `**` 运算符，将模型的类参数 `**t_dict` 传递给模型类创建函数，从而创建出第二个实例。

最后，函数内部获取第二个实例的 `dict()` 方法返回的键为 "Task list" 的值，并使用 `assert` 语句进行断言，如果返回的值与预期的 "game.py"、"app.py"、"static/css/styles.css"、"static/js/script.js"、"templates/index.html" 相等，则说明函数的第二次调用成功。


```py
def test_create_model_class_with_mapping():
    t = ActionOutput.create_model_class("test_class_1", WRITE_TASKS_OUTPUT_MAPPING)
    t1 = t(**t_dict)
    value = t1.dict()["Task list"]
    assert value == ["game.py", "app.py", "static/css/styles.css", "static/js/script.js", "templates/index.html"]


if __name__ == '__main__':
    test_create_model_class()
    test_create_model_class_with_mapping()

```

# `tests/metagpt/actions/test_azure_tts.py`

这段代码是一个Python脚本，使用了Python的`/usr/bin/env python`环境来执行。脚本主要实现了测试Azure TTS（文本转语音）功能。具体地，执行以下操作：

1. 导入Azure TTS库
2. 创建一个名为`AzureTTS`的类，该类实现了`metagpt.actions.azure_tts.AzureTTS`接口。
3. 在`test_azure_tts()`函数中，使用`AzureTTS`类创建一个`AzureTTS`实例。
4. 使用`synthesize_speech()`方法将文本"你好，我是卡卡"转换成语音并输出.wav文件。

脚本主要的作用是测试`AzureTTS`类是否正确实现了`metagpt.actions.azure_tts.AzureTTS`接口，测试其将文本转换为语音的功能。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/7/1 22:50
@Author  : alexanderwu
@File    : test_azure_tts.py
"""
from metagpt.actions.azure_tts import AzureTTS


def test_azure_tts():
    azure_tts = AzureTTS("azure_tts")
    azure_tts.synthesize_speech(
        "zh-CN",
        "zh-CN-YunxiNeural",
        "Boy",
        "你好，我是卡卡",
        "output.wav")

    # 运行需要先配置 SUBSCRIPTION_KEY
    # TODO: 这里如果要检验，还要额外加上对应的asr，才能确保前后生成是接近一致的，但现在还没有

```

# `tests/metagpt/actions/test_clone_function.py`

这段代码的作用是测试一个名为“user_indicator”的函数，该函数使用简单移动平均线（SMA）和布林带（Bollinger Bands）来分析股票数据。

具体来说，这段代码：

1. 导入pytest和metagpt.actions.clone_function模块；
2. 从metagpt.actions.clone_function模块中导入名为CloneFunction的类，该类包含运行函数代码的功能；
3. 从metagpt.actions.clone_function模块中导入名为run_function_code的函数；
4. 定义一个名为“user_indicator”的函数，该函数读取股票数据并计算简单移动平均线、布林带等指标；
5. 在函数内部，使用pandas的read\_csv函数读取股票数据，使用ta库的trend函数计算简单移动平均线，使用pandas的head函数输出数据的前几行；
6. 使用ta库的sma函数计算布林带的上、中、下指标，使用ta库的window参数设置指标的时间窗口；
7. 使用ta库的bollinger\_hband\_indicator函数计算布林带指标，使用ta库的window参数设置指标的时间窗口；
8. 使用ta库的bollinger\_mavg函数计算布林带指标，使用ta库的window参数设置指标的时间窗口；
9. 使用ta库的bollinger\_lband\_indicator函数计算布林带指标，使用ta库的window参数设置指标的时间窗口；
10. 使用pandas的head函数输出指标数据；
11. 使用函数CloneFunction.run\_function\_code()方法运行函数代码，并返回指标数据。


```py
import pytest

from metagpt.actions.clone_function import CloneFunction, run_function_code


source_code = """
import pandas as pd
import ta

def user_indicator():
    # 读取股票数据
    stock_data = pd.read_csv('./tests/data/baba_stock.csv')
    stock_data.head()
    # 计算简单移动平均线
    stock_data['SMA'] = ta.trend.sma_indicator(stock_data['Close'], window=6)
    stock_data[['Date', 'Close', 'SMA']].head()
    # 计算布林带
    stock_data['bb_upper'], stock_data['bb_middle'], stock_data['bb_lower'] = ta.volatility.bollinger_hband_indicator(stock_data['Close'], window=20), ta.volatility.bollinger_mavg(stock_data['Close'], window=20), ta.volatility.bollinger_lband_indicator(stock_data['Close'], window=20)
    stock_data[['Date', 'Close', 'bb_upper', 'bb_middle', 'bb_lower']].head()
```

这段代码定义了一个名为 `stock_indicator` 的函数，它接收一个 `stock_path` 参数，并返回一个 pandas DataFrame。函数内部使用了许多参数 `indicators`，这些参数是过滤和计算技术指标所需的主要函数。

首先，函数导入了 pandas 和 TA (技术分析) 库。然后，函数内部读取股票数据并将其存储在变量 `stock_data` 中。接下来，函数内部使用 TA 库中的 `sma_indicator` 函数计算股票的简单移动平均线，并将其存储在变量 `stock_data` 中。然后，函数使用 pandas 的 `head` 函数来查看数据的前几行。

接下来，函数内部使用 TA 库中的 `bollinger_hband_indicator`、`bollinger_mavg` 和 `bollinger_lband_indicator` 函数计算布林带指标。这些函数使用历史数据计算移动平均线，并返回一个 DataFrame 包含计算得到的指标。最后，函数使用 pandas 的 `head` 函数来查看数据的前几行，并返回经过处理的 DataFrame。

由于 `get_expected_res` 函数没有参数和返回值，因此它不会对程序产生实际的输出。


```py
"""

template_code = """
def stock_indicator(stock_path: str, indicators=['Simple Moving Average', 'BollingerBands', 'MACD]) -> pd.DataFrame:
    import pandas as pd
    # here is your code.
"""


def get_expected_res():
    import pandas as pd
    import ta

    # 读取股票数据
    stock_data = pd.read_csv('./tests/data/baba_stock.csv')
    stock_data.head()
    # 计算简单移动平均线
    stock_data['SMA'] = ta.trend.sma_indicator(stock_data['Close'], window=6)
    stock_data[['Date', 'Close', 'SMA']].head()
    # 计算布林带
    stock_data['bb_upper'], stock_data['bb_middle'], stock_data['bb_lower'] = ta.volatility.bollinger_hband_indicator(stock_data['Close'], window=20), ta.volatility.bollinger_mavg(stock_data['Close'], window=20), ta.volatility.bollinger_lband_indicator(stock_data['Close'], window=20)
    stock_data[['Date', 'Close', 'bb_upper', 'bb_middle', 'bb_lower']].head()
    return stock_data


```

这段代码使用 Python 的 `@pytest.mark.asyncio` 标示，定义了一个名为 `test_clone_function` 的测试函数。函数内部使用 `asyncio` 作为關鍵字，说明该函数是一个异步函数。

函数的主要部分是：

1. 定义了一个名为 `CloneFunction` 的类，该类实现了 `clone` 方法，但并没有定义任何方法体。
2. 使用 `run` 方法（使用 `asyncio` 的 `run` 方法可以异步执行代码块）运行 `CloneFunction` 类的 `run` 方法，并接收两个参数：`template_code` 和 `source_code`。
3. 使用 `assert` 语句检查代码是否符合预期。具体来说：

a. 检查 `'def'` 是否出现在代码中，这是 `CloneFunction` 类中 `__init__` 方法的参数，如果没有该参数，测试会失败。
b. 检查 `stock_path` 是否为 `./tests/data/baba_stock.csv`，如果不是，测试会失败。
c. 运行一个名为 `run_function_code` 的函数，接收两个参数：`code` 和 `stock_path`。这个函数的作用未知，但是和 `test_clone_function` 没有直接关系。
d. 检查 `msg` 是否为空字符串，如果是，说明运行时没有错误。

这里 `test_clone_function` 作为测试函数，主要是为了验证 `CloneFunction` 类是否实现了异步的 `clone` 方法。


```py
@pytest.mark.asyncio
async def test_clone_function():
    clone = CloneFunction()
    code = await clone.run(template_code, source_code)
    assert 'def ' in code
    stock_path = './tests/data/baba_stock.csv'
    df, msg = run_function_code(code, 'stock_indicator', stock_path)
    assert not msg
    expected_df = get_expected_res()
    assert df.equals(expected_df)

```

# `tests/metagpt/actions/test_debug_error.py`

这段代码是一个Python脚本，用于测试metagpt中DebugError动作的错误处理。具体来说，这段代码的作用是导入metagpt中的DebugError类，以便在测试中使用。

具体来说，DebugError类是一个自定义的类，用于处理在调试时产生的错误。在这段代码中，我们通过导入这个类来测试/调试由metagpt产生的错误。

在脚本的正文中，我们使用pytest库来发起测试。pytest是一个流行的Python测试框架，可以轻松地编写和运行测试。

在测试部分，我们定义了一个example_msg_content变量，该变量是一个示例元数据字符串。这个字符串定义了一个错误消息，用于在调试时显示错误信息。

最后，我们在脚本顶部添加了一个说明，用于描述这段代码的作用。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:46
@Author  : alexanderwu
@File    : test_debug_error.py
"""
import pytest

from metagpt.actions.debug_error import DebugError

EXAMPLE_MSG_CONTENT = '''
---
## Development Code File Name
player.py
```

这段代码定义了一个名为Player的类，该类表示Black Jack游戏中的玩家。Player类包含一个 name 属性，用于存储玩家的名字，以及一个 hand 属性，用于存储玩家当前手中的牌。此外，Player类还有一个 calculate\_score() 方法，用于计算玩家当前手中的牌的得分，以及一个 draw() 方法，用于从游戏中的卡堆中抽出一张牌并将其添加到玩家手中。

Player类的构造函数用于初始化Player对象，并传入玩家的名字作为参数。在构造函数中，还创建了一个 hand 属性，用于记录玩家当前手中的牌。在 calculate\_score() 方法中，使用内置函数 deck.draw\_card() 从游戏中的卡堆中抽出一张牌并将其添加到玩家手中。最后，如果玩家的得分超过21，并且任何一张A 牌被认为等于11，那么得分将减少10分。


```py
## Development Code
```python
from typing import List
from deck import Deck
from card import Card

class Player:
    """
    A class representing a player in the Black Jack game.
    """

    def __init__(self, name: str):
        """
        Initialize a Player object.
        
        Args:
            name (str): The name of the player.
        """
        self.name = name
        self.hand: List[Card] = []
        self.score = 0

    def draw(self, deck: Deck):
        """
        Draw a card from the deck and add it to the player's hand.
        
        Args:
            deck (Deck): The deck of cards.
        """
        card = deck.draw_card()
        self.hand.append(card)
        self.calculate_score()

    def calculate_score(self) -> int:
        """
        Calculate the score of the player's hand.
        
        Returns:
            int: The score of the player's hand.
        """
        self.score = sum(card.value for card in self.hand)
        # Handle the case where Ace is counted as 11 and causes the score to exceed 21
        if self.score > 21 and any(card.rank == 'A' for card in self.hand):
            self.score -= 10
        return self.score

```py

The test suite for the `Player` class included 5 test methods:

1. `test_player_initialization`: checks that the initial draw of the player is correct.
2. `test_player_draw`: checks that the player's score is updated correctly after drawing cards.
3. `test_player_calculate_score`: checks that the player's score is calculated correctly based on the value of the cards in the player's hand.
4. `test_player_calculate_score_with_ace`: checks that the player's score is correctly calculated when an Ace card is added to the player's hand.
5. `test_player_calculate_score_with_multiple_aces`: checks that the player's score is correctly calculated when multiple Aces cards are added to the player's hand.


```
```py
## Test File Name
test_player.py
## Test Code
```python
import unittest
from blackjack_game.player import Player
from blackjack_game.deck import Deck
from blackjack_game.card import Card

class TestPlayer(unittest.TestCase):
    ## Test the Player's initialization
    def test_player_initialization(self):
        player = Player("Test Player")
        self.assertEqual(player.name, "Test Player")
        self.assertEqual(player.hand, [])
        self.assertEqual(player.score, 0)

    ## Test the Player's draw method
    def test_player_draw(self):
        deck = Deck()
        player = Player("Test Player")
        player.draw(deck)
        self.assertEqual(len(player.hand), 1)
        self.assertEqual(player.score, player.hand[0].value)

    ## Test the Player's calculate_score method
    def test_player_calculate_score(self):
        deck = Deck()
        player = Player("Test Player")
        player.draw(deck)
        player.draw(deck)
        self.assertEqual(player.score, sum(card.value for card in player.hand))

    ## Test the Player's calculate_score method with Ace card
    def test_player_calculate_score_with_ace(self):
        deck = Deck()
        player = Player("Test Player")
        player.hand.append(Card('A', 'Hearts', 11))
        player.hand.append(Card('K', 'Hearts', 10))
        player.calculate_score()
        self.assertEqual(player.score, 21)

    ## Test the Player's calculate_score method with multiple Aces
    def test_player_calculate_score_with_multiple_aces(self):
        deck = Deck()
        player = Player("Test Player")
        player.hand.append(Card('A', 'Hearts', 11))
        player.hand.append(Card('A', 'Diamonds', 11))
        player.calculate_score()
        self.assertEqual(player.score, 12)

```py

这段代码是一个if语句，它的作用是：如果当前脚本被称为'__main__'，那么执行if语句块内的内容。

在这个if语句块内，使用了一个unittest.main()函数作为子程序入口，这是因为，当脚本被运行时，unittest.main()函数会创建一个新的unittest测试套件并运行测试套件内的所有测试函数。

所以，这段代码的作用是：如果当前脚本被称为'__main__'，那么执行unittest.main()函数，并创建一个新的unittest测试套件并运行其中的测试函数。


```
if __name__ == '__main__':
    unittest.main()

```py
## Running Command
python tests/test_player.py
## Running Output
standard output: ;
standard errors: ..F..
======================================================================
FAIL: test_player_calculate_score_with_multiple_aces (__main__.TestPlayer)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "tests/test_player.py", line 46, in test_player_calculate_score_with_multiple_aces
    self.assertEqual(player.score, 12)
```

这段代码是一个 Python 程序，它旨在验证一个名为 `Player` 的类的一个方法 `calculate_score` 是否正确。这个测试程序遇到了一个 `AssertionError`，它指出方法在传递一个不等于 12 的值时，没有给出预期的结果。

从代码中可以看出， `calculate_score` 方法对于多张 ACE 的牌情况下的处理是不正确的。具体来说，当得分超过 21 的时候，它只会在牌的分数上减少了 10 分，然后将 ACE 的分数减少了 10 分。但是，它没有正确地处理牌的分数低于 21 的情况。

为了输出这个错误，代码会捕获 `AssertionError`，并输出一条相关的错误消息。此外，还会输出测试的名称、代码行号和失败的结果。


```py
AssertionError: 22 != 12

----------------------------------------------------------------------
Ran 5 tests in 0.007s

FAILED (failures=1)
;
## instruction:
The error is in the development code, specifically in the calculate_score method of the Player class. The method is not correctly handling the case where there are multiple Aces in the player's hand. The current implementation only subtracts 10 from the score once if the score is over 21 and there's an Ace in the hand. However, in the case of multiple Aces, it should subtract 10 for each Ace until the score is 21 or less.
## File To Rewrite:
player.py
## Status:
FAIL
## Send To:
Engineer
```

这段代码是一个用于测试 Python 中 `asyncio` 模块的 `asyncio` 标記的函数，具体解释如下：

1. `@pytest.mark.asyncio` 是该函数的声明，表示该函数使用 `asyncio` 语法进行编写。
2. `async def test_debug_error()` 是该函数的主函数，其中 `async` 关键字表示该函数是一个异步函数。
3. `await debug_error.run(context=EXAMPLE_MSG_CONTENT)` 是该函数的实际体，表示异步运行 `debug_error` 函数，并将 `EXAMPLE_MSG_CONTENT` 作为参数传递给 `run` 方法。
4. `file_name, rewritten_code = await debug_error.run(context=EXAMPLE_MSG_CONTENT)` 将异步运行的结果保存到 `file_name` 和 `rewritten_code` 两个变量中。
5. `assert "class Player" in rewritten_code` 是一个断言，用于检查 `rewritten_code` 是否正确。该断言使用了 `assert` 关键字，表示如果 `rewritten_code` 的内容与预期不符，则会引发断言。此处的内容是一个比较，如果 `rewritten_code` 中包含了 `class Player`，则说明函数运行成功，否则断言将会引发。
6. `assert "while self.score > 21" in rewritten_code` 是另一个断言，用于检查 `rewritten_code` 是否正确。该断言使用了 `assert` 关键字，表示如果 `rewritten_code` 中包含了 `while self.score > 21` 这样的内容，则说明函数运行成功，否则断言将会引发。


```py
---
'''

@pytest.mark.asyncio
async def test_debug_error():

    debug_error = DebugError("debug_error")

    file_name, rewritten_code = await debug_error.run(context=EXAMPLE_MSG_CONTENT)

    assert "class Player" in rewritten_code # rewrite the same class
    assert "while self.score > 21" in rewritten_code # a key logic to rewrite to (original one is "if self.score > 12")

```

# `tests/metagpt/actions/test_design_api.py`

这段代码是一个Python脚本，使用了`#!/usr/bin/env python`作为脚本路径的指令，表示该脚本使用Python 3作为运行时环境。

该脚本的主要作用是测试一个名为`test_design_api.py`的文件，该文件中定义了一个名为`WriteDesign`的类，该类可以写入一个设计原型（Design原型）到指定的API。

具体来说，该脚本使用`pytest`库来方便地进行测试，使用`metagpt.actions.design_api`类将设计原型写入API，并使用`metagpt.logs.logger`类记录API的日志信息，使用`metagpt.schema.Message`类定义设计原型中的消息类型。

在该脚本中，还定义了一个名为`PRD_SAMPLE`的常量，表示从产品待测区中抽取的一个样本ID。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 19:26
@Author  : alexanderwu
@File    : test_design_api.py
"""
import pytest

from metagpt.actions.design_api import WriteDesign
from metagpt.logs import logger
from metagpt.schema import Message
from tests.metagpt.actions.mock import PRD_SAMPLE


```

这两行代码都是使用 Python 的 pytest 测试框架，用于编写一个名为 "test_design_api.py" 的测试文件。它们都是使用 asyncio 关键字作为函数指针，说明这两个函数都是使用异步 I/O 进行操作。

具体来说，第一行代码定义了一个名为 "test_design_api" 的函数，使用了 WriteDesign 类从设计中创建一个名为 "design_api" 的会话，并调用它的 run 方法来发送一个内容为 "我们需要一个音乐播放器，它应该有播放、暂停、上一曲、下一曲等功能。" 的消息，并将结果保存在变量 "result"，最后使用 assert 断言这个结果是否为 True。

第二行代码定义了一个名为 "test_design_api_calculator" 的函数，使用了同样 WriteDesign 类从设计中创建一个名为 "design_api" 的会话，并调用它的 run 方法来发送一个内容为 PRD_SAMPLE 的消息，并将结果保存在变量 "result"，最后使用 assert 断言这个结果是否为 True。

这两行代码的作用是测试一个名为 "design_api" 的设计类，它具有发送消息和接收结果的功能。在这些测试中，我们将使用异步 I/O 发送一个消息给设计类，并使用断言来验证设计类的行为是否符合预期。


```py
@pytest.mark.asyncio
async def test_design_api():
    prd = "我们需要一个音乐播放器，它应该有播放、暂停、上一曲、下一曲等功能。"

    design_api = WriteDesign("design_api")

    result = await design_api.run([Message(content=prd, instruct_content=None)])
    logger.info(result)

    assert result


@pytest.mark.asyncio
async def test_design_api_calculator():
    prd = PRD_SAMPLE

    design_api = WriteDesign("design_api")
    result = await design_api.run([Message(content=prd, instruct_content=None)])
    logger.info(result)

    assert result

```

# `tests/metagpt/actions/test_design_api_review.py`

```py
async def test_design_api_review():
   prd = "我们需要一个音乐播放器，它应该有播放、暂停、上一曲、下一曲等功能。"
   api_design = DesignReview(prd)
   
   # 创建一个测试用例，当播放量为"null"时，验证API设计是否正确
   assert api_design.playback_level == DesignReview.ACTION_PLAYBACK_NULL
   
   # 创建一个测试用例，当播放量为"正在播放"时，验证API设计是否正确
   assert api_design.playback_level == DesignReview.ACTION_PLAYBACK_ACTIVE
   
   # 创建一个测试用例，当播放量为"暂停"时，验证API设计是否正确
   assert api_design.playback_level == DesignReview.ACTION_PLAYBACK_PAUSED
   
   # 创建一个测试用例，当播放量为"上一曲"时，验证API设计是否正确
   assert api_design.playback_level == DesignReview.ACTION_PLAYBACK_PREV_VIDEO
   
   # 创建一个测试用例，当播放量为"下一曲"时，验证API设计是否正确
   assert api_design.playback_level == DesignReview.ACTION_PLAYBACK_NEXT_VIDEO
   
   # 播放音乐
   api_design.playback_level = DesignReview.ACTION_PLAYBACK_ACTIVE
   await api_design.run_async()
```
这段代码是一个Python脚本，用于测试设计API评审的功能。具体来说，它通过编写测试用例来验证API设计在不同播放量下的正确性。

当播放量为"null"时，验证API设计是否正确，即不支持播放音乐。当播放量为"正在播放"时，验证API设计是否正确，即支持播放音乐。当播放量为"暂停"时，验证API设计是否正确，即暂停播放音乐。当播放量为"上一曲"时，验证API设计是否正确，即支持播放音乐并可以前进或后退。当播放量为"下一曲"时，验证API设计是否正确，即支持播放音乐并可以前进或后退。

测试用例中的每个函数都使用asyncio让它们可以作为异步函数运行。最后，这段代码使用DesignReview类创建了一个API设计实例，并使用run_async函数运行它。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 19:31
@Author  : alexanderwu
@File    : test_design_api_review.py
"""
import pytest

from metagpt.actions.design_api_review import DesignReview


@pytest.mark.asyncio
async def test_design_api_review():
    prd = "我们需要一个音乐播放器，它应该有播放、暂停、上一曲、下一曲等功能。"
    api_design = """
```

这段代码定义了一个简单的API列表，用于在线播放音乐。其中包括以下功能：

1. `play(song: Song)`：开始播放指定的歌曲，参数`song`是一个包含歌曲信息的歌单（Song）对象。
2. `pause()`：暂停当前播放的歌曲。
3. `next()`：跳到播放列表的下一首歌曲。
4. `previous()`：跳到播放列表的上一首歌曲。

代码中包含一个名为`design_api_review`的设计评审接口，用于评估API设计的合理性。此外，代码中还包含一个名为`prd`的产品需求文档（PRD），用于描述该产品的需求。最后，代码会输出一个产品需求文档（PRD），并要求对接该API设计的团队进行评审。


```py
数据结构:
1. Song: 包含歌曲信息，如标题、艺术家等。
2. Playlist: 包含一系列歌曲。

API列表:
1. play(song: Song): 开始播放指定的歌曲。
2. pause(): 暂停当前播放的歌曲。
3. next(): 跳到播放列表的下一首歌曲。
4. previous(): 跳到播放列表的上一首歌曲。
"""
    _ = "API设计看起来非常合理，满足了PRD中的所有需求。"

    design_api_review = DesignReview("design_api_review")

    result = await design_api_review.run(prd, api_design)

    _ = f"以下是产品需求文档(PRD):\n\n{prd}\n\n以下是基于这个PRD设计的API列表:\n\n{api_design}\n\n请审查这个API设计是否满足PRD的需求，以及是否符合良好的设计实践。"
    # mock_llm.ask.assert_called_once_with(prompt)
    assert len(result) > 0

```

# `tests/metagpt/actions/test_detail_mining.py`

该代码是一个Python脚本，用于测试一个名为“detail_mining”的模型的功能。该模型使用名为“metagpt”的人工智能机制，可以自动生成文本描述，包括对生日蛋糕制作的步骤和建议。

以下是代码的主要部分：

```pypython
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/13 00:26
@Author  : fisherdeng
@File    : test_detail_mining.py
"""
import pytest

from metagpt.actions.detail_mining import DetailMining
from metagpt.logs import logger

@pytest.mark.asyncio
async def test_detail_mining():
   topic = "如何做一个生日蛋糕"
   record = "我认为应该先准备好材料，然后再开始做蛋糕。"
   detail_mining = DetailMining("detail_mining")
   rsp = await detail_mining.run(topic=topic, record=record)
   logger.info(f"{rsp.content=}")
   
   assert '##OUTPUT' in rsp.content
   assert '蛋糕' in rsp.content
```

首先，我们导入 `pytest`、`metagpt.actions.detail_mining` 和 `metagpt.logs` 模块。

然后，我们定义一个名为 `test_detail_mining` 的测试函数。

接着，我们定义一个名为 `topic` 的变量，用于存储要测试的蛋糕制作主题。

然后，我们定义一个名为 `record` 的变量，用于存储要在模型中生成的文本描述。

接下来，我们实例化名为 `detail_mining` 的模型。

接着，我们使用 `run` 方法将模型运行，并接收两个参数：`topic` 和 `record`。

最后，我们使用 `logger.info` 方法输出生成的内容，并使用 `assert` 语句检查结果。

请注意，为了使此代码具有可读性，我将导出的日志信息的字符串格式化为 `{}`。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/13 00:26
@Author  : fisherdeng
@File    : test_detail_mining.py
"""
import pytest

from metagpt.actions.detail_mining import DetailMining
from metagpt.logs import logger

@pytest.mark.asyncio
async def test_detail_mining():
    topic = "如何做一个生日蛋糕"
    record = "我认为应该先准备好材料，然后再开始做蛋糕。"
    detail_mining = DetailMining("detail_mining")
    rsp = await detail_mining.run(topic=topic, record=record)
    logger.info(f"{rsp.content=}")
    
    assert '##OUTPUT' in rsp.content
    assert '蛋糕' in rsp.content


```

# `tests/metagpt/actions/test_invoice_ocr.py`

这段代码是一个Python脚本，它使用`/usr/bin/env python3`作为解释器，用于运行Python 3程序。

该脚本导入了`os`和`typing.List`类型，分别用于操作系统路径和列表类型的导入。

接下来，定义了一个函数`test_invoice_ocr()`，它没有具体的实现，只是一个函数声明。

在函数声明下面，使用`@pytest.mark.asyncio`注解，表示该函数是一个异步函数，将会使用Python的`asyncio`库实现异步操作。

最后，在函数体内部，由于没有具体的实现，所以没有产生任何输出。


```py
#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

"""
@Time    : 2023/10/09 18:40:34
@Author  : Stitch-z
@File    : test_invoice_ocr.py
"""

import os
from typing import List

import pytest
from pathlib import Path

```

这段代码使用了Python标准库中的 metagpt.actions.invoice_ocr 模块中的 InvoiceOCR、GenerateTable 和 ReplyQuestion 类对发票进行 OCR 识别，并将识别结果保存到指定文件夹中。

具体来说，这段代码通过使用 `parametrize` 函数，让该函数可以生成不同的测试用例。在 `test_invoice_ocr` 函数中，使用了 `os.path.abspath` 和 `os.path.basename` 函数获取文件绝对路径和文件名，然后将这些参数作为参数传递给 `InvoiceOCR` 类中的 `run` 方法，用于运行 OCR 识别。

如果识别结果为空，该函数会抛出 `RuntimeError`，并打印出详细的错误信息。


```py
from metagpt.actions.invoice_ocr import InvoiceOCR, GenerateTable, ReplyQuestion


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invoice_path",
    [
        "../../data/invoices/invoice-3.jpg",
        "../../data/invoices/invoice-4.zip",
    ]
)
async def test_invoice_ocr(invoice_path: str):
    invoice_path = os.path.abspath(os.path.join(os.getcwd(), invoice_path))
    filename = os.path.basename(invoice_path)
    resp = await InvoiceOCR().run(file_path=Path(invoice_path), filename=filename)
    assert isinstance(resp, list)


```

这段代码使用了参数化编程来定义一个函数测试，具体作用如下：

1. `@pytest.mark.asyncio` 是 pytest 加载器的标记，用于告知 pytest 可以使用异步编程来编写测试。

2. `@pytest.mark.parametrize(...)` 是参数化函数的标记，用于定义一个参数组，其中的参数可以在测试函数中被使用，也可以在测试函数之外的其他地方使用。

3. `("invoice_path", "expected_result")` 是参数组中的参数，其中 `invoice_path` 是参数组的第一个参数，`expected_result` 是参数组的第二个参数。

4. `[(...)...)` 是参数组的括号，用于将参数组中的参数与括号中的内容进行绑定，其中每个参数组中的参数都对应于 `parametrize` 函数中的 `param` 参数，而括号中的内容则是参数组中参数的默认值或者描述。

5. `[("../../data/invoices/invoice-1.pdf",), ("../../data/invoices/invoice-2.pdf")]` 是两个参数组，分别对应于 `invoice_path` 参数的默认值和描述，用于测试不同的发票文件。

6. `)` 是一个占位符，表示参数组的结束。

7. 以上代码会在测试函数中随机生成两个参数，分别是 `invoice_path` 和 `expected_result`，并且分别传入一个包含两个公司的发票文件，测试函数会输出一个预期结果，即两个文件中应该包含不同的发票信息。


```py
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("invoice_path", "expected_result"),
    [
        (
            "../../data/invoices/invoice-1.pdf",
            [
                {
                    "收款人": "小明",
                    "城市": "深圳市",
                    "总费用/元": "412.00",
                    "开票日期": "2023年02月03日"
                }
            ]
        ),
    ]
)
```

这段代码使用了Python的异步编程库——asyncio，定义了一个名为`test_generate_table`的函数。这个函数接收两个参数：`invoice_path`（invoice文件的绝对路径）和`expected_result`（预期输出的表格数据）。

函数内部先将`invoice_path`转换为绝对路径，然后使用`os.path.abspath`获取绝对路径，再使用`os.path.basename`获取文件名。接下来分别定义了两个使用`asyncio`库的函数：`InvoiceOCR()`和`GenerateTable()`。这两个函数的作用是读取OCR结果和生成表格数据，具体实现可能因具体的需求而异。

函数内部还有一个`asyncio`库的注解，`@pytest.mark.asyncio`，这个注解用于告知Python的测试框架pytest将这个函数设置为异步函数，并且在测试过程中使用`await`来模拟函数的执行。

最后，函数内部还有一个`parametrize`函数，用于对参数进行枚举，生成一个包含多个参数的元组，这个函数的第一个参数是一个包含参数名称和参数描述的元组，后面的参数会一个一个被赋值，例如：`("../../data/invoices/invoice-1.pdf", "Invoicing date", "2023年02月03日")`，这个函数会生成一个包含参数`file_path`、`query`和`expected_result`的元组，依次赋值为`"../../data/invoices/invoice-1.pdf"`、`"Invoicing date"`和`"2023年02月03日"`。


```py
async def test_generate_table(invoice_path: str, expected_result: list[dict]):
    invoice_path = os.path.abspath(os.path.join(os.getcwd(), invoice_path))
    filename = os.path.basename(invoice_path)
    ocr_result = await InvoiceOCR().run(file_path=Path(invoice_path), filename=filename)
    table_data = await GenerateTable().run(ocr_results=ocr_result, filename=filename)
    assert table_data == expected_result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("invoice_path", "query", "expected_result"),
    [
        ("../../data/invoices/invoice-1.pdf", "Invoicing date", "2023年02月03日")
    ]
)
```

这段代码定义了一个名为 `test_reply_question` 的函数，它接受三个参数：

- `invoice_path`: 一个字符串，指定了一个发票文件的路径。
- `query`: 一个字典，包含一个问题。
- `expected_result`: 一个字符串，指定了一个预期的结果。

函数内部首先使用 `os.path.abspath` 函数获取发票文件的完整路径，然后使用 `os.path.basename` 函数获取发票文件的名称。接下来，使用 `InvoiceOCR()` 类中的 `run()` 方法将文件路径和名称作为参数传入，得到 OCR 结果。然后，使用 `ReplyQuestion()` 类中的 `run()` 方法将问题和 OCR 结果作为参数传入，得到结果。最后，使用 `assert` 语句检查结果是否与预期结果相同。

这段代码的作用是测试一个简单的 OCR 系统的功能，该系统可以回复从用户输入的问题和发票文件中提取出来的问题。


```py
async def test_reply_question(invoice_path: str, query: dict, expected_result: str):
    invoice_path = os.path.abspath(os.path.join(os.getcwd(), invoice_path))
    filename = os.path.basename(invoice_path)
    ocr_result = await InvoiceOCR().run(file_path=Path(invoice_path), filename=filename)
    result = await ReplyQuestion().run(query=query, ocr_result=ocr_result)
    assert expected_result in result


```

# `tests/metagpt/actions/test_project_management.py`

该代码是一个Python脚本，用于测试项目管理中的创建项目计划和分配任务功能。

具体来说，该脚本会创建两个测试类：TestCreateProjectPlan和TestAssignTasks。TestCreateProjectPlan类用于测试创建项目计划功能，TestAssignTasks类用于测试分配任务功能。

在脚本中，没有对类或函数进行定义。因此，您需要自己定义这些类或函数才能使用它们。

如果您想让脚本具有更多信息可读性，可以添加一些文档或注释。在当前脚本中，有几个简单的注释，但仍有改进的空间。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 19:12
@Author  : alexanderwu
@File    : test_project_management.py
"""


class TestCreateProjectPlan:
    pass


class TestAssignTasks:
    pass

```

# `tests/metagpt/actions/test_run_code.py`

该代码是一个Python脚本，使用了`pytest`模块进行测试驱动开发。

该脚本的主要作用是测试运行代码（RunCode）的`run_text`方法。`run_text`方法接受一个字符串参数，代表要运行的代码。

具体来说，以下是针对`test_run_text`测试用例的实现：

1. 导入`metagpt.actions.run_code`模块的`RunCode`类。
2. 在`test_run_text`测试用例外部，定义`@pytest.mark.asyncio`装饰器，用于将所有测试用例标记为异步函数。
3. 在`test_run_text`测试用例内部，实现两个函数：`test_run_text`和`test_run_text_with_error_message`。这两个函数分别测试运行代码中的两个不同的参数，并记录测试结果和错误信息。
4. 在`test_run_text`测试用例内部，使用`await`关键字，解开函数内部紧跟着的异常（如`asyncio.baseisclose`），并返回结果和错误信息，以便于将结果记录到测试报告中和显示错误信息。
5. 在`test_run_text_with_error_message`测试用例内部，使用`assert`关键字，测试运行代码中的错误信息是否正确，并记录到测试报告中。
6. 使用`@pytest.mark.asyncio`装饰器，将`test_run_text`和`test_run_text_with_error_message`两个测试用例都作为异步函数，这样就可以让它们在同一个协程中运行，方便日后的结果记录和错误信息的通知。


```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:46
@Author  : alexanderwu
@File    : test_run_code.py
"""
import pytest

from metagpt.actions.run_code import RunCode


@pytest.mark.asyncio
async def test_run_text():
    result, errs = await RunCode.run_text("result = 1 + 1")
    assert result == 2
    assert errs == ""

    result, errs = await RunCode.run_text("result = 1 / 0")
    assert result == ""
    assert "ZeroDivisionError" in errs


```

这段代码使用了Python的pytest库进行测试，包括两个测试函数：

1. test\_run\_script()函数是异步函数，使用了Python 3.7+的asyncio特性。该函数运行一个命令行工具"."和一个名为"RunCode"的类，并传入一个命令参数列表。这个命令参数列表包含一个名为"echo"的命令和一个参数"'Hello World'"，以及一个名为"print(1/0)"的命令。函数的作用是接收输出和错误并验证是否与期望的结果一致。

2. test\_run()函数也是异步函数，使用了Python 3.7+的asyncio特性。该函数使用RunCode类，并传递一个代码字符串作为参数。函数的作用是运行这个代码字符串，并验证返回的结果是否为"PASS"。


```py
@pytest.mark.asyncio
async def test_run_script():
    # Successful command
    out, err = await RunCode.run_script(".", command=["echo", "Hello World"])
    assert out.strip() == "Hello World"
    assert err == ""

    # Unsuccessful command
    out, err = await RunCode.run_script(".", command=["python", "-c", "print(1/0)"])
    assert "ZeroDivisionError" in err


@pytest.mark.asyncio
async def test_run():
    action = RunCode()
    result = await action.run(mode="text", code="print('Hello, World')")
    assert "PASS" in result

    result = await action.run(
        mode="script",
        code="echo 'Hello World'",
        code_file_name="",
        test_code="",
        test_file_name="",
        command=["echo", "Hello World"],
        working_directory=".",
        additional_python_paths=[],
    )
    assert "PASS" in result


```

这段代码使用了Python的pytest库，用于编写和运行测试。主要目的是测试使用异步io操作时，如何处理运行代码的失败情况。

具体来说，这段代码包括以下内容：

1.定义了一个名为`test_run_failure`的测试函数，使用了Python的mark.asyncio注解，说明该函数使用异步io操作。

2.内部使用了RunCode类，该类可能从参数中获取代码作为参数，并使用asyncio操作将代码运行起来。但不知道代码具体是什么，因此使用mark.asyncio注解将该函数标记为异步io操作。

3.在函数内部，使用了两个await语句，分别是在运行代码时，通过两个不同的模式运行代码，并输出运行结果。

4.第一个运行代码的模式是`mode="text"`，意味着使用文本模式运行代码。在该模式下，如果运行失败，将在控制台上输出"FAIL"。

5.第二个运行代码的模式是`mode="script"`，意味着使用脚本模式运行代码。在该模式下，运行代码的失败信息可能会有所不同，需要通过参数来指定。但是该函数的第二个参数是空字符串，因此不会使用该模式。

6.在函数内部，通过两个不同的命令行参数传递给运行代码的函数，分别是打印1除以0的值。

7.最后，在函数内部，使用assert语句来验证运行结果中是否包含"FAIL"。


```py
@pytest.mark.asyncio
async def test_run_failure():
    action = RunCode()
    result = await action.run(mode="text", code="result = 1 / 0")
    assert "FAIL" in result

    result = await action.run(
        mode="script",
        code='python -c "print(1/0)"',
        code_file_name="",
        test_code="",
        test_file_name="",
        command=["python", "-c", "print(1/0)"],
        working_directory=".",
        additional_python_paths=[],
    )
    assert "FAIL" in result

```

# `tests/metagpt/actions/test_ui_design.py`

这段代码的作用是定义一个名为 "UI Design Description" 的函数，它返回一个字符串，描述了游戏界面的设计描述。函数中包含了一系列描述游戏界面的元素，例如游戏网格、蛇、食物、得分和游戏结束消息。这些元素将被设计为简单、干净和直观。

接下来，定义了一个名为 "Selected Elements" 的函数，它返回一个包含游戏界面上特定元素的字符串。这个函数将游戏网格、蛇和食物作为参数，将它们都定义为游戏界面的元素。然后，这三种类型的元素都使用了不同的颜色，使它们在屏幕上更加显眼。


```py
# -*- coding: utf-8 -*-
# @Date    : 2023/7/22 02:40
# @Author  : stellahong (stellahong@fuzhi.ai)
#
from tests.metagpt.roles.ui_role import UIDesign

llm_resp= '''
    # UI Design Description
```The user interface for the snake game will be designed in a way that is simple, clean, and intuitive. The main elements of the game such as the game grid, snake, food, score, and game over message will be clearly defined and easy to understand. The game grid will be centered on the screen with the score displayed at the top. The game controls will be intuitive and easy to use. The design will be modern and minimalist with a pleasing color scheme.```py

## Selected Elements

Game Grid: The game grid will be a rectangular area in the center of the screen where the game will take place. It will be defined by a border and will have a darker background color.

Snake: The snake will be represented by a series of connected blocks that move across the grid. The color of the snake will be different from the background color to make it stand out.

```

This code is an HTML file that creates a simple game board for a snake game.

The game board consists of a grid of 20 cells, with a small food element randomly placed at the center.

The snake game is controlled by a player who moves the snake using the arrow keys.

The score is displayed at the top of the screen and increases each time the player eats a piece of food.

When the game is over, a message is displayed in the center of the screen, giving the player the option to restart the game.


```py
Food: The food will be represented by small objects that are a different color from the snake and the background. The food will be randomly placed on the grid.

Score: The score will be displayed at the top of the screen. The score will increase each time the snake eats a piece of food.

Game Over: When the game is over, a message will be displayed in the center of the screen. The player will be given the option to restart the game.

## HTML Layout
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Snake Game</title>
    <link rel="stylesheet" href="styles.css">
```py

这段代码是一个 HTML 页面，包含一个分数（Score）标签、一个游戏网格（game-grid）区域以及一个游戏结束（game-over）标签。在分数标签内部，显示了当前的得分，为0。在游戏网格区域，可能会有蛇（Snake）和食物（Food）的相关信息，这些信息可能会在 JavaScript 中进行动态生成。最后，使用了一个 div 元素来包含游戏结束标签。整个页面的布局采用了 Flexbox 布局，游戏网格区域使用了一个嵌套的 div 元素来包含游戏元素，游戏元素可能是在 JavaScript 中进行动态生成的。


```
</head>
<body>
    <div class="score">Score: 0</div>
    <div class="game-grid">
        <!-- Snake and food will be dynamically generated here using JavaScript -->
    </div>
    <div class="game-over">Game Over</div>
</body>
</html>
```py

## CSS Styles (styles.css)
```css
body {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100vh;
    margin: 0;
    background-color: #f0f0f0;
}

```py

这段代码的作用是定义两个类：.score和.game-grid。

.score类包含以下样式：
```css
font-size: 2em;
margin-bottom: 1em;
```py

.game-grid类包含以下样式：
```css
width: 400px;
height: 400px;
display: grid;
grid-template-columns: repeat(20, 1fr);
grid-template-rows: repeat(20, 1fr);
gap: 1px;
background-color: #222;
border: 1px solid #555;
```py

Score类的作用可能不是很清楚，但根据名字和这些样式，我猜测它可能是用来设置游戏得分视觉效果的。grid-template-columns和grid-template-rows属性似乎用于将游戏面板分割成20行20列的网格，然后设定了网格中每个单元格的大小（通过repeat()函数重复使用相同的宽度1fr来使网格中的单元格大小相同）。gap和background-color属性似乎用于在网格和得分之间添加一些空间和背景颜色。


```
.score {
    font-size: 2em;
    margin-bottom: 1em;
}

.game-grid {
    width: 400px;
    height: 400px;
    display: grid;
    grid-template-columns: repeat(20, 1fr);
    grid-template-rows: repeat(20, 1fr);
    gap: 1px;
    background-color: #222;
    border: 1px solid #555;
}

```py

这段代码是一段CSS（客户端样式表）代码，为网页元素的应用样式。具体解释如下：

1. `.snake-segment {` 是开始标签，表示这是一个类（ID选择器）。类选择器是CSS中的一种特殊选择器，用于选择某个元素的特定样式。这里的类选择器指定了背景颜色为灰绿色（#00cc66）。

2. `.food {` 同样是开始标签，但表示一个类选择器。这里的类选择器指定了背景颜色为橙红色（#cc3300）。

3. `.control-panel {` 是开始标签，表示这是一个容器元素。容器元素在CSS中有很多用途，例如划分页面结构、容纳其他元素等。这里的类选择器指定了宽度为400px（400像素），并将其居中分配给其它子元素。同时，通过设置 margin-top: 1em（外边距为1em，即10像素），使该容器元素稍微高于其它子元素。

4. `我为网页添加了一些样式。` 这句话是强调，说明这段代码的作用是为网页添加了一些样式。具体来说，通过设置背景颜色、设置背景图像等，使得网页呈现出特定的外观。


```
.snake-segment {
    background-color: #00cc66;
}

.food {
    background-color: #cc3300;
}

.control-panel {
    display: flex;
    justify-content: space-around;
    width: 400px;
    margin-top: 1em;
}

```py

这段代码实际上是创建了一个 HTML 元素，一个 CSS 类，以及一些 JavaScript 代码。我会逐点解释这段代码的作用。

1. `.control-button` 是一个 CSS 类，表示一个控制按钮。这个按钮的样式包括内边距（padding）、字体大小（font-size）、边框（border）以及背景颜色（background-color）。同时，这个按钮还绑定了鼠标事件（cursor），使得用户可以使用鼠标进行点击操作。

2. `.game-over` 是一个 CSS 类，表示一个游戏胜利或者输出的消息。这个消息在页面上以绝对定位（position: absolute）的方式固定在页面的顶部，然后以左旋括号（left, top: 50%, left: 50%）的坐标被移动到该位置。这个类使用了繁体中文字体。

3. `.control-button` 和 `.game-over` 都使用了 ID 选择器（`:hover` 和 `:abort`）。ID 选择器可以让他们在鼠标悬停或者点击按钮时被触发，而 `:hover` 和 `:abort` 则是 CSS 选择器，表示当鼠标悬停或者点击按钮时，这些类或者元素的特定样式会被应用。


```
.control-button {
    padding: 1em;
    font-size: 1em;
    border: none;
    background-color: #555;
    color: #fff;
    cursor: pointer;
}

.game-over {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 3em;
    '''

```py

这段代码是一个测试用例，名为 "test_ui_design_parse_css"，用于测试 UI 设计组件中的 CSS 样式是否正确。具体来说，这段代码定义了一个名为 "UI design action" 的 UI 设计组件，该组件的样式中包含以下内容：

1. "body" 元素的样式设置，包括设置其为 "display: flex"，"flex-direction: column"，"justify-content: center"，"align-items: center"，以及设置其高度为 "100vh"（即 100% 的视口高度）。
2. 一个名为 "score" 的元素的样式设置，包括设置其字体大小为 "2em"， margin 属性中包含一个垂直方向上单留空一行，并设置其样式中包含一个 "font-size: 2em"（即 2 倍的字体大小）和 "margin-bottom: 1em"（即 1 倍的垂直距离）。

这段代码的作用是测试 UI 设计组件中的 CSS 样式是否符合预期，如果出现错误则可以通过调试逐步排除问题。


```
def test_ui_design_parse_css():
    ui_design_work = UIDesign(name="UI design action")

    css = '''
    body {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100vh;
    margin: 0;
    background-color: #f0f0f0;
}

.score {
    font-size: 2em;
    margin-bottom: 1em;
}

```py

这段代码创建了一个游戏网格，宽度为400px，高度为400px，使用grid布局将元素居中布局。网格有两个子网格，每个子网格有20行和20列，行和列之间用1px的间隔填充。网格的背景颜色为灰色，边框为1像素厚度的黑色。

.snake-segment 是一个类，定义了一个蛇形分割 segment的背景颜色为紫色，以便在游戏中更容易识别。


```
.game-grid {
    width: 400px;
    height: 400px;
    display: grid;
    grid-template-columns: repeat(20, 1fr);
    grid-template-rows: repeat(20, 1fr);
    gap: 1px;
    background-color: #222;
    border: 1px solid #555;
}

.snake-segment {
    background-color: #00cc66;
}

```py

这段代码是一段CSS（层叠样式表）代码，用于为网页添加样式。具体来说，这段代码定义了一个名为“food”的类，设置了其背景颜色为灰绿色（#cc3300）。这个类被应用于“.food”选择器所指定的元素。

接下来，定义了一个名为“control-panel”的类，设置其display属性为flex，并设置宽度为400px，margin-top为1em。元素padding，font-size和border属性。设置其背景颜色为淡灰色（#555），并设置其文本颜色为白色。定义了一个名为“control-button”的类，设置其padding为1em，font-size为1em，border为none，背景颜色为淡灰色，文本颜色为白色，并设置其初始状态为鼠标悬停状态（cursor: pointer;）。


```
.food {
    background-color: #cc3300;
}

.control-panel {
    display: flex;
    justify-content: space-around;
    width: 400px;
    margin-top: 1em;
}

.control-button {
    padding: 1em;
    font-size: 1em;
    border: none;
    background-color: #555;
    color: #fff;
    cursor: pointer;
}

```py

这段代码的作用是创建一个游戏结束的标签，并将其位置在屏幕的左上角。这个标签的宽度和高都是3倍，字体大小也是3倍。

这段代码是在使用UI设计工具（UIDesign） parsing CSS code时执行的。UI设计工具是一个类，可以解析CSS代码并返回相应的UI设计动作。在这里，UI设计工具的parse_css_code方法被调用来解析一个CSS文件，然后将返回的UI设计动作赋值给变量ui_design_work。最后，ui_design_work对象的一个assert语句确保UI设计动作正确解析，但不会输出任何调试信息。


```
.game-over {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 3em;
    '''
    assert ui_design_work.parse_css_code(context=llm_resp)==css


def test_ui_design_parse_html():
    ui_design_work = UIDesign(name="UI design action")

    html = '''
    <!DOCTYPE html>
```py

这段代码是一个网页，其中包含一个游戏面板和一个得分板。游戏面板包括一个 snake 对象，它会在每个时间步移动一定距离，并且会吃掉一个随机生成的食物。得分板显示了当前的得分。

具体来说，这个代码实现了一个 Snake Game 的前端部分。在 HTML 文件中，定义了一些元素，包括一个 div 元素用来显示游戏得分和 div 元素用来显示游戏界面的基本样式。此外，还引入了一个 styles.css 文件，用来设置游戏界面的样式。

在 JavaScript 文件中，定义了一个 Snake 对象，用来存储所有元素的位置。在每个时间步，Snake 对象会移动一定距离，并且会吃掉一个随机生成的食物。在游戏结束时，显示一个游戏失败的 div。

最后，通过 assert 函数来确保页面可以正确渲染，并且通过将 score 和 gameOver div 元素的状态与页面中实际存在的元素进行比较，来确保它们可以正确显示。


```
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Snake Game</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="score">Score: 0</div>
    <div class="game-grid">
        <!-- Snake and food will be dynamically generated here using JavaScript -->
    </div>
    <div class="game-over">Game Over</div>
</body>
</html>
    '''
    assert ui_design_work.parse_css_code(context=llm_resp)==html




```py

# `tests/metagpt/actions/test_write_code.py`

这段代码是一个Python脚本，使用了Python的裸机器代码格式。脚本的作用是测试Metagpt中write_code包的行为。

具体来说，该脚本包含了以下操作：

1. 导入pytest，用于调用自己的测试套件。
2. 从metagpt包中导入WriteCode类，用于编写测试用例。
3. 从metagpt包中导入LLM类，用于测试中使用的LLM实例。
4. 导入logger，用于记录Metagpt的日志。
5. 在test_write_code.py目录下创建一个名为test_write_code.py的文件。
6. 在test_write_code.py文件中写入了如下代码：
```python
@pytest.fixture(scope="session")
def mock_write_code():
   mock = TASKS_2.write_code_prompt_sample
   yield mock
```py
7. 运行该脚本时，pytest会创建一个测试套件，并在其中运行所有测试用例。
8. 如果所有测试用例都通过了，那么该脚本会输出Metagpt的日志，以供开发人员参考。


```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:45
@Author  : alexanderwu
@File    : test_write_code.py
"""
import pytest

from metagpt.actions.write_code import WriteCode
from metagpt.llm import LLM
from metagpt.logs import logger
from tests.metagpt.actions.mock import TASKS_2, WRITE_CODE_PROMPT_SAMPLE


```py

这段代码使用了Python的异步io库，用于编写一个名为'add'的函数。该函数接受两个整数作为输入，并返回它们的和。

首先，定义了一个名为'write_code'的函数，这个函数使用了异步io库中的WriteCode类。WriteCode类是一个异步编程的标准库，它提供了一个用于编写异步代码的接口。

在test_write_code.py文件中，使用await write_code.run(api_design)方法运行WriteCode类的一个实例，其中api_design参数是一个字符串，表示待测用的函数的描述。

最后，通过比较生成的代码和预期代码，验证函数add是否存在于生成的代码中，并输出日志。


```
@pytest.mark.asyncio
async def test_write_code():
    api_design = "设计一个名为'add'的函数，该函数接受两个整数作为输入，并返回它们的和。"
    write_code = WriteCode("write_code")

    code = await write_code.run(api_design)
    logger.info(code)

    # 我们不能精确地预测生成的代码，但我们可以检查某些关键字
    assert 'def add' in code
    assert 'return' in code


@pytest.mark.asyncio
async def test_write_code_directly():
    prompt = WRITE_CODE_PROMPT_SAMPLE + '\n' + TASKS_2[0]
    llm = LLM()
    rsp = await llm.aask(prompt)
    logger.info(rsp)

```py

# `tests/metagpt/actions/test_write_code_review.py`

This code is a Python script using the `metagpt` library for writing code reviews. The `WriteCodeReview` class is used to perform the main test.

The script imports the `pytest` library and defines a test function `test_write_code_review`. This function uses the `capfd` parameter, which is a file descriptor object representing the standard input and standard output of the `pytest` command.

The `test_write_code_review` function creates an instance of the `WriteCodeReview` class and then calls its `review_code` method on this instance. The `review_code` method takes two arguments: the code to be reviewed (in this case, the code in the `code` parameter) and a filename to which the review should be written.

The code creates a file named `test_write_code_review.py` in the same directory as the script and writes its contents to the standard input of the `pytest` command. The `pytest` command then runs the script, passing in the `--cov` option to `pytest` to enable code coverage reports. The `--cov-report=xml` option is also passed to `pytest` to specify the format of the coverage report to generate.

If the `test_write_code_review` function runs successfully, the script is expected to output a message indicating that the code has been reviewed and the filename that the review was written to.


```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:45
@Author  : alexanderwu
@File    : test_write_code_review.py
"""
import pytest

from metagpt.actions.write_code_review import WriteCodeReview


@pytest.mark.asyncio
async def test_write_code_review(capfd):
    code = """
```py

这段代码定义了一个名为 add 的函数，它接收两个参数 a 和 b，然后返回它们的和。函数的实现仅包含两行代码。

代码作为函数被导出，我们无法直接访问它的实现。这个函数是通过 WriteCodeReview 函数进行运行的。

这段代码的作用是定义了一个加法函数 add ，它接受两个参数 a 和 b，并返回它们的和。函数的实现中包含两行代码，这两行代码定义了函数的接口，即函数需要接收两个参数并返回它们的和。


```
def add(a, b):
    return a + 
"""
    # write_code_review = WriteCodeReview("write_code_review")

    code = await WriteCodeReview().run(context="编写一个从a加b的函数，返回a+b", code=code, filename="math.py")

    # 我们不能精确地预测生成的代码评审，但我们可以检查返回的是否为字符串
    assert isinstance(code, str)
    assert len(code) > 0

    captured = capfd.readouterr()
    print(f"输出内容: {captured.out}")


```py

这段代码使用Python的异步io库编写了一个名为"asyncio"的测试标记函数。该函数内部定义了一个名为"test_write_code_review_directly"的测试函数。

具体来说，这段代码的作用是编写一个写代码的示例，并在另一个名为"WriteCodeReview"的类中使用该函数运行写入代码的评测。评测运行成功后，代码将作为"info"级别的事件输出一个包含评测信息的日志。


```
# @pytest.mark.asyncio
# async def test_write_code_review_directly():
#     code = SEARCH_CODE_SAMPLE
#     write_code_review = WriteCodeReview("write_code_review")
#     review = await write_code_review.run(code)
#     logger.info(review)

```py

# `tests/metagpt/actions/test_write_docstring.py`

这段代码包括以下几个部分：

1. 导入pytest库，用于用于pytest测试套件中的其他测试函数和模块。
2. 从metagpt库中引入了一个名为WriteDocstring的行动，该行动可以用于创建文档。
3. 定义了一个名为add\_numbers的函数，该函数接收两个整数参数，返回它们的和。
4. 定义了一个名为Person的类，该类包含一个__init__方法和一个greet方法。__init__方法用于初始化对象的属性，而greet方法用于在对象被调用时返回一个问候语，其中包含对象的名称和年龄。

这段代码的主要目的是创建一个用于测试的对象，该对象可以测量年龄。通过初始化对象并调用greet方法，可以测试对象如何正确地获取和设置年龄。


```
import pytest

from metagpt.actions.write_docstring import WriteDocstring

code = '''
def add_numbers(a: int, b: int):
    return a + b


class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def greet(self):
        return f"Hello, my name is {self.name} and I am {self.age} years old."
```py

这段代码使用了参数化编程的方式来编写一个测试，目的是为了测试不同编程语言的文档字符串生成器。

具体来说，这段代码定义了一个名为 `test_write_docstring` 的测试，使用了 `pytest.mark.asyncio` 和 `pytest.mark.parametrize` 这两个装饰器。其中，`parametrize` 装饰器允许将一个或多个参数传递给 `test_write_docstring` 函数，用于控制测试的运行参数。

在 `test_write_docstring` 函数中，使用了两个参数 `style` 和 `part`，分别对应于 `style` 和 `part` 两个参数的文档字符串生成器。通过将这两个参数作为元组传递给 `parametrize` 装饰器，使得 `test_write_docstring` 函数可以正确地接受一个或多个参数，并且可以生成符合 `style` 和 `part` 参数的文档字符串。

在 `test_write_docstring` 函数中，使用了 `asyncio` 包中的 `WriteDocstring` 类来生成文档字符串。这个类的参数是一个字符串 `style` 和两个字符串 `part` 和 `name`。通过调用 `WriteDocstring.run` 方法，并将生成器设置为 `style` 和 `part` 参数，可以生成符合 `style` 和 `part` 参数的文档字符串。最后，通过调用 `assert` 语句，检查 `part` 参数是否在返回的结果中出现了。


```
'''


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("style", "part"),
    [
        ("google", "Args:"),
        ("numpy", "Parameters"),
        ("sphinx", ":param name:"),
    ],
    ids=["google", "numpy", "sphinx"]
)
async def test_write_docstring(style: str, part: str):
    ret = await WriteDocstring().run(code, style=style)
    assert part in ret

```