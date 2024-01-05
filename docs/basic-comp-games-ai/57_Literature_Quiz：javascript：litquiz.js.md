# `d:/src/tocomm/basic-computer-games\57_Literature_Quiz\javascript\litquiz.js`

```
// 创建一个新的 Promise 对象，用于处理异步操作
// 创建一个 INPUT 元素，用于用户输入
// 在页面上打印问号，提示用户输入
// 设置 INPUT 元素的类型为文本输入
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 添加键盘按下事件监听器，当按下回车键时执行相应操作
input_element.addEventListener("keydown", function (event) {
    # 如果按下的键是回车键
    if (event.keyCode == 13) {
        # 将输入元素的值赋给输入字符串
        input_str = input_element.value;
        # 从 id 为 "output" 的元素中移除输入元素
        document.getElementById("output").removeChild(input_element);
        # 打印输入字符串
        print(input_str);
        # 打印换行符
        print("\n");
        # 解析并返回输入字符串
        resolve(input_str);
    }
});
# 结束键盘按下事件监听器的添加
});
# 结束函数定义

# 定义一个名为 tab 的函数，参数为 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
```
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串

}

// Main program
async function main()
{
    print(tab(25) + "LITERATURE QUIZ\n");  // 打印标题
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印作者信息
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    r = 0;  // 初始化变量 r 为 0
    print("TEST YOUR KNOWLEDGE OF CHILDREN'S LITERATURE.\n");  // 打印提示信息
    print("\n");  // 打印空行
    print("THIS IS A MULTIPLE-CHOICE QUIZ.\n");  // 打印提示信息
    print("TYPE A 1, 2, 3, OR 4 AFTER THE QUESTION MARK.\n");  // 打印提示信息
    print("\n");  // 打印空行
    print("GOOD LUCK!\n");  // 打印祝福信息
    print("\n");  // 打印空行
}
    # 打印空行
    print("\n");
    # 打印问题
    print("IN PINOCCHIO, WHAT WAS THE NAME OF THE CAT\n");
    # 打印选项
    print("1)TIGGER, 2)CICERO, 3)FIGARO, 4)GUIPETTO\n");
    # 从用户输入中获取答案并转换为整数
    a = parseInt(await input());
    # 如果答案是3
    if (a == 3) {
        # 打印正确提示
        print("VERY GOOD!  HERE'S ANOTHER.\n");
        # 答对题目数量加一
        r++;
    } else {
        # 打印错误提示
        print("SORRY...FIGARO WAS HIS NAME.\n");
    }
    # 打印空行
    print("\n");
    print("\n");
    # 打印问题
    print("FROM WHOSE GARDEN DID BUGS BUNNY STEAL THE CARROTS?\n");
    # 打印选项
    print("1)MR. NIXON'S, 2)ELMER FUDD'S, 3)CLEM JUDD'S, 4)STROMBOLI'S\n");
    # 从用户输入中获取答案并转换为整数
    a = parseInt(await input());
    # 如果答案是2
    if (a == 2) {
        # 打印正确提示
        print("PRETTY GOOD!\n");
        # 答对题目数量加一
        r++;
    } else {
        # 打印错误提示
        print("TOO BAD...IT WAS ELMER FUDD'S GARDEN.\n");
    }  # 结束 if 语句块
    print("\n");  # 打印空行
    print("\n");  # 打印空行
    print("IN THE WIZARD OF OS, DOROTHY'S DOG WAS NAMED\n");  # 打印提示信息
    print("1)CICERO, 2)TRIXIA, 3)KING, 4)TOTO\n");  # 打印选项
    a = parseInt(await input());  # 获取用户输入并转换为整数赋值给变量 a
    if (a == 4) {  # 如果用户输入为 4
        print("YEA!  YOU'RE A REAL LITERATURE GIANT.\n");  # 打印正确提示
        r++;  # 变量 r 自增 1
    } else {  # 如果用户输入不为 4
        print("BACK TO THE BOOKS,...TOTO WAS HIS NAME.\n");  # 打印错误提示
    }
    print("\n");  # 打印空行
    print("\n");  # 打印空行
    print("WHO WAS THE FAIR MAIDEN WHO ATE THE POISON APPLE\n");  # 打印提示信息
    print("1)SLEEPING BEAUTY, 2)CINDERELLA, 3)SNOW WHITE, 4)WENDY\n");  # 打印选项
    a = parseInt(await input());  # 获取用户输入并转换为整数赋值给变量 a
    if (a == 3) {  # 如果用户输入为 3
        print("GOOD MEMORY!\n");  # 打印正确提示
        r++;  # 变量 r 自增 1
    } else {
        # 如果 r 等于 4，则打印以下内容
        print("OH, COME ON NOW...IT WAS SNOW WHITE.\n");
    }
    # 打印两个空行
    print("\n");
    print("\n");
    if (r == 4) {
        # 如果 r 等于 4，则打印以下内容
        print("WOW!  THAT'S SUPER!  YOU REALLY KNOW YOUR NURSERY\n");
        print("YOUR NEXT QUIZ WILL BE ON 2ND CENTURY CHINESE\n");
        print("LITERATURE (HA, HA, HA)\n");
    } else if (r < 2) {
        # 如果 r 小于 2，则打印以下内容
        print("UGH.  THAT WAS DEFINITELY NOT TOO SWIFT.  BACK TO\n");
        print("NURSERY SCHOOL FOR YOU, MY FRIEND.\n");
    } else {
        # 如果 r 不等于 4 且不小于 2，则打印以下内容
        print("NOT BAD, BUT YOU MIGHT SPEND A LITTLE MORE TIME\n");
        print("READING THE NURSERY GREATS.\n");
    }
}

main();
```