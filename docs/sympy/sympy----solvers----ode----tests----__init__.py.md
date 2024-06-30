# `D:\src\scipysrc\sympy\sympy\solvers\ode\tests\__init__.py`

```
# 导入需要使用的模块
import os
import sys

# 定义一个类，表示图书馆
class Library:
    # 初始化方法，创建一个新的图书馆实例
    def __init__(self, book_list):
        # 将传入的书籍列表存储为实例变量
        self.book_list = book_list

    # 定义方法，显示图书馆中的书籍列表
    def display_books(self):
        # 遍历图书馆中的每本书，并逐行打印书名
        for book in self.book_list:
            print(book)

    # 定义方法，借书功能
    def lend_book(self, book):
        # 检查借阅的书籍是否在图书馆中
        if book in self.book_list:
            # 如果在图书馆中，则从书籍列表中移除
            self.book_list.remove(book)
            print(f"You have borrowed {book}")
        else:
            # 如果不在图书馆中，则打印错误信息
            print("Sorry, the book is not available.")

    # 定义方法，增加书籍到图书馆
    def add_book(self, book):
        # 将新书添加到图书馆的书籍列表中
        self.book_list.append(book)
        print(f"You have added {book} to the library.")

# 创建一个图书馆实例，初始书籍列表包含三本书
my_library = Library(['Book1', 'Book2', 'Book3'])

# 显示图书馆中的书籍列表
my_library.display_books()

# 借阅一本书
my_library.lend_book('Book2')

# 显示更新后的图书馆书籍列表
my_library.display_books()

# 添加一本新书到图书馆
my_library.add_book('Book4')

# 显示最终更新后的图书馆书籍列表
my_library.display_books()
```