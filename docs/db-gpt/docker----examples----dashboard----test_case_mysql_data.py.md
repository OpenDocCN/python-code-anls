# `.\DB-GPT-src\docker\examples\dashboard\test_case_mysql_data.py`

```py
import os  # 导入操作系统相关模块
import random  # 导入随机数生成模块
import string  # 导入字符串处理模块
from datetime import datetime, timedelta  # 导入日期时间相关模块
from typing import List  # 导入类型提示模块

import pymysql  # 导入 MySQL 数据库操作模块
import pymysql.cursors  # 导入 MySQL 数据库游标模块

# 创建一个名为 dbgpt_test 的测试数据库；
# 可使用下面的 SQL 命令来创建：
# CREATE DATABASE IF NOT EXISTS dbgpt_test CHARACTER SET utf8;

def build_table(connection):
    # 创建名为 user 的数据表，包含用户信息的各个字段
    connection.cursor().execute(
        """CREATE TABLE user (
              id INT(11) NOT NULL AUTO_INCREMENT COMMENT '用户ID',
              name VARCHAR(50) NOT NULL COMMENT '用户名',
              email VARCHAR(50) NOT NULL COMMENT '电子邮件',
              mobile CHAR(11) NOT NULL COMMENT '手机号码',
              gender VARCHAR(20) COMMENT '性别，可选值：Male, Female',
              birth DATE COMMENT '出生日期',
              country VARCHAR(20) COMMENT '国家',
              city VARCHAR(20) COMMENT '城市',
              create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
              update_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
              PRIMARY KEY (id),
              UNIQUE KEY uk_email (email)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户信息表';"""
    )
    # 创建名为 transaction_order 的数据表，记录交易订单信息的各个字段
    connection.cursor().execute(
        """CREATE TABLE transaction_order (
              id INT(11) NOT NULL AUTO_INCREMENT COMMENT '订单ID',
              order_no CHAR(20) NOT NULL COMMENT '订单编号',
              product_name VARCHAR(50) NOT NULL COMMENT '产品名称',
              product_category VARCHAR(20) COMMENT '产品分类',
              amount DECIMAL(10, 2) NOT NULL COMMENT '订单金额',
              pay_status VARCHAR(20) COMMENT '付款状态，可选值：CANCEL, REFUND, SUCCESS, FAILD',
              user_id INT(11) NOT NULL COMMENT '用户ID',
              user_name VARCHAR(50) COMMENT '用户名',
              create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
              update_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
              PRIMARY KEY (id),
              UNIQUE KEY uk_order_no (order_no),
              KEY idx_user_id (user_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='交易订单表';"""
    )

def user_build(names: List, country: str, grander: str = "Male") -> List:
    countries = ["China", "US", "India", "Indonesia", "Pakistan"]  # 国家列表
    cities = {
        "China": ["Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Hangzhou"],
        "US": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
        "India": ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai"],
        "Indonesia": ["Jakarta", "Surabaya", "Medan", "Bandung", "Makassar"],
        "Pakistan": ["Karachi", "Lahore", "Faisalabad", "Rawalpindi", "Multan"],
    }

    users = []  # 初始化用户列表
    # 遍历从 1 到 names 列表长度的范围
    for i in range(1, len(names) + 1):
        # 如果性别为男性
        if grander == "Male":
            # 计算用户ID，格式为国家索引加上后缀"10"再加上当前循环索引i
            id = int(str(countries.index(country) + 1) + "10") + i
        else:
            # 计算用户ID，格式为国家索引加上后缀"20"再加上当前循环索引i
            id = int(str(countries.index(country) + 1) + "20") + i

        # 获取当前索引i对应的姓名
        name = names[i - 1]
        # 根据姓名生成示例邮箱地址
        email = f"{name}@example.com"
        # 生成随机的10位数字字符串作为手机号码
        mobile = "".join(random.choices(string.digits, k=10))
        # 使用给定的性别作为用户的性别
        gender = grander
        # 生成随机的出生日期，格式为19YY-MM-DD
        birth = f"19{random.randint(60, 99)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        # 将当前国家设置为用户的国家
        country = country
        # 随机选择用户所在城市，从给定国家的城市列表中
        city = random.choice(cities[country])

        # 获取当前时间
        now = datetime.now()
        # 获取当前年份
        year = now.year

        # 设置当前年份的起始时间和结束时间
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31)
        # 随机生成在当前年份范围内的日期
        random_date = start + timedelta(days=random.randint(0, (end - start).days))
        # 随机生成一天中的时间
        random_time = datetime.combine(random_date, datetime.min.time()) + timedelta(
            seconds=random.randint(0, 24 * 60 * 60 - 1)
        )

        # 将随机生成的日期和时间转换为字符串格式
        random_datetime_str = random_time.strftime("%Y-%m-%d %H:%M:%S")
        # 将生成时间作为用户的创建时间
        create_time = random_datetime_str

        # 将生成的用户信息作为元组加入到用户列表中
        users.append(
            (
                id,
                name,
                email,
                mobile,
                gender,
                birth,
                country,
                city,
                create_time,
                create_time,  # 这里两次添加create_time是因为用户信息包含创建时间和更新时间，格式相同
            )
        )

    # 返回生成的用户列表
    return users
# 定义一个函数用于生成所有用户信息并插入到数据库中
def gnerate_all_users(cursor):
    # 初始化一个空列表用于存储所有用户信息
    users = []
    
    # 第一组中国男性用户
    users_f = ["ZhangWei", "LiQiang", "ZhangSan", "LiSi"]
    # 调用 user_build 函数生成用户信息并加入到 users 列表中
    users.extend(user_build(users_f, "China", "Male"))
    
    # 第二组中国女性用户
    users_m = ["Hanmeimei", "LiMeiMei", "LiNa", "ZhangLi", "ZhangMing"]
    # 调用 user_build 函数生成用户信息并加入到 users 列表中
    users.extend(user_build(users_m, "China", "Female"))

    # 第一组美国男性用户
    users1_f = ["James", "John", "David", "Richard"]
    # 调用 user_build 函数生成用户信息并加入到 users 列表中
    users.extend(user_build(users1_f, "US", "Male"))
    
    # 第二组美国女性用户
    users1_m = ["Mary", "Patricia", "Sarah"]
    # 调用 user_build 函数生成用户信息并加入到 users 列表中
    users.extend(user_build(users1_m, "US", "Female"))

    # 第一组印度男性用户
    users2_f = ["Ravi", "Rajesh", "Ajay", "Arjun", "Sanjay"]
    # 调用 user_build 函数生成用户信息并加入到 users 列表中
    users.extend(user_build(users2_f, "India", "Male"))
    
    # 第二组印度女性用户
    users2_m = ["Priya", "Sushma", "Pooja", "Swati"]
    # 调用 user_build 函数生成用户信息并加入到 users 列表中
    users.extend(user_build(users2_m, "India", "Female"))
    
    # 遍历所有生成的用户信息
    for user in users:
        # 执行 SQL 插入语句，将用户信息插入到 user 表中
        cursor.execute(
            "INSERT INTO user (id, name, email, mobile, gender, birth, country, city, create_time, update_time) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            user,
        )

    # 返回所有生成的用户信息列表
    return users


# 定义一个函数用于生成所有订单信息并插入到数据库中
def gnerate_all_orders(users, cursor):
    # 初始化一个空列表用于存储所有订单信息
    orders = []
    # 定义订单数量
    orders_num = 200
    # 定义商品分类列表
    categories = ["Clothing", "Food", "Home Appliance", "Mother and Baby", "Travel"]

    # 定义商品分类和对应商品名称的字典
    categories_product = {
        "Clothing": ["T-shirt", "Jeans", "Skirt", "Other"],
        "Food": ["Snack", "Fruit"],
        "Home Appliance": ["Refrigerator", "Television", "Air conditioner"],
        "Mother and Baby": ["Diapers", "Milk Powder", "Stroller", "Toy"],
        "Travel": ["Tent", "Fishing Rod", "Bike", "Rawalpindi", "Multan"],
    }

    # 循环生成指定数量的订单信息
    for i in range(1, orders_num + 1):
        # 订单ID
        id = i
        # 订单号，由三个随机大写字母和十位随机数字组成
        order_no = "".join(random.choices(string.ascii_uppercase, k=3)) + "".join(
            random.choices(string.digits, k=10)
        )
        # 随机选择一个商品分类
        product_category = random.choice(categories)
        # 根据商品分类随机选择一个商品名称
        product_name = random.choice(categories_product[product_category])
        # 随机生成订单金额（0到10000之间）
        amount = round(random.uniform(0, 10000), 2)
        # 随机选择支付状态
        pay_status = random.choice(["SUCCESS", "FAILD", "CANCEL", "REFUND"])
        # 从已生成的用户列表中随机选择一个用户的 ID 和姓名
        user_id = random.choice(users)[0]
        user_name = random.choice(users)[1]

        # 获取当前时间
        now = datetime.now()
        year = now.year

        # 设置订单创建的随机时间在当前年份范围内
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31)
        random_date = start + timedelta(days=random.randint(0, (end - start).days))
        random_time = datetime.combine(random_date, datetime.min.time()) + timedelta(
            seconds=random.randint(0, 24 * 60 * 60 - 1)
        )

        # 将随机生成的时间格式化为字符串
        random_datetime_str = random_time.strftime("%Y-%m-%d %H:%M:%S")
        create_time = random_datetime_str

        # 构造订单信息元组
        order = (
            id,
            order_no,
            product_category,
            product_name,
            amount,
            pay_status,
            user_id,
            user_name,
            create_time,
        )
        
        # 执行 SQL 插入语句，将订单信息插入到 transaction_order 表中
        cursor.execute(
            "INSERT INTO transaction_order (id, order_no, product_name, product_category, amount, pay_status, user_id, user_name, create_time) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            order,
        )
if __name__ == "__main__":
    # 如果作为主程序执行，则连接到数据库

    connection = pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),  # 获取数据库主机地址，若未设置则默认为本地地址
        port=int(
            os.getenv("DB_PORT", 3306),  # 获取数据库端口号，若未设置则默认为 3306
        ),
        user=os.getenv("DB_USER", "root"),  # 获取数据库用户名，若未设置则默认为 root
        password=os.getenv("DB_PASSWORD", "aa12345678"),  # 获取数据库密码，若未设置则默认为 aa12345678
        database=os.getenv("DB_DATABASE", "dbgpt_test"),  # 获取数据库名称，若未设置则默认为 dbgpt_test
        charset="utf8mb4",  # 设置数据库连接字符集为 UTF-8
        ssl_ca=None,  # 不使用 SSL CA 证书
    )

    # 在数据库连接上建立表格
    build_table(connection)

    # 提交当前事务
    connection.commit()

    # 创建数据库游标
    cursor = connection.cursor()

    # 生成所有用户数据并提交事务
    users = gnerate_all_users(cursor)  # 注意：此处可能是笔误，应为 generate_all_users，而不是 gnerate_all_users
    connection.commit()

    # 生成所有订单数据
    gnerate_all_orders(users, cursor)  # 同上，可能应为 generate_all_orders

    # 提交当前事务
    connection.commit()

    # 执行查询并获取所有用户数据
    cursor.execute("SELECT * FROM user")
    data = cursor.fetchall()
    print(data)

    # 执行查询并获取订单数量
    cursor.execute("SELECT count(*) FROM transaction_order")
    data = cursor.fetchall()
    print("orders:" + str(data))

    # 关闭游标
    cursor.close()

    # 关闭数据库连接
    connection.close()
```