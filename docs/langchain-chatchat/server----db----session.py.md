# `.\Langchain-Chatchat\server\db\session.py`

```py
# 导入必要的模块
from functools import wraps
from contextlib import contextmanager
from server.db.base import SessionLocal
from sqlalchemy.orm import Session

# 定义上下文管理器，用于自动获取数据库会话 Session，避免错误
@contextmanager
def session_scope() -> Session:
    """上下文管理器用于自动获取 Session, 避免错误"""
    # 创建数据库会话 Session
    session = SessionLocal()
    try:
        # 执行 yield 语句之前的代码块
        yield session
        # 提交事务
        session.commit()
    except:
        # 回滚事务
        session.rollback()
        # 抛出异常
        raise
    finally:
        # 关闭数据库会话 Session
        session.close()

# 装饰器函数，用于在函数执行前后自动获取和关闭数据库会话 Session
def with_session(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        # 使用 session_scope 上下文管理器获取数据库会话 Session
        with session_scope() as session:
            try:
                # 执行被装饰的函数，并传入数据库会话 Session
                result = f(session, *args, **kwargs)
                # 提交事务
                session.commit()
                # 返回函数执行结果
                return result
            except:
                # 回滚事务
                session.rollback()
                # 抛出异常
                raise

    return wrapper

# 生成器函数，用于获取数据库会话 Session
def get_db() -> SessionLocal:
    db = SessionLocal()
    try:
        # 返回数据库会话 Session
        yield db
    finally:
        # 关闭数据库会话 Session
        db.close()

# 生成器函数，用于获取数据库会话 Session
def get_db0() -> SessionLocal:
    db = SessionLocal()
    # 返回数据库会话 Session
    return db
```