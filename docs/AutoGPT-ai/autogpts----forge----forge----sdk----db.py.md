# `.\AutoGPT\autogpts\forge\forge\sdk\db.py`

```py
"""
This is an example implementation of the Agent Protocol DB for development Purposes
It uses SQLite as the database and file store backend.
IT IS NOT ADVISED TO USE THIS IN PRODUCTION!
"""

# 导入所需的模块
import datetime
import math
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    String,
    create_engine,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, joinedload, relationship, sessionmaker

# 导入自定义的错误和日志模块
from .errors import NotFoundError
from .forge_log import ForgeLogger
from .model import Artifact, Pagination, Status, Step, StepRequestBody, Task

# 创建日志对象
LOG = ForgeLogger(__name__)

# 创建基类
class Base(DeclarativeBase):
    pass

# 定义任务模型
class TaskModel(Base):
    __tablename__ = "tasks"

    # 任务ID
    task_id = Column(String, primary_key=True, index=True)
    # 输入数据
    input = Column(String)
    # 附加输入数据
    additional_input = Column(JSON)
    # 创建时间
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    # 修改时间
    modified_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    # 任务与Artifact模型的关联
    artifacts = relationship("ArtifactModel", back_populates="task")

# 定义步骤模型
class StepModel(Base):
    __tablename__ = "steps"

    # 步骤ID
    step_id = Column(String, primary_key=True, index=True)
    # 任务ID
    task_id = Column(String, ForeignKey("tasks.task_id"))
    # 步骤名称
    name = Column(String)
    # 输入数据
    input = Column(String)
    # 状态
    status = Column(String)
    # 输出数据
    output = Column(String)
    # 是否为最后一个步骤
    is_last = Column(Boolean, default=False)
    # 创建时间
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    # 修改时间
    modified_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    # 附加输入数据
    additional_input = Column(JSON)
    # 附加输出数据
    additional_output = Column(JSON)
    # 步骤与Artifact模型的关联
    artifacts = relationship("ArtifactModel", back_populates="step")

# 定义Artifact模型
class ArtifactModel(Base):
    __tablename__ = "artifacts"

    # Artifact ID
    artifact_id = Column(String, primary_key=True, index=True)
    # 定义任务ID列，外键关联tasks表的task_id列
    task_id = Column(String, ForeignKey("tasks.task_id"))
    # 定义步骤ID列，外键关联steps表的step_id列
    step_id = Column(String, ForeignKey("steps.step_id"))
    # 定义代理创建标志列，默认为False
    agent_created = Column(Boolean, default=False)
    # 定义文件名列
    file_name = Column(String)
    # 定义相对路径列
    relative_path = Column(String)
    # 定义创建时间列，默认为当前时间
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    # 定义修改时间列，默认为当前时间，更新时也为当前时间
    modified_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    # 建立与StepModel表的关联关系，反向关联字段为artifacts
    step = relationship("StepModel", back_populates="artifacts")
    # 建立与TaskModel表的关联关系，反向关联字段为artifacts
    task = relationship("TaskModel", back_populates="artifacts")
# 将 TaskModel 对象转换为 Task 对象
def convert_to_task(task_obj: TaskModel, debug_enabled: bool = False) -> Task:
    # 如果 debug_enabled 为 True，则记录日志
    if debug_enabled:
        LOG.debug(f"Converting TaskModel to Task for task_id: {task_obj.task_id}")
    # 转换任务对象中的所有 artifact 到对应的 Artifact 对象列表
    task_artifacts = [convert_to_artifact(artifact) for artifact in task_obj.artifacts]
    # 返回转换后的 Task 对象
    return Task(
        task_id=task_obj.task_id,
        created_at=task_obj.created_at,
        modified_at=task_obj.modified_at,
        input=task_obj.input,
        additional_input=task_obj.additional_input,
        artifacts=task_artifacts,
    )


# 将 StepModel 对象转换为 Step 对象
def convert_to_step(step_model: StepModel, debug_enabled: bool = False) -> Step:
    # 如果 debug_enabled 为 True，则记录日志
    if debug_enabled:
        LOG.debug(f"Converting StepModel to Step for step_id: {step_model.step_id}")
    # 转换步骤对象中的所有 artifact 到对应的 Artifact 对象列表
    step_artifacts = [
        convert_to_artifact(artifact) for artifact in step_model.artifacts
    ]
    # 根据步骤状态转换为对应的 Status 枚举值
    status = Status.completed if step_model.status == "completed" else Status.created
    # 返回转换后的 Step 对象
    return Step(
        task_id=step_model.task_id,
        step_id=step_model.step_id,
        created_at=step_model.created_at,
        modified_at=step_model.modified_at,
        name=step_model.name,
        input=step_model.input,
        status=status,
        output=step_model.output,
        artifacts=step_artifacts,
        is_last=step_model.is_last == 1,
        additional_input=step_model.additional_input,
        additional_output=step_model.additional_output,
    )


# 将 ArtifactModel 对象转换为 Artifact 对象
def convert_to_artifact(artifact_model: ArtifactModel) -> Artifact:
    # 返回转换后的 Artifact 对象
    return Artifact(
        artifact_id=artifact_model.artifact_id,
        created_at=artifact_model.created_at,
        modified_at=artifact_model.modified_at,
        agent_created=artifact_model.agent_created,
        relative_path=artifact_model.relative_path,
        file_name=artifact_model.file_name,
    )


# 数据库连接字符串，格式为 sqlite:///{database_name}
class AgentDB:
    # 初始化函数，接受数据库连接字符串和是否启用调试模式作为参数
    def __init__(self, database_string, debug_enabled: bool = False) -> None:
        # 调用父类的初始化函数
        super().__init__()
        # 设置是否启用调试模式
        self.debug_enabled = debug_enabled
        # 如果启用调试模式，记录初始化 AgentDB 的信息
        if self.debug_enabled:
            LOG.debug(f"Initializing AgentDB with database_string: {database_string}")
        # 创建数据库引擎
        self.engine = create_engine(database_string)
        # 创建数据库中的所有表
        Base.metadata.create_all(self.engine)
        # 创建数据库会话
        self.Session = sessionmaker(bind=self.engine)

    # 创建任务函数，接受输入和额外输入作为参数
    async def create_task(
        self, input: Optional[str], additional_input: Optional[dict] = {}
    ) -> Task:
        # 如果启用调试模式，记录创建新任务的信息
        if self.debug_enabled:
            LOG.debug("Creating new task")

        try:
            # 使用数据库会话创建新任务
            with self.Session() as session:
                # 创建新任务对象
                new_task = TaskModel(
                    task_id=str(uuid.uuid4()),
                    input=input,
                    additional_input=additional_input if additional_input else {},
                )
                # 将新任务添加到数据库会话
                session.add(new_task)
                # 提交会话中的所有更改
                session.commit()
                # 刷新新任务对象
                session.refresh(new_task)
                # 如果启用调试模式，记录创建新任务的信息
                if self.debug_enabled:
                    LOG.debug(f"Created new task with task_id: {new_task.task_id}")
                # 将数据库中的任务对象转换为任务对象并返回
                return convert_to_task(new_task, self.debug_enabled)
        # 捕获 SQLAlchemyError 异常
        except SQLAlchemyError as e:
            LOG.error(f"SQLAlchemy error while creating task: {e}")
            raise
        # 捕获 NotFoundError 异常
        except NotFoundError as e:
            raise
        # 捕获其他异常
        except Exception as e:
            LOG.error(f"Unexpected error while creating task: {e}")
            raise

    # 创建步骤函数，接受任务 ID、步骤输入、是否为最后一步和额外输入作为参数
    async def create_step(
        self,
        task_id: str,
        input: StepRequestBody,
        is_last: bool = False,
        additional_input: Optional[Dict[str, Any]] = {},
    # 定义一个方法，用于创建一个新的步骤对象
    ) -> Step:
        # 如果开启了调试模式，则记录创建新步骤的日志信息
        if self.debug_enabled:
            LOG.debug(f"Creating new step for task_id: {task_id}")
        try:
            # 使用上下文管理器创建数据库会话
            with self.Session() as session:
                # 创建一个新的步骤对象
                new_step = StepModel(
                    task_id=task_id,
                    step_id=str(uuid.uuid4()),
                    name=input.input,
                    input=input.input,
                    status="created",
                    is_last=is_last,
                    additional_input=additional_input,
                )
                # 将新步骤对象添加到数据库会话中
                session.add(new_step)
                # 提交会话中的所有更改
                session.commit()
                # 刷新新步骤对象，以确保获取最新的数据库状态
                session.refresh(new_step)
                # 如果开启了调试模式，则记录创建新步骤的日志信息
                if self.debug_enabled:
                    LOG.debug(f"Created new step with step_id: {new_step.step_id}")
                # 将数据库中的步骤对象转换为应用程序中的步骤对象，并返回
                return convert_to_step(new_step, self.debug_enabled)
        # 捕获 SQLAlchemyError 异常
        except SQLAlchemyError as e:
            LOG.error(f"SQLAlchemy error while creating step: {e}")
            raise
        # 捕获 NotFoundError 异常
        except NotFoundError as e:
            raise
        # 捕获所有其他异常
        except Exception as e:
            LOG.error(f"Unexpected error while creating step: {e}")
            raise

    # 异步方法，用于创建一个新的文件对象
    async def create_artifact(
        # 任务 ID
        self,
        task_id: str,
        # 文件名
        file_name: str,
        # 相对路径
        relative_path: str,
        # 是否由代理创建
        agent_created: bool = False,
        # 步骤 ID，可选参数
        step_id: str | None = None,
    # 定义一个方法，用于创建新的 Artifact 对象
    ) -> Artifact:
        # 如果开启了调试模式，记录创建新 artifact 的日志信息
        if self.debug_enabled:
            LOG.debug(f"Creating new artifact for task_id: {task_id}")
        try:
            # 使用 Session 上下文管理器创建数据库会话
            with self.Session() as session:
                # 查询数据库中是否已存在相同 task_id、file_name 和 relative_path 的 ArtifactModel 对象
                if (
                    existing_artifact := session.query(ArtifactModel)
                    .filter_by(
                        task_id=task_id,
                        file_name=file_name,
                        relative_path=relative_path,
                    )
                    .first()
                ):
                    # 如果存在相同的 ArtifactModel 对象，关闭数据库会话并返回该对象的转换后的 Artifact 对象
                    session.close()
                    if self.debug_enabled:
                        LOG.debug(
                            f"Artifact already exists with relative_path: {relative_path}"
                        )
                    return convert_to_artifact(existing_artifact)

                # 如果不存在相同的 ArtifactModel 对象，创建一个新的 ArtifactModel 对象
                new_artifact = ArtifactModel(
                    artifact_id=str(uuid.uuid4()),
                    task_id=task_id,
                    step_id=step_id,
                    agent_created=agent_created,
                    file_name=file_name,
                    relative_path=relative_path,
                )
                # 将新的 ArtifactModel 对象添加到数据库会话中
                session.add(new_artifact)
                # 提交会话中的所有更改
                session.commit()
                # 刷新新创建的 ArtifactModel 对象
                session.refresh(new_artifact)
                if self.debug_enabled:
                    LOG.debug(
                        f"Created new artifact with artifact_id: {new_artifact.artifact_id}"
                    )
                # 返回新创建的 Artifact 对象
                return convert_to_artifact(new_artifact)
        except SQLAlchemyError as e:
            # 捕获 SQLAlchemyError 异常并记录错误日志
            LOG.error(f"SQLAlchemy error while creating step: {e}")
            raise
        except NotFoundError as e:
            # 捕获 NotFoundError 异常并重新抛出
            raise
        except Exception as e:
            # 捕获其他异常并记录错误日志
            LOG.error(f"Unexpected error while creating step: {e}")
            raise
    # 异步函数，通过任务ID获取任务对象
    async def get_task(self, task_id: str) -> Task:
        """Get a task by its id"""
        # 如果启用了调试模式，记录获取任务的日志信息
        if self.debug_enabled:
            LOG.debug(f"Getting task with task_id: {task_id}")
        try:
            # 使用会话对象查询任务
            with self.Session() as session:
                # 查询任务对象，并加载关联的附件信息
                if task_obj := (
                    session.query(TaskModel)
                    .options(joinedload(TaskModel.artifacts))
                    .filter_by(task_id=task_id)
                    .first()
                ):
                    # 将数据库查询结果转换为任务对象
                    return convert_to_task(task_obj, self.debug_enabled)
                else:
                    # 如果未找到任务，记录错误信息并抛出异常
                    LOG.error(f"Task not found with task_id: {task_id}")
                    raise NotFoundError("Task not found")
        except SQLAlchemyError as e:
            # 处理SQLAlchemy错误，记录错误信息并重新抛出异常
            LOG.error(f"SQLAlchemy error while getting task: {e}")
            raise
        except NotFoundError as e:
            # 处理任务未找到异常，直接重新抛出异常
            raise
        except Exception as e:
            # 处理其他异常，记录错误信息并重新抛出异常
            LOG.error(f"Unexpected error while getting task: {e}")
            raise
    # 异步方法，用于获取特定任务和步骤的信息
    async def get_step(self, task_id: str, step_id: str) -> Step:
        # 如果启用了调试模式，则记录获取步骤的日志信息
        if self.debug_enabled:
            LOG.debug(f"Getting step with task_id: {task_id} and step_id: {step_id}")
        try:
            # 使用会话对象查询数据库中的步骤信息
            with self.Session() as session:
                # 查询数据库中包含步骤信息的对象，并加载关联的 artifacts
                if step := (
                    session.query(StepModel)
                    .options(joinedload(StepModel.artifacts))
                    .filter(StepModel.step_id == step_id)
                    .first()
                ):
                    # 将数据库查询结果转换为 Step 对象，并根据调试模式返回结果
                    return convert_to_step(step, self.debug_enabled)

                else:
                    # 如果未找到步骤信息，则记录错误日志并抛出 NotFoundError 异常
                    LOG.error(
                        f"Step not found with task_id: {task_id} and step_id: {step_id}"
                    )
                    raise NotFoundError("Step not found")
        except SQLAlchemyError as e:
            # 捕获 SQLAlchemyError 异常并记录错误日志，然后重新抛出异常
            LOG.error(f"SQLAlchemy error while getting step: {e}")
            raise
        except NotFoundError as e:
            # 捕获 NotFoundError 异常并重新抛出异常
            raise
        except Exception as e:
            # 捕获其他异常并记录错误日志，然后重新抛出异常
            LOG.error(f"Unexpected error while getting step: {e}")
            raise
    # 异步方法，用于获取特定 artifact_id 的 Artifact 对象
    async def get_artifact(self, artifact_id: str) -> Artifact:
        # 如果开启了调试模式，则记录获取 artifact_id 的日志
        if self.debug_enabled:
            LOG.debug(f"Getting artifact with and artifact_id: {artifact_id}")
        try:
            # 使用 Session 上下文管理器创建 session 对象
            with self.Session() as session:
                # 查询数据库中是否存在指定 artifact_id 的 ArtifactModel 对象
                if (
                    artifact_model := session.query(ArtifactModel)
                    .filter_by(artifact_id=artifact_id)
                    .first()
                ):
                    # 如果存在，则将 ArtifactModel 转换为 Artifact 对象并返回
                    return convert_to_artifact(artifact_model)
                else:
                    # 如果不存在，则记录错误日志并抛出 NotFoundError 异常
                    LOG.error(f"Artifact not found with and artifact_id: {artifact_id}")
                    raise NotFoundError("Artifact not found")
        except SQLAlchemyError as e:
            # 捕获 SQLAlchemyError 异常，记录错误日志并重新抛出异常
            LOG.error(f"SQLAlchemy error while getting artifact: {e}")
            raise
        except NotFoundError as e:
            # 捕获 NotFoundError 异常，直接重新抛出异常
            raise
        except Exception as e:
            # 捕获其他异常，记录错误日志并重新抛出异常
            LOG.error(f"Unexpected error while getting artifact: {e}")
            raise

    # 异步方法，用于更新特定 task_id 和 step_id 的步骤信息
    async def update_step(
        self,
        task_id: str,
        step_id: str,
        status: Optional[str] = None,
        output: Optional[str] = None,
        additional_input: Optional[Dict[str, Any]] = None,
        additional_output: Optional[Dict[str, Any]] = None,
    # 更新步骤信息，根据任务ID和步骤ID
    ) -> Step:
        # 如果启用了调试模式，则记录更新步骤的任务ID和步骤ID
        if self.debug_enabled:
            LOG.debug(f"Updating step with task_id: {task_id} and step_id: {step_id}")
        try:
            # 使用会话对象查询数据库中符合条件的步骤
            with self.Session() as session:
                if (
                    # 查询数据库中符合任务ID和步骤ID的步骤对象
                    step := session.query(StepModel)
                    .filter_by(task_id=task_id, step_id=step_id)
                    .first()
                ):
                    # 如果传入了状态信息，则更新步骤的状态
                    if status is not None:
                        step.status = status
                    # 如果传入了额外输入信息，则更新步骤的额外输入
                    if additional_input is not None:
                        step.additional_input = additional_input
                    # 如果传入了输出信息，则更新步骤的输出
                    if output is not None:
                        step.output = output
                    # 如果传入了额外输出信息，则更新步骤的额外输出
                    if additional_output is not None:
                        step.additional_output = additional_output
                    # 提交事务
                    session.commit()
                    # 返回更新后的步骤对象
                    return await self.get_step(task_id, step_id)
                else:
                    # 如果未找到符合条件的步骤，则记录错误信息并抛出异常
                    LOG.error(
                        f"Step not found for update with task_id: {task_id} and step_id: {step_id}"
                    )
                    raise NotFoundError("Step not found")
        except SQLAlchemyError as e:
            # 捕获SQLAlchemy错误并记录错误信息
            LOG.error(f"SQLAlchemy error while getting step: {e}")
            raise
        except NotFoundError as e:
            # 捕获自定义的未找到错误并抛出
            raise
        except Exception as e:
            # 捕获其他异常并记录错误信息
            LOG.error(f"Unexpected error while getting step: {e}")
            raise

    # 更新工件信息，根据工件ID
    async def update_artifact(
        self,
        artifact_id: str,
        *,
        file_name: str = "",
        relative_path: str = "",
        agent_created: Optional[Literal[True]] = None,
    # 更新具有给定 artifact_id 的 artifact，记录调试信息
    ) -> Artifact:
        LOG.debug(f"Updating artifact with artifact_id: {artifact_id}")
        # 创建数据库会话
        with self.Session() as session:
            # 查询具有给定 artifact_id 的 ArtifactModel 对象
            if (
                artifact := session.query(ArtifactModel)
                .filter_by(artifact_id=artifact_id)
                .first()
            ):
                # 如果存在该 artifact，则更新其属性
                if file_name:
                    artifact.file_name = file_name
                if relative_path:
                    artifact.relative_path = relative_path
                if agent_created:
                    artifact.agent_created = agent_created
                # 提交更改到数据库
                session.commit()
                # 返回更新后的 artifact
                return await self.get_artifact(artifact_id)
            else:
                # 如果未找到该 artifact，则记录错误信息并引发异常
                LOG.error(f"Artifact not found with artifact_id: {artifact_id}")
                raise NotFoundError("Artifact not found")

    # 列出任务列表，可指定页码和每页数量
    async def list_tasks(
        self, page: int = 1, per_page: int = 10
    # 定义一个方法，用于列出任务和分页信息
    ) -> Tuple[List[Task], Pagination]:
        # 如果启用了调试模式，记录日志
        if self.debug_enabled:
            LOG.debug("Listing tasks")
        try:
            # 使用会话对象查询数据库
            with self.Session() as session:
                # 查询任务表，根据分页参数获取指定范围内的任务
                tasks = (
                    session.query(TaskModel)
                    .offset((page - 1) * per_page)
                    .limit(per_page)
                    .all()
                )
                # 查询任务总数
                total = session.query(TaskModel).count()
                # 计算总页数
                pages = math.ceil(total / per_page)
                # 创建分页对象
                pagination = Pagination(
                    total_items=total,
                    total_pages=pages,
                    current_page=page,
                    page_size=per_page,
                )
                # 将查询到的任务转换为 Task 对象，并返回任务列表和分页对象
                return [
                    convert_to_task(task, self.debug_enabled) for task in tasks
                ], pagination
        except SQLAlchemyError as e:
            # 捕获 SQLAlchemy 错误并记录日志
            LOG.error(f"SQLAlchemy error while listing tasks: {e}")
            raise
        except NotFoundError as e:
            # 捕获 NotFoundError 异常并抛出
            raise
        except Exception as e:
            # 捕获其他异常并记录日志
            LOG.error(f"Unexpected error while listing tasks: {e}")
            raise

    # 定义一个异步方法，用于列出步骤
    async def list_steps(
        # 方法参数包括任务 ID、页码和每页数量，默认值为 1 和 10
        self, task_id: str, page: int = 1, per_page: int = 10
    # 定义一个方法，用于列出给定任务ID的步骤和分页信息
    ) -> Tuple[List[Step], Pagination]:
        # 如果启用了调试模式，则记录调试信息
        if self.debug_enabled:
            LOG.debug(f"Listing steps for task_id: {task_id}")
        try:
            # 使用会话对象查询数据库，获取指定任务ID的步骤信息
            with self.Session() as session:
                steps = (
                    session.query(StepModel)
                    .filter_by(task_id=task_id)
                    .offset((page - 1) * per_page)
                    .limit(per_page)
                    .all()
                )
                # 查询总步骤数
                total = session.query(StepModel).filter_by(task_id=task_id).count()
                # 计算总页数
                pages = math.ceil(total / per_page)
                # 创建分页对象
                pagination = Pagination(
                    total_items=total,
                    total_pages=pages,
                    current_page=page,
                    page_size=per_page,
                )
                # 将步骤信息转换为Step对象，并返回步骤列表和分页信息
                return [
                    convert_to_step(step, self.debug_enabled) for step in steps
                ], pagination
        except SQLAlchemyError as e:
            # 如果发生SQLAlchemy错误，则记录错误信息并抛出异常
            LOG.error(f"SQLAlchemy error while listing steps: {e}")
            raise
        except NotFoundError as e:
            # 如果发生NotFoundError异常，则直接抛出异常
            raise
        except Exception as e:
            # 如果发生其他异常，则记录错误信息并抛出异常
            LOG.error(f"Unexpected error while listing steps: {e}")
            raise

    # 定义一个异步方法，用于列出给定任务ID的工件信息
    async def list_artifacts(
        self, task_id: str, page: int = 1, per_page: int = 10
    # 定义一个方法，用于列出任务ID对应的所有工件和分页信息
    ) -> Tuple[List[Artifact], Pagination]:
        # 如果启用了调试模式，则记录调试信息
        if self.debug_enabled:
            LOG.debug(f"Listing artifacts for task_id: {task_id}")
        # 尝试执行以下代码块，捕获可能出现的异常
        try:
            # 使用数据库会话查询指定任务ID的工件信息
            with self.Session() as session:
                artifacts = (
                    session.query(ArtifactModel)
                    .filter_by(task_id=task_id)
                    .offset((page - 1) * per_page)
                    .limit(per_page)
                    .all()
                )
                # 查询指定任务ID的工件总数
                total = session.query(ArtifactModel).filter_by(task_id=task_id).count()
                # 计算总页数
                pages = math.ceil(total / per_page)
                # 创建分页对象
                pagination = Pagination(
                    total_items=total,
                    total_pages=pages,
                    current_page=page,
                    page_size=per_page,
                )
                # 将查询到的工件转换为自定义对象，并返回工件列表和分页信息
                return [
                    convert_to_artifact(artifact) for artifact in artifacts
                ], pagination
        # 捕获SQLAlchemyError异常，记录错误信息并重新抛出异常
        except SQLAlchemyError as e:
            LOG.error(f"SQLAlchemy error while listing artifacts: {e}")
            raise
        # 捕获NotFoundError异常，直接重新抛出异常
        except NotFoundError as e:
            raise
        # 捕获其他异常，记录错误信息并重新抛出异常
        except Exception as e:
            LOG.error(f"Unexpected error while listing artifacts: {e}")
            raise
```