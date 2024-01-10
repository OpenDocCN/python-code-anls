# `MetaGPT\metagpt\schema.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/8 22:12
@Author  : alexanderwu
@File    : schema.py
@Modified By: mashenquan, 2023-10-31. According to Chapter 2.2.1 of RFC 116:
        Replanned the distribution of responsibilities and functional positioning of `Message` class attributes.
@Modified By: mashenquan, 2023/11/22.
        1. Add `Document` and `Documents` for `FileRepository` in Section 2.2.3.4 of RFC 135.
        2. Encapsulate the common key-values set to pydantic structures to standardize and unify parameter passing
        between actions.
        3. Add `id` to `Message` according to Section 2.2.3.1.1 of RFC 135.
"""

from __future__ import annotations  # Importing annotations from the future

import asyncio  # Asynchronous I/O
import json  # JSON encoding and decoding
import os.path  # Common pathname manipulations
import uuid  # UUID objects
from abc import ABC  # Abstract Base Classes
from asyncio import Queue, QueueEmpty, wait_for  # Asynchronous I/O
from json import JSONDecodeError  # JSON decoding error
from pathlib import Path  # Object-oriented filesystem paths
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union  # Type hints

from pydantic import (  # Data validation and settings management using python type annotations
    BaseModel,  # Base model for pydantic
    ConfigDict,  # Configuration dictionary
    Field,  # Field for pydantic
    PrivateAttr,  # Private attribute for pydantic
    field_serializer,  # Field serializer for pydantic
    field_validator,  # Field validator for pydantic
)
from pydantic_core import core_schema  # Core schema for pydantic

from metagpt.config import CONFIG  # Configuration for metagpt
from metagpt.const import (  # Constants for metagpt
    MESSAGE_ROUTE_CAUSE_BY,  # Message route cause by
    MESSAGE_ROUTE_FROM,  # Message route from
    MESSAGE_ROUTE_TO,  # Message route to
    MESSAGE_ROUTE_TO_ALL,  # Message route to all
    SYSTEM_DESIGN_FILE_REPO,  # System design file repository
    TASK_FILE_REPO,  # Task file repository
)
from metagpt.logs import logger  # Logger for metagpt
from metagpt.utils.common import any_to_str, any_to_str_set, import_class  # Utility functions for metagpt
from metagpt.utils.exceptions import handle_exception  # Exception handling for metagpt
from metagpt.utils.serialize import (  # Serialization utilities for metagpt
    actionoutout_schema_to_mapping,  # Action output schema to mapping
    actionoutput_mapping_to_str,  # Action output mapping to string
    actionoutput_str_to_mapping,  # Action output string to mapping
)


# Serialization Mixin class for polymorphic serialization/deserialization
class SerializationMixin(BaseModel):
    """
    PolyMorphic subclasses Serialization / Deserialization Mixin
    - First of all, we need to know that pydantic is not designed for polymorphism.
    - If Engineer is subclass of Role, it would be serialized as Role. If we want to serialize it as Engineer, we need
        to add `class name` to Engineer. So we need Engineer inherit SerializationMixin.

    More details:
    - https://docs.pydantic.dev/latest/concepts/serialization/
    - https://github.com/pydantic/pydantic/discussions/7008 discuss about avoid `__get_pydantic_core_schema__`
    """

    __is_polymorphic_base = False  # Flag for polymorphic base
    __subclasses_map__ = {}  # Map for subclasses

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type["SerializationMixin"], handler: Callable[[Any], core_schema.CoreSchema]
    ) -> core_schema.CoreSchema:
        schema = handler(source)
        og_schema_ref = schema["ref"]
        schema["ref"] += ":mixin"

        return core_schema.no_info_before_validator_function(
            cls.__deserialize_with_real_type__,
            schema=schema,
            ref=og_schema_ref,
            serialization=core_schema.wrap_serializer_function_ser_schema(cls.__serialize_add_class_type__),
        )

    @classmethod
    def __serialize_add_class_type__(
        cls,
        value,
        handler: core_schema.SerializerFunctionWrapHandler,
    ) -> Any:
        ret = handler(value)
        if not len(cls.__subclasses__()):
            # only subclass add `__module_class_name`
            ret["__module_class_name"] = f"{cls.__module__}.{cls.__qualname__}"
        return ret

    @classmethod
    def __deserialize_with_real_type__(cls, value: Any):
        if not isinstance(value, dict):
            return value

        if not cls.__is_polymorphic_base or (len(cls.__subclasses__()) and "__module_class_name" not in value):
            # add right condition to init BaseClass like Action()
            return value
        module_class_name = value.get("__module_class_name", None)
        if module_class_name is None:
            raise ValueError("Missing field: __module_class_name")

        class_type = cls.__subclasses_map__.get(module_class_name, None)

        if class_type is None:
            raise TypeError("Trying to instantiate {module_class_name} which not defined yet.")

        return class_type(**value)

    def __init_subclass__(cls, is_polymorphic_base: bool = False, **kwargs):
        cls.__is_polymorphic_base = is_polymorphic_base
        cls.__subclasses_map__[f"{cls.__module__}.{cls.__qualname__}"] = cls
        super().__init_subclass__(**kwargs)


# SimpleMessage class for representing a simple message
class SimpleMessage(BaseModel):
    content: str  # Content of the message
    role: str  # Role of the message


# Document class for representing a document
class Document(BaseModel):
    """
    Represents a document.
    """

    root_path: str = ""  # Root path of the document
    filename: str = ""  # Filename of the document
    content: str = ""  # Content of the document

    def get_meta(self) -> Document:
        """Get metadata of the document.

        :return: A new Document instance with the same root path and filename.
        """

        return Document(root_path=self.root_path, filename=self.filename)

    @property
    def root_relative_path(self):
        """Get relative path from root of git repository.

        :return: relative path from root of git repository.
        """
        return os.path.join(self.root_path, self.filename)

    @property
    def full_path(self):
        if not CONFIG.git_repo:
            return None
        return str(CONFIG.git_repo.workdir / self.root_path / self.filename)

    def __str__(self):
        return self.content

    def __repr__(self):
        return self.content


# Documents class for representing a collection of documents
class Documents(BaseModel):
    """A class representing a collection of documents.

    Attributes:
        docs (Dict[str, Document]): A dictionary mapping document names to Document instances.
    """

    docs: Dict[str, Document] = Field(default_factory=dict)


# UserMessage class for representing a user message
class UserMessage(Message):
    """Facilitate support for OpenAI messages"""

    def __init__(self, content: str):
        super().__init__(content=content, role="user")


# SystemMessage class for representing a system message
class SystemMessage(Message):
    """Facilitate support for OpenAI messages"""

    def __init__(self, content: str):
        super().__init__(content=content, role="system")


# AIMessage class for representing an AI message
class AIMessage(Message):
    """Facilitate support for OpenAI messages"""

    def __init__(self, content: str):
        super().__init__(content=content, role="assistant")


# MessageQueue class for representing a message queue
class MessageQueue(BaseModel):
    """Message queue which supports asynchronous updates."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _queue: Queue = PrivateAttr(default_factory=Queue)

    def pop(self) -> Message | None:
        """Pop one message from the queue."""
        try:
            item = self._queue.get_nowait()
            if item:
                self._queue.task_done()
            return item
        except QueueEmpty:
            return None

    def pop_all(self) -> List[Message]:
        """Pop all messages from the queue."""
        ret = []
        while True:
            msg = self.pop()
            if not msg:
                break
            ret.append(msg)
        return ret

    def push(self, msg: Message):
        """Push a message into the queue."""
        self._queue.put_nowait(msg)

    def empty(self):
        """Return true if the queue is empty."""
        return self._queue.empty()

    async def dump(self) -> str:
        """Convert the `MessageQueue` object to a json string."""
        if self.empty():
            return "[]"

        lst = []
        msgs = []
        try:
            while True:
                item = await wait_for(self._queue.get(), timeout=1.0)
                if item is None:
                    break
                msgs.append(item)
                lst.append(item.dump())
                self._queue.task_done()
        except asyncio.TimeoutError:
            logger.debug("Queue is empty, exiting...")
        finally:
            for m in msgs:
                self._queue.put_nowait(m)
        return json.dumps(lst, ensure_ascii=False)

    @staticmethod
    def load(data) -> "MessageQueue":
        """Convert the json string to the `MessageQueue` object."""
        queue = MessageQueue()
        try:
            lst = json.loads(data)
            for i in lst:
                msg = Message.load(i)
                queue.push(msg)
        except JSONDecodeError as e:
            logger.warning(f"JSON load failed: {data}, error:{e}")

        return queue


# BaseContext class for representing a base context
class BaseContext(BaseModel, ABC):
    @classmethod
    @handle_exception
    def loads(cls: Type[T], val: str) -> Optional[T]:
        i = json.loads(val)
        return cls(**i)


# CodingContext class for representing a coding context
class CodingContext(BaseContext):
    filename: str  # Filename
    design_doc: Optional[Document] = None  # Design document
    task_doc: Optional[Document] = None  # Task document
    code_doc: Optional[Document] = None  # Code document


# TestingContext class for representing a testing context
class TestingContext(BaseContext):
    filename: str  # Filename
    code_doc: Document  # Code document
    test_doc: Optional[Document] = None  # Test document


# RunCodeContext class for representing a run code context
class RunCodeContext(BaseContext):
    mode: str = "script"  # Mode
    code: Optional[str] = None  # Code
    code_filename: str = ""  # Code filename
    test_code: Optional[str] = None  # Test code
    test_filename: str = ""  # Test filename
    command: List[str] = Field(default_factory=list)  # Command
    working_directory: str = ""  # Working directory
    additional_python_paths: List[str] = Field(default_factory=list)  # Additional python paths
    output_filename: Optional[str] = None  # Output filename
    output: Optional[str] = None  # Output


# RunCodeResult class for representing a run code result
class RunCodeResult(BaseContext):
    summary: str  # Summary
    stdout: str  # Standard output
    stderr: str  # Standard error


# CodeSummarizeContext class for representing a code summarize context
class CodeSummarizeContext(BaseModel):
    design_filename: str = ""  # Design filename
    task_filename: str = ""  # Task filename
    codes_filenames: List[str] = Field(default_factory=list)  # Code filenames
    reason: str = ""  # Reason

    @staticmethod
    def loads(filenames: List) -> CodeSummarizeContext:
        ctx = CodeSummarizeContext()
        for filename in filenames:
            if Path(filename).is_relative_to(SYSTEM_DESIGN_FILE_REPO):
                ctx.design_filename = str(filename)
                continue
            if Path(filename).is_relative_to(TASK_FILE_REPO):
                ctx.task_filename = str(filename)
                continue
        return ctx

    def __hash__(self):
        return hash((self.design_filename, self.task_filename))


# BugFixContext class for representing a bug fix context
class BugFixContext(BaseContext):
    filename: str = ""  # Filename


# ClassMeta class for representing a class meta
class ClassMeta(BaseModel):
    name: str = ""  # Name
    abstraction: bool = False  # Abstraction
    static: bool = False  # Static
    visibility: str = ""  # Visibility


# ClassAttribute class for representing a class attribute
class ClassAttribute(ClassMeta):
    value_type: str = ""  # Value type
    default_value: str = ""  # Default value

    def get_mermaid(self, align=1) -> str:
        content = "".join(["\t" for i in range(align)]) + self.visibility
        if self.value_type:
            content += self.value_type + " "
        content += self.name
        if self.default_value:
            content += "="
            if self.value_type not in ["str", "string", "String"]:
                content += self.default_value
            else:
                content += '"' + self.default_value.replace('"', "") + '"'
        if self.abstraction:
            content += "*"
        if self.static:
            content += "$"
        return content


# ClassMethod class for representing a class method
class ClassMethod(ClassMeta):
    args: List[ClassAttribute] = Field(default_factory=list)  # Arguments
    return_type: str = ""  # Return type

    def get_mermaid(self, align=1) -> str:
        content = "".join(["\t" for i in range(align)]) + self.visibility
        content += self.name + "(" + ",".join([v.get_mermaid(align=0) for v in self.args]) + ")"
        if self.return_type:
            content += ":" + self.return_type
        if self.abstraction:
            content += "*"
        if self.static:
            content += "$"
        return content


# ClassView class for representing a class view
class ClassView(ClassMeta):
    attributes: List[ClassAttribute] = Field(default_factory=list)  # Attributes
    methods: List[ClassMethod] = Field(default_factory=list)  # Methods

    def get_mermaid(self, align=1) -> str:
        content = "".join(["\t" for i in range(align)]) + "class " + self.name + "{\n"
        for v in self.attributes:
            content += v.get_mermaid(align=align + 1) + "\n"
        for v in self.methods:
            content += v.get_mermaid(align=align + 1) + "\n"
        content += "".join(["\t" for i in range(align)]) + "}\n"
        return content

```