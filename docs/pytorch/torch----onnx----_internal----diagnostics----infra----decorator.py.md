# `.\pytorch\torch\onnx\_internal\diagnostics\infra\decorator.py`

```py
# mypy: allow-untyped-defs
# Import necessary modules and functions for type hinting and logging
from __future__ import annotations

import functools  # Import functools module for functional programming utilities
import logging   # Import logging module for logging diagnostic messages
import traceback # Import traceback module for retrieving exception traceback information
from typing import Any, Callable, Dict, Optional, Tuple, Type  # Import necessary types from typing module

# Import internal modules from torch.onnx._internal package
from torch.onnx._internal import _beartype
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, utils

# Define a type alias for a callable that formats messages
MessageFormatterType = Callable[..., str]

# Decorator function to format a message in text format for a given function
@_beartype.beartype
def format_message_in_text(fn: Callable, *args: Any, **kwargs: Any) -> str:
    return f"{formatter.display_name(fn)}. "

# Decorator function to format an exception in markdown format
@_beartype.beartype
def format_exception_in_markdown(exception: Exception) -> str:
    msg_list = ["### Exception log", "```"]
    msg_list.extend(
        traceback.format_exception(type(exception), exception, exception.__traceback__)
    )
    msg_list.append("```py")
    return "\n".join(msg_list)

# Decorator function to format a function signature in markdown format
@_beartype.beartype
def format_function_signature_in_markdown(
    fn: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    format_argument: Callable[[Any], str] = formatter.format_argument,
) -> str:
    msg_list = [f"### Function Signature {formatter.display_name(fn)}"]

    # Obtain the state of the function using utility function
    state = utils.function_state(fn, args, kwargs)

    # Append formatted function state to the message list
    for k, v in state.items():
        msg_list.append(f"- {k}: {format_argument(v)}")

    return "\n".join(msg_list)

# Decorator function to format return values in markdown format
@_beartype.beartype
def format_return_values_in_markdown(
    return_values: Any,
    format_argument: Callable[[Any], str] = formatter.format_argument,
) -> str:
    return f"{format_argument(return_values)}"

# Type alias for a callable modifier function that takes diagnostic information
# and modifies it based on specific rules.
ModifierCallableType = Callable[
    [infra.Diagnostic, Callable, Tuple[Any, ...], Dict[str, Any], Any], None
]

# Function to diagnose a function call using a specified rule and formatting options
@_beartype.beartype
def diagnose_call(
    rule: infra.Rule,
    *,
    level: infra.Level = infra.Level.NONE,
    diagnostic_type: Type[infra.Diagnostic] = infra.Diagnostic,
    format_argument: Callable[[Any], str] = formatter.format_argument,
    diagnostic_message_formatter: MessageFormatterType = format_message_in_text,
) -> Callable:
    return decorator

# TODO(bowbao): decorator to report only when failed.
```