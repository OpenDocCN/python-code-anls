# `MetaGPT\metagpt\actions\write_docstring.py`

```py

"""Code Docstring Generator.

This script provides a tool to automatically generate docstrings for Python code. It uses the specified style to create
docstrings for the given code and system text.

Usage:
    python3 -m metagpt.actions.write_docstring <filename> [--overwrite] [--style=<docstring_style>]

Arguments:
    filename           The path to the Python file for which you want to generate docstrings.

Options:
    --overwrite        If specified, overwrite the original file with the code containing docstrings.
    --style=<docstring_style>   Specify the style of the generated docstrings.
                                Valid values: 'google', 'numpy', or 'sphinx'.
                                Default: 'google'

Example:
    python3 -m metagpt.actions.write_docstring ./metagpt/startup.py --overwrite False --style=numpy

This script uses the 'fire' library to create a command-line interface. It generates docstrings for the given Python code using
the specified docstring style and adds them to the code.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal, Optional

from metagpt.actions.action import Action
from metagpt.utils.common import OutputParser, aread, awrite
from metagpt.utils.pycst import merge_docstring

PYTHON_DOCSTRING_SYSTEM = """### Requirements
1. Add docstrings to the given code following the {style} style.
2. Replace the function body with an Ellipsis object(...) to reduce output.
3. If the types are already annotated, there is no need to include them in the docstring.
4. Extract only class, function or the docstrings for the module parts from the given Python code, avoiding any other text.

### Input Example

def function_with_pep484_type_annotations(param1: int) -> bool:
    return isinstance(param1, int)

class ExampleError(Exception):
    def __init__(self, msg: str):
        self.msg = msg


### Output Example

{example}

"""

# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

PYTHON_DOCSTRING_EXAMPLE_GOOGLE = '''
...  # Example docstring in Google style
'''

PYTHON_DOCSTRING_EXAMPLE_NUMPY = '''
...  # Example docstring in Numpy style
'''

PYTHON_DOCSTRING_EXAMPLE_SPHINX = '''
...  # Example docstring in Sphinx style
'''

_python_docstring_style = {
    "google": PYTHON_DOCSTRING_EXAMPLE_GOOGLE.strip(),
    "numpy": PYTHON_DOCSTRING_EXAMPLE_NUMPY.strip(),
    "sphinx": PYTHON_DOCSTRING_EXAMPLE_SPHINX.strip(),
}


class WriteDocstring(Action):
    """This class is used to write docstrings for code.

    Attributes:
        desc: A string describing the action.
    """

    desc: str = "Write docstring for code."
    context: Optional[str] = None

    async def run(
        self,
        code: str,
        system_text: str = PYTHON_DOCSTRING_SYSTEM,
        style: Literal["google", "numpy", "sphinx"] = "google",
    ) -> str:
        """Writes docstrings for the given code and system text in the specified style.

        Args:
            code: A string of Python code.
            system_text: A string of system text.
            style: A string specifying the style of the docstring. Can be 'google', 'numpy', or 'sphinx'.

        Returns:
            The Python code with docstrings added.
        """
        system_text = system_text.format(style=style, example=_python_docstring_style[style])
        simplified_code = _simplify_python_code(code)
        documented_code = await self._aask(f"```python\n{simplified_code}\n```", [system_text])
        documented_code = OutputParser.parse_python_code(documented_code)
        return merge_docstring(code, documented_code)

    @staticmethod
    async def write_docstring(
        filename: str | Path, overwrite: bool = False, style: Literal["google", "numpy", "sphinx"] = "google"
    ) -> str:
        data = await aread(str(filename))
        code = await WriteDocstring().run(data, style=style)
        if overwrite:
            await awrite(filename, code)
        return code


def _simplify_python_code(code: str) -> None:
    """Simplifies the given Python code by removing expressions and the last if statement.

    Args:
        code: A string of Python code.

    Returns:
        The simplified Python code.
    """
    code_tree = ast.parse(code)
    code_tree.body = [i for i in code_tree.body if not isinstance(i, ast.Expr)]
    if isinstance(code_tree.body[-1], ast.If):
        code_tree.body.pop()
    return ast.unparse(code_tree)


if __name__ == "__main__":
    import fire

    fire.Fire(WriteDocstring.write_docstring)

```