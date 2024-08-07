# 1.1 Hello World

## Preparation

In this tutorial, we'll use `poetry` to manage our project dependencies. If you don't have `poetry` installed, you can install it by following the instructions [here](https://python-poetry.org/docs/).

## Creating A Project

You'll start by creating a new python project. You can name it whatever you like; for this tutorial, we'll call it `awel-tutorial`.

We suggest making a project directory in your home directory, but you can put it wherever you like.

Open a terminal and run the following commands to make a project directory and an AWEL tutorial directory:

For Linux, macOS, or PowerShell, enter this:
```py
mkdir -p ~/projects
cd ~/projects
```

Then, run the following commands to create a new project and change to the new directory:

```py
poetry new awel-tutorial
cd awel-tutorial
```

The tree of the project should look like this:

```py
awel-tutorial
├── README.md
├── awel_tutorial
│   └── __init__.py
├── pyproject.toml
└── tests
    └── __init__.py
```

## Adding DB-GPT Dependency

```py
poetry add "dbgpt>=0.5.1"
```

## First Hello World

Next, you'll create a simple DAG that prints "Hello, world" to the console.

Now create a new file called `first_hello_world.py` in the `awel_tutorial` directory and add the following code:

```py
from dbgpt.core.awel import DAG, MapOperator

with DAG("awel_hello_world") as dag:
    task = MapOperator(map_function=lambda x: print(f"Hello, {x}!"))
task._blocking_call(call_data="world")
```

Now, the tree of the project should look like this:

```py
awel-tutorial
├── README.md
├── awel_tutorial
│   ├── __init__.py
│   └── first_hello_world.py
├── poetry.lock
├── pyproject.toml
└── tests
    └── __init__.py

```

Then, run the following command to execute the code:

```py
poetry run python awel_tutorial/first_hello_world.py
```

And you will see "Hello, world!" printed to the console.
```py
Hello, world!
```

## Anatomy Of AWEL Code

Let's break down the code you just wrote.

```py
with DAG("awel_hello_world") as dag:
    task = MapOperator(map_function=lambda x: print(f"Hello, {x}!"))
```

This code creates a new DAG(directed acyclic graph) with the name `awel_hello_world`. 
The `MapOperator` is a simple operator that takes a function and calls it with the data 
passed to it. In this case, the function is a lambda that prints "Hello, world!" to the console.


The task is the instance of the `MapOperator` class. And we call the `call` method of 
the task with the `call_data` parameter set to `"world"`.

```py
task._blocking_call(call_data="world")
```
When you call the task, the lambda function is called with the data(`"world"`) you passed to it.

THe `_blocking_call` method is used to call the task in a blocking way. Just for 
testing here, and we will find a better way to call the task in the next section.


## Hello World With `asyncio`

ALL task calls in AWEL are asynchronous. This example shows how to run the task with
asyncio.

Create a new file called `first_hello_world_asyncio.py` in the `awel_tutorial` directory and add the following code:

```py
import asyncio

from dbgpt.core.awel import DAG, MapOperator

with DAG("awel_hello_world") as dag:
    task = MapOperator(map_function=lambda x: print(f"Hello, {x}!"))

asyncio.run(task.call(call_data="world"))
```
And run the following command to execute the code:

```py
poetry run python awel_tutorial/first_hello_world_asyncio.py
```
And you will see "Hello, world!" printed to the console.
```py
Hello, world!
```

## Hello World With Two Tasks

When we call a single node, we can pass data to it. This example shows how to pass data
to tasks with a InputOperator.

Create a new file called `first_hello_world_two_tasks.py` in the `awel_tutorial` 
directory and add the following code:

```py
import asyncio

from dbgpt.core.awel import DAG, MapOperator, InputOperator, SimpleCallDataInputSource

with DAG("awel_hello_world") as dag:
    input_task = InputOperator(
        input_source=SimpleCallDataInputSource()
    )
    task = MapOperator(map_function=lambda x: print(f"Hello, {x}!"))
    input_task >> task
    

asyncio.run(task.call(call_data="world"))
```

And run the following command to execute the code:

```py
poetry run python awel_tutorial/first_hello_world_two_tasks.py
```
And you will see "Hello, world!" printed to the console.
```py
Hello, world!
```

In this case, we have two tasks. The first task is an `InputOperator` that takes data 
from the `SimpleCallDataInputSource`. The second task is a `MapOperator` that takes the 
data from the first task and prints "Hello, world!" to the console.

And we use the `>>` operator to connect the two tasks. This operator is used to define 
the parent-child relationship between tasks, also known as the task dependency.
You can define the task dependency by using the `set_downstream` method as well, flollowing is the example:

```py
input_task.set_downstream(task)
```

The one task DAG above is a special case of the two tasks DAG, where the `InputOperator` is not used.

```py
with DAG("awel_hello_world") as dag:
    task = MapOperator(map_function=lambda x: print(f"Hello, {x}!"))

asyncio.run(task.call(call_data="world"))
```

## DAG Visualization

Install the graphviz package to visualize the DAG graph.

```py
poetry add graphviz
```

Modify the `first_hello_world_two_tasks.py` file to add the following code:

```py
dag.visualize_dag()
```

The full code is like this:

```py
import asyncio

from dbgpt.core.awel import DAG, MapOperator, InputOperator, SimpleCallDataInputSource

with DAG("awel_hello_world") as dag:
    input_task = InputOperator(
        input_source=SimpleCallDataInputSource()
    )
    task = MapOperator(map_function=lambda x: print(f"Hello, {x}!"))
    input_task >> task

dag.visualize_dag()
asyncio.run(task.call(call_data="world"))
```

Run `first_hello_world_two_tasks.py` again:

```py
poetry run python awel_tutorial/first_hello_world_two_tasks.py
```

You will see the following output:

```py
InputOperator(node_id=a307d921-3bd0-423d-80f0-30aa25aaa9fe)
 -> MapOperator(node_id=bdb335f8-179d-4e08-b1ec-3b58a52d1e84)
Hello, world!
```

The graph of the DAG is like this:

<p align="left">
  <img src={'/img/awel/awel_tutorial/first_hello_world_two_tasks.png'} width="720px" />
</p>
