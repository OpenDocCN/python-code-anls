# 2.1 Map Operator

The `MapOperator` is most commonly used to apply a function to input data.

There are two ways to use the `MapOperator`:

## Build a `MapOperator` with a map function

```py
from dbgpt.core.awel import DAG, MapOperator

with DAG("awel_hello_world") as dag:
    task = MapOperator(map_function=lambda x: print(f"Hello, {x}!"))
```

## Implement a custom `MapOperator`

```py
from dbgpt.core.awel import DAG, MapOperator

class MyMapOperator(MapOperator[str, None]):
    async def map(self, x: str) -> None:
        print(f"Hello, {x}!")

with DAG("awel_hello_world") as dag:
    task = MyMapOperator()
```

## Examples

### Double the number

Create a new file named `map_operator_double_number.py` in the `awel_tutorial` directory and add the following code:
```py
import asyncio
from dbgpt.core.awel import DAG, MapOperator

class DoubleNumberOperator(MapOperator[int, int]):
    async def map(self, x: int) -> int:
        print(f"Received {x}, returning {x * 2}")
        return x * 2

with DAG("awel_double_number") as dag:
    task = DoubleNumberOperator()  
assert asyncio.run(task.call(2)) == 4
```

And run the following command to execute the code:
```py
poetry run python awel_tutorial/map_operator_double_number.py
```

And you will see "Received 2, returning 4" printed to the console.
```py
Received 2, returning 4
```
