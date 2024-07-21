# `.\pytorch\torch\csrc\jit\runtime\symbolic_shape_registry.h`

```py
/*
#pragma once
// This file is temporary until native_functions.yaml and derivatives.yaml are
// merged. Ideally this should all go into native_functions.yaml

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

/*
ADDING A NEW SHAPE GRAPH:
- For one node schema, there is one corresponding registered shape compute
graph. The schema of the graph should be the same except for Tensor arguments.
For every Tensor input in operator schema, there should be a List[int]
corresponding to that Tensor's shape. For example: "aten::linear(Tensor input,
Tensor weight, Tensor? bias=None) -> Tensor" ==> def linear(input: List[int],
weight: List[int], bias: Optional[List[int]])

Additionally, arguments which are unused at the end of the schema may be left
off. This allows sharing a single graph for multiple function schemas, such as
unary operators with different trailing arguments that do not affect the output
shape.

The shape graph should return a new, unaliased List[int] (or tuple of lists for
multiple returns) and should not modify any input lists. This allows the shape
graphs to be composed and executed.

The shape analysis (particularly for non-complete, or symbolic shapes) works by
partially evaluating the JIT IR. It may be possible for a Graph to be registered
that we cannot currently partially evaluate. If this happens, please file an
issue. There are lints registered to avoid particular known patterns (continue
or break or early return in a loop). Those may be improved in the future, please
file an issue if necessary.

To debug (and write initially) the recommended flow is to define these functions
in python and iterate there. Functions should be added to
torch/jit/_shape_functions.

To test operators, the preferred flow is through OpInfos, with
`assert_jit_shape_analysis=True`. If this is not feasible, you can look at tests
in `test_symbolic_shape_analysis.py` such as `test_adaptive_avg_pool2d`.

Operators which take in a list of tensors, such as concat, are not yet
supported. Concat has been special cased and could be generalized as needed.
Please file an issue.
*/

// Struct to hold lower and upper bound graphs for shape analysis
struct BoundedShapeGraphs {
  std::shared_ptr<Graph> lower_bound;   // Lower bound graph for shape analysis
  std::shared_ptr<Graph> upper_bound;   // Upper bound graph for shape analysis
};

// Registers a shape compute graph for a given function schema
TORCH_API void RegisterShapeComputeGraphForSchema(
    const FunctionSchema& schema,        // Function schema to register
    std::shared_ptr<Graph> g);          // Graph to register

// Retrieves the shape compute graph for a given function schema, if available
TORCH_API std::optional<std::shared_ptr<Graph>> shapeComputeGraphForSchema(
    const FunctionSchema& schema);       // Function schema to query

// Retrieves bounded graphs (lower and upper bounds) for a given function schema, if available
TORCH_API std::optional<BoundedShapeGraphs> boundedGraphsForSchema(
    const FunctionSchema& schema);       // Function schema to query

// Retrieves all registered shape compute schemas
TORCH_API std::vector<const FunctionSchema*> RegisteredShapeComputeSchemas();

// Lints a shape compute graph against a function schema
TORCH_API void LintShapeComputeGraph(
    const FunctionSchema* schema,        // Function schema to lint against
    const std::shared_ptr<Graph>& graph);// Graph to lint

} // namespace torch::jit
*/
```