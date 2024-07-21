# `.\pytorch\docs\source\scripts\onnx\build_onnx_torchscript_supported_aten_op_csv_table.py`

```py
"""
This script generates a CSV table with all ATen operators
supported by `torch.onnx.export`. The generated table is included by
docs/source/onnx_supported_aten_list.rst.
"""

# Importing necessary libraries
import os

# Importing _onnx_supported_ops from torch.onnx module
from torch.onnx import _onnx_supported_ops

# Constants
BUILD_DIR = "build/onnx"
SUPPORTED_OPS_CSV_FILE = "auto_gen_supported_op_list.csv"
UNSUPPORTED_OPS_CSV_FILE = "auto_gen_unsupported_op_list.csv"

# Function to define sorting key for operator names
def _sort_key(namespaced_opname):
    return tuple(reversed(namespaced_opname.split("::")))

# Function to retrieve lists of supported and unsupported operators
def _get_op_lists():
    # Fetching all forward schemas of supported operators
    all_schemas = _onnx_supported_ops.all_forward_schemas()
    # Fetching all symbolic schemas of supported operators
    symbolic_schemas = _onnx_supported_ops.all_symbolics_schemas()
    
    # Initializing result sets for supported and unsupported operators
    supported_result = set()
    not_supported_result = set()
    
    # Iterating through each operator name in all_schemas
    for opname in all_schemas:
        # Handling underscore suffix if present
        if opname.endswith("_"):
            opname = opname[:-1]
        
        # Checking if the operator is in symbolic_schemas (supported)
        if opname in symbolic_schemas:
            # Extracting opsets information
            opsets = symbolic_schemas[opname].opsets
            # Adding supported operator to the set
            supported_result.add(
                (
                    opname,
                    f"Since opset {opsets[0]}",
                )
            )
        else:
            # Adding unsupported operator to the set
            not_supported_result.add(
                (
                    opname,
                    "Not yet supported",
                )
            )
    
    # Sorting and returning both sets of operators
    return (
        sorted(supported_result, key=lambda x: _sort_key(x[0])),
        sorted(not_supported_result),
    )

# Main function to execute the script
def main():
    # Creating directory if not exists
    os.makedirs(BUILD_DIR, exist_ok=True)

    # Retrieving supported and unsupported operator lists
    supported, unsupported = _get_op_lists()

    # Writing supported operators to CSV file
    with open(os.path.join(BUILD_DIR, SUPPORTED_OPS_CSV_FILE), "w") as f:
        f.write("Operator,opset_version(s)\n")
        for name, opset_version in supported:
            f.write(f'"``{name}``","{opset_version}"\n')

    # Writing unsupported operators to CSV file
    with open(os.path.join(BUILD_DIR, UNSUPPORTED_OPS_CSV_FILE), "w") as f:
        f.write("Operator,opset_version(s)\n")
        for name, opset_version in unsupported:
            f.write(f'"``{name}``","{opset_version}"\n')

# Executing main function if this script is run directly
if __name__ == "__main__":
    main()
```