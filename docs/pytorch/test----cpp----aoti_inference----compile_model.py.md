# `.\pytorch\test\cpp\aoti_inference\compile_model.py`

```
import torch  # 导入 PyTorch 库

from torch.export import Dim  # 从 torch.export 模块导入 Dim 类


# custom op that loads the aot-compiled model
# 定义用于加载 AOT 编译模型的自定义操作库路径
AOTI_CUSTOM_OP_LIB = "libaoti_custom_class.so"
torch.classes.load_library(AOTI_CUSTOM_OP_LIB)  # 加载自定义操作库


class TensorSerializer(torch.nn.Module):
    """
    Serialize a dictionary of data into a torch Module.
    """

    def __init__(self, data):
        super().__init__()
        # Iterate through data dictionary and set each key-value pair as an attribute
        for key in data:
            setattr(self, key, data[key])


class SimpleModule(torch.nn.Module):
    """
    A simple neural network module with a linear layer and ReLU activation.
    """

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 6)  # Linear layer with input size 4 and output size 6
        self.relu = torch.nn.ReLU()  # ReLU activation function

    def forward(self, x):
        a = self.fc(x)  # Forward pass through the linear layer
        b = self.relu(a)  # Apply ReLU activation
        return b  # Return the output


class MyAOTIModule(torch.nn.Module):
    """
    Wrapper module that uses MyAOTIClass for AOT-compiled operations.
    """

    def __init__(self, lib_path, device):
        super().__init__()
        # Instantiate MyAOTIClass from custom library path and device
        self.aoti_custom_op = torch.classes.aoti.MyAOTIClass(
            lib_path,
            device,
        )

    def forward(self, *x):
        # Forward pass through MyAOTIClass
        outputs = self.aoti_custom_op.forward(x)
        return tuple(outputs)  # Return outputs as a tuple


def make_script_module(lib_path, device, *inputs):
    """
    Create a scripted module for AOT-compiled operations.
    """
    m = MyAOTIModule(lib_path, device)  # Instantiate MyAOTIModule
    # Perform a sanity check by running the module with inputs
    m(*inputs)
    # Trace the module to create a scripted version
    return torch.jit.trace(m, inputs)


def compile_model(device, data):
    """
    Compile a SimpleModule for both CPU and CUDA devices and save the results.
    """
    module = SimpleModule().to(device)  # Instantiate SimpleModule and move to device
    x = torch.randn((4, 4), device=device)  # Generate random input tensor on specified device
    inputs = (x,)  # Create a tuple of inputs
    # Define dynamic shapes for AOT compilation
    batch_dim = Dim("batch", min=1, max=1024)
    dynamic_shapes = {
        "x": {0: batch_dim},
    }
    with torch.no_grad():
        # AOT-compile the module into a shared object file (.so) at lib_path
        lib_path = torch._export.aot_compile(
            module, inputs, dynamic_shapes=dynamic_shapes
        )
    # Create a scripted module from the compiled library path and inputs
    script_module = make_script_module(lib_path, device, *inputs)
    # Save the scripted module as a .pt file
    aoti_script_model = f"script_model_{device}.pt"
    script_module.save(aoti_script_model)

    # Save sample inputs and reference output for later use
    with torch.no_grad():
        ref_output = module(*inputs)  # Compute reference output
    data.update(
        {
            f"inputs_{device}": list(inputs),  # Store inputs in data dictionary
            f"outputs_{device}": [ref_output],  # Store reference output in data dictionary
        }
    )


def main():
    """
    Main function to compile and save SimpleModule for both CPU and CUDA devices,
    and serialize data into a TensorSerializer module.
    """
    data = {}  # Initialize an empty dictionary for data storage
    for device in ["cuda", "cpu"]:  # Iterate over CPU and CUDA devices
        compile_model(device, data)  # Compile SimpleModule for each device and store data
    # Serialize the collected data dictionary into a TensorSerializer module
    torch.jit.script(TensorSerializer(data)).save("script_data.pt")  # Save serialized data


if __name__ == "__main__":
    main()  # Execute the main function if script is run directly
```