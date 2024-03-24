## Attention for Tensorflow Mesh

A collection of attention related functions, for building and scaling large attention neural networks.

## Install

```bash
$ pip install attention-tensorflow-mesh
```

## Usage

```python
from attention_tensorflow_mesh import transformer_lm

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import mesh_tensorflow as mtf
from mesh_tensorflow import placement_mesh_impl

graph = mtf.Graph()
mesh = mtf.Mesh(graph, "my_mesh")

# setup dimensions

batch 		= mtf.Dimension('batch', 1)
seq_len 	= mtf.Dimension('sequence', 1024)
dim 		= mtf.Dimension('dim', 512)
dim_head 	= mtf.Dimension('dim_head', 12)
dim_features_head = mtf.Dimension('dim_features_head', 64)

# input

input = mtf.ones(mesh, mtf.Shape([batch, seq_len]), dtype=tf.int32)

# transformer

logits = transformer_lm(
	input,
	dim = 512,
	num_tokens = 20000,
	depth = 1,
	max_seq_len = 1024,
	dim_head = 12,
	dim_features_head = 75,
	causal = True
)

# placement

mesh_impl = placement_mesh_impl.PlacementMeshImpl(shape=[], layout={}, devices=[""])
lowering = mtf.Lowering(graph, {mesh: mesh_impl})

# export

logits = lowering.export_to_tf_tensor(logits)
print(logits)
```

More tools to come
