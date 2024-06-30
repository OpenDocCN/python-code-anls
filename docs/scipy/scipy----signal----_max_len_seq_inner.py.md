# `D:\src\scipysrc\scipy\scipy\signal\_max_len_seq_inner.py`

```
# Author: Eric Larson
# 2014

import numpy as np

#pythran export _max_len_seq_inner(int32[], int8[], int, int, int8[])
#pythran export _max_len_seq_inner(int64[], int8[], int, int, int8[])

# Fast inner loop of max_len_seq.
def _max_len_seq_inner(taps, state, nbits, length, seq):
    # Here we compute MLS (Maximum Length Sequence) using a shift register
    # indexed using a ring buffer technique, which is faster than using np.roll
    # to shift the array.
    
    n_taps = taps.shape[0]  # Number of taps in the sequence
    idx = 0  # Initialize the index for the state buffer
    
    # Loop over the specified length to generate the sequence
    for i in range(length):
        feedback = state[idx]  # Get the current state value
        seq[i] = feedback  # Store the current state value in the output sequence
        
        # Compute the feedback using XOR with tapped states
        for ti in range(n_taps):
            feedback ^= state[(taps[ti] + idx) % nbits]
        
        state[idx] = feedback  # Update the state with the computed feedback
        idx = (idx + 1) % nbits  # Move to the next index in the ring buffer
        
    # Adjust the state array so that the next run starts with idx == 0
    return np.roll(state, -idx, axis=0)
```