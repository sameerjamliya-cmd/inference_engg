import torch

d_model = 16
d_ff = 4 * d_model   # standard convention: hidden layer 4x wider than d_model

mlp = torch.nn.Sequential(
    torch.nn.Linear(d_model, d_ff),
    torch.nn.GELU(),
    torch.nn.Linear(d_ff, d_model),
)

# fake a small toy "attention output" -- 2 positions, d_model=16, just like step 2
x = torch.randn(2, d_model)
print(f"input shape:  {x.shape}")

output = mlp(x)
print(f"output shape: {output.shape}   (same as input -- MLP doesn't change shape)")
print()

# prove positions are processed independently: running the MLP on position 0
# ALONE should give the exact same result as position 0's row when run as
# part of the full batch.
out_pos0_alone = mlp(x[0:1])       # shape [1, d_model]
out_pos0_in_batch = output[0:1]    # shape [1, d_model]

print("MLP(x[0]) computed alone:        ", out_pos0_alone)
print("MLP(x)[0] computed as part of x: ", out_pos0_in_batch)
print("identical?", torch.allclose(out_pos0_alone, out_pos0_in_batch))
