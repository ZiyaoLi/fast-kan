import torch
from fastkan import AttentionWithFastKANTransform

batch_shape = (1,)
num_q = 12
num_kv = 24
q_dim = k_dim = v_dim = 32
head_dim = 8
num_heads = 2
q = torch.randn(*batch_shape, num_q, q_dim)
k = torch.randn(*batch_shape, num_kv, k_dim)
v = torch.randn(*batch_shape, num_kv, v_dim)

fast_kan_att = AttentionWithFastKANTransform(q_dim, k_dim, v_dim, head_dim, num_heads, gating=True)
out = fast_kan_att(q, k, v, bias=None)
assert out.shape == q.shape, out.shape

bias = torch.rand(*batch_shape, num_q, num_kv)
out = fast_kan_att(q, k, v, bias=bias)
assert out.shape == q.shape, out.shape

print("test attention: attention with fast kan transform got correct shapes.")
