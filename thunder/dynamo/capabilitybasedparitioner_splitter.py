# CapabilityBasedParitioner returns a GraphModule where `fused_*` represent the subgraphs
# that should go to `thunder` and the forward of this graph module should be passed to `torch.compile` (after removing thunder bits)
# Example -
# class GraphModule(torch.nn.Module):
#     def forward(self, L_x_: "f32[2]"):
#         l_x_ = L_x_

#         # No stacktrace found for following nodes
#         _enter_autocast = torch.amp.autocast_mode._enter_autocast('cpu', None, True, None)

#          # File: /home/kkalambarkar/lightning-thunder/scratchpad/test.py:181 in func, code: return torch.matmul(x, y)
#         fused_0: "bf16[]" = self.fused_0(l_x_);  l_x_ = None

#         # No stacktrace found for following nodes
#         _exit_autocast = torch.amp.autocast_mode._exit_autocast(_enter_autocast);  _enter_autocast = _exit_autocast = None
#         return (fused_0,)

#     class fused_0(torch.nn.Module):
#         def forward(self, l_x_: "f32[2]"):
#              # File: /home/kkalambarkar/lightning-thunder/scratchpad/test.py:177 in func, code: x = x + 2
#             x: "f32[2]" = l_x_ + 2;  l_x_ = None

#              # File: /home/kkalambarkar/lightning-thunder/scratchpad/test.py:179 in func, code: z = torch.ones(3, 3)
#             z: "f32[3, 3]" = torch.ones(3, 3);  z = None

#              # File: /home/kkalambarkar/lightning-thunder/scratchpad/test.py:180 in func, code: y = torch.sin(x)
#             y: "f32[2]" = torch.sin(x)

#              # File: /home/kkalambarkar/lightning-thunder/scratchpad/test.py:181 in func, code: return torch.matmul(x, y)
#             matmul: "bf16[]" = torch.matmul(x, y);  x = y = None
#             return matmul

# def capability_partitioner_splitter(gm, sample_args):
#     gm_copy = copy.deepcopy(gm)
#     op_support = ThunderOperatorSupport(gm_copy)
#     partitioner = CapabilityBasedPartitioner(gm_copy, op_support)
#     fused_partition = partitioner.partition_and_fuse()
#     gm_copy.print_readable()
#     return gm_copy
