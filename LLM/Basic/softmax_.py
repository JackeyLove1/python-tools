import torch
from torch import Tensor, Generator, strided, memory_format, contiguous_format, strided
'''
对每个项求幂（使用exp）；
对每一行求和（小批量中每个样本是一行），得到每个样本的规范化常数；
将每一行除以其规范化常数，确保结果的和为1。
'''
def my_softmax(input_tensor:Tensor, dim:int):
    # 计算指数函数的幂次
    exp_values = torch.exp(input_tensor)
    # 计算指数函数的幂次的总和
    sum_exp_values = torch.sum(exp_values, dim=dim, keepdim=True)
    # 计算每个元素的softmax值
    softmax_output = exp_values / sum_exp_values
    return softmax_output

x = torch.rand((2,3,4))
print(torch.softmax(x, dim=1))
print(my_softmax(x, 1))