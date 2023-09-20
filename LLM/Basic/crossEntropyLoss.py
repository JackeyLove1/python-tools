import torch
import torch.nn as nn
import numpy as np

input = torch.tensor([[1., 2., 3., 4.],
                      [10., 12., 14., 16.],
                      [20., 50., 30., 40.]])
'''
(3, 4)
tensor([[ 1.,  2.,  3.,  4.],
        [10., 12., 14., 16.],
        [20., 50., 30., 40.]])
'''
print(f"input:{input}")

y_target = torch.tensor([1, 2, 3])

softmax = nn.Softmax(dim=1)
softmax_input = softmax(input)
print(softmax_input)
'''
tensor([[3.2059e-02, 8.7144e-02, 2.3688e-01, 6.4391e-01],
        [2.1440e-03, 1.5842e-02, 1.1706e-01, 8.6495e-01],
        [9.3572e-14, 9.9995e-01, 2.0611e-09, 4.5398e-05]])
'''
log_output = torch.log(softmax_input)
print(log_output)  # (3, 4)
'''
tensor([[-3.4402e+00, -2.4402e+00, -1.4402e+00, -4.4019e-01],
        [-6.1451e+00, -4.1451e+00, -2.1451e+00, -1.4508e-01],
        [-3.0000e+01, -4.5420e-05, -2.0000e+01, -1.0000e+01]])
target: torch.tensor([1, 2, 3])
nllloss = [-log_output[0, 1] , -log_output[1, 2] , -log_output[2, 3]]
'''

nllloss_func = nn.NLLLoss(reduction='none')
nllloss_output = nllloss_func(log_output, y_target)
print(f'nllloss_output:{nllloss_output}')
print("output:", [log_output[0, 1] , log_output[1, 2] , log_output[2, 3]])
'''
tensor([ 2.4402,  2.1451, 10.0000])
'''
