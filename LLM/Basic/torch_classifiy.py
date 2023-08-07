import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1) # (100 ,2)
y0 = torch.zeros(100) #(100,)
x1 = torch.normal(-2 * n_data, 1) # (100,2)
y1 = torch.ones(100) #(100,)

x = torch.cat((x0, x1), dim=0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor).unsqueeze(1)
print(x.shape)
print(y.shape)
# plt.scatter(x.data.numpy(), y.data.numpy())

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super().__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.predict = torch.nn.Linear(n_hidden2, n_output)
        nn.init.xavier_uniform_(self.hidden1.weight)
        nn.init.xavier_uniform_(self.hidden2.weight)
        nn.init.xavier_uniform_(self.predict.weight)
    def forward(self, x):
        x = F.relu(self.hidden2(F.relu(self.hidden1(x))))
        x = self.predict(x)
        return x

net = Net(n_feature=2, n_hidden1=10, n_hidden2=20,n_output=2)
optimizer = torch.optim.SGD(net.parameters(),lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

for t in range(100):
    y_prediction = net(x) # (batch_size, n_output)
    loss = loss_func(y_prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 5 == 0:
        plt.cla()
        prediction  = torch.max(y_prediction, dim=1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200.  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)