import torch
import torch.nn.functional as F  # 主要实现激活函数
import matplotlib.pyplot as plt  # 绘图的工具
from torch.autograd import Variable
 
 
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # 合成数据
y = x.pow(2) + 0.2 * torch.rand(x.size())
x, y = Variable(x), Variable(y)  # 变为Variable 因为网络只支持变量形式
# plt.scatter(x.data.numpy(), y.data.numpy())  # 绘制数据图像
# plt.show()
 
 
class Net(torch.nn.Module):  # 建立网络
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 标准写法
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层 输入 输出个数
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层 输入 输出个数
 
    def forward(self, x):
        x = F.relu(self.hidden(x))  # 隐藏层的激活函数
        x = self.predict(x)  # 线性输出
        return x
 
 
model = Net(n_feature=1, n_hidden=10, n_output=1)  # 初始化网络
loss_func = torch.nn.MSELoss()  # 定义损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 定义优化方法
 
plt.ion()
for t in range(1000):  # 训练1000次
    prediction = model(x)   # input x and predict based on x
    loss = loss_func(prediction, y)  # X 在前(x1, x2,...)，标签在后(y1, y2,...)
    optimizer.zero_grad()  # 梯度设置为0
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # apply gradients
 
    if t % 5 == 0:
        plt.cla()  # 图形显示
        plt.scatter(x.data.numpy(), y.data.numpy())  # 先画数据分布
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.item(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()