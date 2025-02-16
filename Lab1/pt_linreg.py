import torch
import torch.nn as nn
import torch.optim as optim

# ------TODO
# generating data
N = 50
X = torch.randn(N)  # Input data
Y = 2 * X + 1 + torch.randn(N) * 0.5  # output data with noise


# parameter initialization
a = torch.randn(1, requires_grad=True)  #w
b = torch.randn(1, requires_grad=True)  #b


# optimization procedure: SGD = STOCHASTIC gradient descent
optimizer = optim.SGD([a, b], lr=0.01)

for i in range(10):
    # affine regression model
    Y_ = a*X + b

    diff = (Y-Y_)

    # MSE
    loss = torch.sum(diff**2) / N

    # -------TODO
    # https://chatgpt.com/share/6724c9eb-a2ac-8011-bd6b-f0893b37f8f0
    
    # manual grads
    grad_a = sum(2*(Y-Y_)*-X) / N
    grad_b = sum(2*(Y-Y_)*-1) / N
    
    # --------TODO
    print('Gradient wrt to a: ',grad_a)
    print('Gradient wrt to b: ',grad_b)

    # gradient reset
    optimizer.zero_grad()

    # gradient calculation
    loss.backward()

    # optimization step
    optimizer.step()
    print('Grad a: ',a.grad)  #pytorch grads
    print('Grad b: ',b.grad)

    # ---TODO
    assert torch.allclose(a.grad, grad_a, atol=1e-6), 'Gradient wrt a is not correct'
    assert torch.allclose(b.grad, grad_b, atol=1e-6), 'Gradient wrt b is not correct'


    if i % 1000 == 0:
        print(f'step: {i}, loss:{loss}')

print(f'a: {a}, b: {b}')