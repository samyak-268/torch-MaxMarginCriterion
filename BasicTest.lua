require 'nn'
require 'torch'
require 'MaxMarginCriterion'

input = torch.Tensor{1.02, 1.05, 1.15, 0.87}
labels = torch.Tensor{-1, -1, 1, 1}

loss = nn.MaxMarginCriterion(1, 0.1)
err = loss:forward(input, labels)
gradErr = loss:backward(input, labels)
print (err)
print (gradErr)
