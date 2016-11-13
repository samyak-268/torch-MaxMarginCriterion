require 'nn'
require 'torch'
require 'optim'
require 'MaxMarginCriterion'

torch.manualSeed(0)

batchSize = 5
embeddingSize = 3
imgSize = 20

function getModel()
    local convNet = nn.Sequential()
    convNet:add(nn.SpatialConvolution(3, 8, 5, 5))
    convNet:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    convNet:add(nn.ReLU())
    convNet:add(nn.SpatialConvolution(8, 8, 5, 5))
    convNet:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    convNet:add(nn.ReLU())
    convNet:add(nn.View(8*2*2))
    convNet:add(nn.Linear(8*2*2, embeddingSize))
    convNet:add(nn.BatchNormalization(embeddingSize))

    local convNetClone = convNet:clone('weight', 'bias', 'gradWeight', 'gradBias')

    local siamese = nn.ParallelTable()
    siamese:add(convNet)
    siamese:add(convNetClone)

    local model = nn.Sequential()
    model:add(nn.SplitTable(1))
    model:add(siamese)
    model:add(nn.PairwiseDistance(2))

    return model
end

function writeMarginStats(gradErrs)
    file = io.open("marginStats.txt", "w")
    
    for i = 1, gradErrs:size(1) do
        local gradErr = gradErrs[i]
        local numZeros = torch.eq(gradErr, 0):sum()
        file:write( numZeros .. ':' .. (batchSize - numZeros) )
        file:write("\n")
    end
    
    file:close()
end

function train(numIters)
   
    local gradErrs = torch.Tensor(numIters, batchSize)
    for iter = 1, numIters do

        -- Prepare data and labels for mini-batch
        pair_1 = torch.randn(batchSize, 3, imgSize, imgSize)
        pair_2 = torch.randn(batchSize, 3, imgSize, imgSize)
        batchInputs = torch.Tensor(2, batchSize, 3, imgSize, imgSize)
        batchInputs[1] = pair_1;  batchInputs[2] = pair_2;

        labels = torch.Tensor(batchSize)
        labels[1] = 1; labels[2] = -1; labels[3] = -1; labels[4] = 1; labels[5] = 1;

        -- Closure function to evaluate loss and gradient
        local eval = function(x)
            if x ~= params then params:copy(x) end
            gradParams:zero()

            local batchOutputs = net:forward(batchInputs)
            local batchLoss = loss:forward(batchOutputs, labels)
            local dLoss_dout = loss:backward(batchOutputs, labels)
            gradErrs[iter] = dLoss_dout
            net:backward(batchInputs, dLoss_dout)

            return batchLoss, gradParams
        end

        _, fs = optim.sgd(eval, params, sgd_params)
        xlua.progress(iter, numIters)

    end

    writeMarginStats(gradErrs)
end

net = getModel()
params, gradParams = net:getParameters()
print (params:size())

loss = nn.MaxMarginCriterion(1, 0.1)
sgd_params = {
    learningRate = 1e-2,
    weightDecay = 0.0005,
    momentum = 0.9
}

train(1000)
