require 'nn'
require 'MaxMarginCriterion'

GradCheck = torch.class('GradCheck')

function GradCheck:__init(criterion, input, y)
    self.criterion = criterion
    self.input = input
    if not torch.isTensor(y) then
        self.y = torch.Tensor(1)
        self.y[1] = y
    else
        self.y = y
    end

    self.EPS = 1e-4
end

function GradCheck:getAnalyticGrad()
    local err = self.criterion:forward(self.input, self.y)
    return self.criterion:backward(self.input, self.y)
end

function GradCheck:getNumericalGrad()
    local batchSize = self.input:size(1)
    
    local I = torch.eye(batchSize)
    local numGrad = torch.Tensor(batchSize)
    for i = 1, batchSize do
        local inputPlusEps = torch.add(self.input, self.EPS, I[i])
        local inputMinusEps = torch.add(self.input, -self.EPS, I[i])
        numGrad[i] = ( self.criterion:forward(inputPlusEps, self.y) - self.criterion:forward(inputMinusEps, self.y) ) / (2 * self.EPS)
    end

    return numGrad
end

function GradCheck:runTests(analyticGrad, numericalGrad)
    local gradTests = torch.TestSuite()
    local tester = torch.Tester()

    function gradTests.test()
        tester:eq( analyticGrad, numericalGrad, 0.00001, "should be equal" )
    end

    tester:add(gradTests)
    tester:run()
end

function GradCheck:run()
    local analyticGrad = self:getAnalyticGrad()
    local numericalGrad = self:getNumericalGrad()
    self:runTests(analyticGrad, numericalGrad)
end

local criterion = nn.MaxMarginCriterion(1, 0.1)
local inputs = torch.Tensor{1.02, 1.05, 1.15, 0.87}
local labels = torch.Tensor{-1, -1, 1, 1}
local gradChecker = GradCheck.new(criterion, inputs, labels)
gradChecker:run()

