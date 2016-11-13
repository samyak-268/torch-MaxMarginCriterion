local MaxMarginCriterion, parent = torch.class('nn.MaxMarginCriterion', 'nn.Criterion')

function MaxMarginCriterion:__init(bias, margin)
    parent.__init(self)
    self.bias = bias or 1
    self.margin = margin or 0.1
    self.sizeAverage = true
end

function MaxMarginCriterion:updateOutput(input, y)
    self.buffer = self.buffer or input.new()
    if not torch.isTensor(y) then
        self.ty = self.ty or input.new():resize(1)
        self.ty[1] = y
        y = self.ty
    end
    
    input_sq = torch.cmul(input, input)
    self.buffer:resizeAs(input):fill(self.margin - self.bias)
    self.buffer:add(input_sq):cmax(0)
    self.buffer[torch.eq(y, -1)] = 0
    self.output = self.buffer:sum()
   
    self.buffer:fill(self.margin + self.bias)
    self.buffer:add(-1, input_sq):cmax(0)
    self.buffer[torch.eq(y, 1)] = 0
    self.output = self.output + self.buffer:sum()

    if (self.sizeAverage == nil or self.sizeAverage == true) then 
        self.output = self.output / input:nElement()
    end

    return (self.output * 0.5)
end

function MaxMarginCriterion:updateGradInput(input, y)
    if not torch.isTensor(y) then 
        self.ty[1] = y
        y = self.ty
    end

    input_sq = torch.cmul(input, input)
    self.gradInput:resizeAs(input):copy(y)
    self.gradInput[ torch.cmul( torch.eq(y, 1), torch.lt(input_sq, self.bias-self.margin) ) ] = 0
    self.gradInput[ torch.cmul( torch.eq(y, -1), torch.gt(input_sq, self.bias+self.margin) ) ] = 0
    self.gradInput:cmul(input)
    
    if (self.sizeAverage == nil or self.sizeAverage == true) then
        self.gradInput:mul(1 / input:nElement())
    end

    return self.gradInput
end
