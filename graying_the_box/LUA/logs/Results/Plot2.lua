require 'gnuplot'

n = 0
local file1 = io.open("./Results/reward2.txt")
if file1 then
    for line in file1:lines() do
	n = n+1
    end
end
file1:close()

local file1 = io.open("./Results/reward1.txt")
local file2 = io.open("./Results/reward2.txt")

i = 1


x = torch.Tensor(n)
if file1 then
    for line in file1:lines() do
        x[i] = tonumber(line)
	i = i+1
    end
end


x_ax = torch.range(1,i-1)


i = 1
y = torch.Tensor(n)
if file2 then
    for line in file2:lines() do
        y[i] = tonumber(line)
	i = i+1
    end
end
y_ax = torch.range(1,i-1)


if x_ax:numel()>y_ax:numel() then
	n = y_ax:numel()
else
	n = x_ax:numel()
end

data1 = torch.Tensor(n)
data2 = torch.Tensor(n)
for i = 1,n do
	data1[i] = x[i]
	data2[i] = y[i]
end


gnuplot.pngfigure('reward.png')
gnuplot.title('Average Reward over training')
gnuplot.plot({'Policy Noise',data1},{'Q noise',data2})
gnuplot.xlabel('Training epochs')
gnuplot.ylabel('Average testing reward')
gnuplot.movelegend('left','top')
gnuplot.plotflush()


