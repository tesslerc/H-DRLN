require 'gnuplot'

n = 0
local file1 = io.open("./Results/reward1.txt")
if file1 then
    for line in file1:lines() do
	n = n+1
    end
end
file1:close()

local file1 = io.open("./Results/reward1.txt")
local file2 = io.open("./Results/reward2.txt")
local file3 = io.open("./Results/reward3.txt")
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

i = 1
z = torch.Tensor(n)
if file3 then
    for line in file3:lines() do
        z[i] = tonumber(line)
	i = i+1
    end
end
z_ax = torch.range(1,i-1)


if x_ax:numel()>y_ax:numel() then
	n = y_ax:numel()
else 
	n = x_ax:numel()
end
if z_ax:numel()<n then
	n = z_ax:numel()
end


data1 = torch.Tensor(n)
data2 = torch.Tensor(n)
data3 = torch.Tensor(n)
for i = 1,n do
	data1[i] = x[i]
	data2[i] = y[i]
	data3[i] = z[i]
end


gnuplot.pngfigure('reward.png')
gnuplot.title('Average Reward over training')
gnuplot.plot({'Q noise',data1},{'Orig',data2},{'Policy noise',data3})
gnuplot.xlabel('Training epochs')
gnuplot.ylabel('Average testing reward')
gnuplot.movelegend('left','top')
gnuplot.plotflush()


