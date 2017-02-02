require 'gnuplot'

n = 0
local file1 = io.open("./Results/reward.txt")
if file1 then
    for line in file1:lines() do
	n = n+1
    end
end
file1:close()

m = 0
local file1 = io.open("./Results/conv1.txt")
if file1 then
    for line in file1:lines() do
	m = m+1
    end
end
file1:close()
m = m-1
local file1 = io.open("./Results/reward.txt")
local file2 = io.open("./Results/TD.txt")
local file3 = io.open("./Results/vavg.txt")
local file4 = io.open("./Results/conv1.txt")
local file5 = io.open("./Results/conv2.txt")
local file6 = io.open("./Results/conv3.txt")
local file7 = io.open("./Results/lin1.txt")
local file8 = io.open("./Results/lin2.txt")

i = 1
conv1_norm = torch.Tensor(m/4-1)
conv1_norm_max = torch.Tensor(m/4-1)
conv1_grad = torch.Tensor(m/4-1)
conv1_grad_max = torch.Tensor(m/4-1)

if file4 then
    for line in file4:lines() do
	if i == 1 then
	print(i)
	elseif i == m-4 then
	 	break
	elseif i % 4 == 2 then
		conv1_norm[math.floor(i/4)+1] = tonumber(line)
	elseif i % 4 == 3 then
		conv1_norm_max[math.floor(i/4)+1] = tonumber(line)
	elseif i % 4 == 0 then
		conv1_grad[math.floor(i/4)+1] = tonumber(line)
	elseif i % 4 == 1 then
		conv1_grad_max[math.floor(i/4)+1] = tonumber(line)
	end
	i = i+1
    end
end

gnuplot.pngfigure('./Results/tmp/conv1.png')
gnuplot.title('conv 1 over training')
gnuplot.plot({'Norm',conv1_norm},{'Grad',conv1_grad})
gnuplot.xlabel('Training epochs')
gnuplot.movelegend('left','top')
gnuplot.plotflush()


gnuplot.pngfigure('./Results/tmp/conv1_max.png')
gnuplot.title('conv 1 max over training')
gnuplot.plot({'Max norm',conv1_norm_max},{'Max grad',conv1_grad_max})
gnuplot.xlabel('Training epochs')
gnuplot.movelegend('left','top')
gnuplot.plotflush()


i = 1
conv2_norm = torch.Tensor(m/4-1)
conv2_norm_max = torch.Tensor(m/4-1)
conv2_grad = torch.Tensor(m/4-1)
conv2_grad_max = torch.Tensor(m/4-1)

if file5 then
    for line in file5:lines() do
	if i == 1 then
	print(i)
	elseif i == m-4 then
	 	break
	elseif i % 4 == 2 then
		conv2_norm[math.floor(i/4)+1] = tonumber(line)
	elseif i % 4 == 3 then
		conv2_norm_max[math.floor(i/4)+1] = tonumber(line)
	elseif i % 4 == 0 then
		conv2_grad[math.floor(i/4)+1] = tonumber(line)
	elseif i % 4 == 1 then
		conv2_grad_max[math.floor(i/4)+1] = tonumber(line)
	end
	i = i+1
    end
end

gnuplot.pngfigure('./Results/tmp/conv2.png')
gnuplot.title('conv 2 over training')
gnuplot.plot({'Norm',conv2_norm},{'Grad',conv2_grad})
gnuplot.xlabel('Training epochs')
gnuplot.movelegend('left','top')
gnuplot.plotflush()


gnuplot.pngfigure('./Results/tmp/conv2_max.png')
gnuplot.title('conv 2 max over training')
gnuplot.plot({'Max norm',conv2_norm_max},{'Max grad',conv2_grad_max})
gnuplot.xlabel('Training epochs')
gnuplot.movelegend('left','top')
gnuplot.plotflush()

i = 1
conv3_norm = torch.Tensor(m/4-1)
conv3_norm_max = torch.Tensor(m/4-1)
conv3_grad = torch.Tensor(m/4-1)
conv3_grad_max = torch.Tensor(m/4-1)

if file6 then
    for line in file6:lines() do
	if i == 1 then
	print(i)
	elseif i == m-4 then
	 	break
	elseif i % 4 == 2 then
		conv3_norm[math.floor(i/4)+1] = tonumber(line)
	elseif i % 4 == 3 then
		conv3_norm_max[math.floor(i/4)+1] = tonumber(line)
	elseif i % 4 == 0 then
		conv3_grad[math.floor(i/4)+1] = tonumber(line)
	elseif i % 4 == 1 then
		conv3_grad_max[math.floor(i/4)+1] = tonumber(line)
	end
	i = i+1
    end
end

gnuplot.pngfigure('./Results/tmp/conv3.png')
gnuplot.title('conv 3 over training')
gnuplot.plot({'Norm',conv3_norm},{'Grad',conv3_grad})
gnuplot.xlabel('Training epochs')
gnuplot.movelegend('left','top')
gnuplot.plotflush()


gnuplot.pngfigure('./Results/tmp/conv3_max.png')
gnuplot.title('conv 3 max over training')
gnuplot.plot({'Max norm',conv3_norm_max},{'Max grad',conv3_grad_max})
gnuplot.xlabel('Training epochs')
gnuplot.movelegend('left','top')
gnuplot.plotflush()

i = 1
lin1_norm = torch.Tensor(m/4-1)
lin1_norm_max = torch.Tensor(m/4-1)
lin1_grad = torch.Tensor(m/4-1)
lin1_grad_max = torch.Tensor(m/4-1)

if file7 then
    for line in file7:lines() do
	if i == 1 then
	print(i)
	elseif i == m-4 then
	 	break
	elseif i % 4 == 2 then
		lin1_norm[math.floor(i/4)+1] = tonumber(line)
	elseif i % 4 == 3 then
		lin1_norm_max[math.floor(i/4)+1] = tonumber(line)
	elseif i % 4 == 0 then
		lin1_grad[math.floor(i/4)+1] = tonumber(line)
	elseif i % 4 == 1 then
		lin1_grad_max[math.floor(i/4)+1] = tonumber(line)
	end
	i = i+1
    end
end

gnuplot.pngfigure('./Results/tmp/lin1.png')
gnuplot.title('lin1 over training')
gnuplot.plot({'Norm',lin1_norm},{'Grad',lin1_grad})
gnuplot.xlabel('Training epochs')
gnuplot.movelegend('left','top')
gnuplot.plotflush()


gnuplot.pngfigure('./Results/tmp/lin1_max.png')
gnuplot.title('lin1 max over training')
gnuplot.plot({'Max norm',lin1_norm_max},{'Max grad',lin1_grad_max})
gnuplot.xlabel('Training epochs')
gnuplot.movelegend('left','top')
gnuplot.plotflush()

i = 1
lin2_norm = torch.Tensor(m/4-1)
lin2_norm_max = torch.Tensor(m/4-1)
lin2_grad = torch.Tensor(m/4-1)
lin2_grad_max = torch.Tensor(m/4-1)

if file8 then
    for line in file8:lines() do
	if i == 1 then
	print(i)
	elseif i == m-4 then
	 	break
	elseif i % 4 == 2 then
		lin2_norm[math.floor(i/4)+1] = tonumber(line)
	elseif i % 4 == 3 then
		lin2_norm_max[math.floor(i/4)+1] = tonumber(line)
	elseif i % 4 == 0 then
		lin2_grad[math.floor(i/4)+1] = tonumber(line)
	elseif i % 4 == 1 then
		lin2_grad_max[math.floor(i/4)+1] = tonumber(line)
	end
	i = i+1
    end
end

gnuplot.pngfigure('./Results/tmp/lin2.png')
gnuplot.title('lin2 over training')
gnuplot.plot({'Norm',lin2_norm},{'Grad',lin2_grad})
gnuplot.xlabel('Training epochs')
gnuplot.movelegend('left','top')
gnuplot.plotflush()


gnuplot.pngfigure('./Results/tmp/lin2_max.png')
gnuplot.title('lin2 max over training')
gnuplot.plot({'Max norm',lin2_norm_max},{'Max grad',lin2_grad_max})
gnuplot.xlabel('Training epochs')
gnuplot.movelegend('left','top')
gnuplot.plotflush()


i = 1
x = torch.Tensor(n)
if file1 then
    for line in file1:lines() do
        x[i] = tonumber(line)
	i = i+1
    end
end

i = 1
y = torch.Tensor(n)
if file2 then
    for line in file2:lines() do
        y[i] = tonumber(line)
	i = i+1
    end
end

i = 1
z = torch.Tensor(n)
if file3 then
    for line in file3:lines() do
        z[i] = tonumber(line)
        z[i] = z[i]/y[i]
	i = i+1
    end
end

gnuplot.pngfigure('./Results/tmp/reward.png')
gnuplot.title('reward over testing')
gnuplot.plot(x)
gnuplot.plotflush()

gnuplot.pngfigure('./Results/tmp/vavg.png')
gnuplot.title('vavg over testing')
gnuplot.plot(y)
gnuplot.plotflush()

gnuplot.pngfigure('./Results/tmp/TD_Error.png')
gnuplot.title('Normalized TD error over testing')
gnuplot.plot(z)
gnuplot.plotflush()



