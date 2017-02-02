--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

if not dqn then
    require 'initenv'
end

local nql = torch.class('dqn.NeuralQLearner')


function nql:__init(args)

    self.state_dim  = args.state_dim -- State dimensionality.
    self.actions    = args.actions
    self.n_actions  = #self.actions
    self.verbose    = args.verbose
    self.best       = args.best

    --- epsilon annealing
    self.ep_start   = args.ep or 1
    self.ep         = self.ep_start -- Exploration probability.
    self.ep_end     = args.ep_end or self.ep
    self.ep_endt    = args.ep_endt or 1000000

    ---- learning rate annealing
    self.lr_start       = args.lr or 0.01 --Learning rate.
    self.lr             = self.lr_start
    self.lr_end         = args.lr_end or self.lr
    self.lr_endt        = args.lr_endt or 1000000
    self.wc             = args.wc or 0  -- L2 weight cost.
    self.minibatch_size = args.minibatch_size or 1
    self.valid_size     = args.valid_size or 500

    --- Q-learning parameters
    self.discount       = args.discount or 0.99 --Discount factor.
    self.update_freq    = args.update_freq or 1
    -- Number of points to replay per learning step.
    self.n_replay       = args.n_replay or 1
    -- Number of steps after which learning starts.
    self.learn_start    = args.learn_start or 0
     -- Size of the transition table.
    self.replay_memory  = args.replay_memory or 1000000
    self.hist_len       = args.hist_len or 1
    self.rescale_r      = args.rescale_r
    self.max_reward     = args.max_reward
    self.min_reward     = args.min_reward
    self.clip_delta     = args.clip_delta
    self.target_q       = args.target_q
    self.bestq          = 0

    self.gpu            = args.gpu

    self.ncols          = args.ncols or 1  -- number of color channels in input
    self.input_dims     = args.input_dims or {self.hist_len*self.ncols, 84, 84}
    self.preproc        = args.preproc  -- name of preprocessing network
    self.histType       = args.histType or "linear"  -- history type to use
    self.histSpacing    = args.histSpacing or 1
    self.nonTermProb    = args.nonTermProb or 1
    self.bufferSize     = args.bufferSize or 512

    self.transition_params = args.transition_params or {}

    self.network    = args.network or self:createNetwork()
 -- Tom
    self.num_knn_points = 20000
    self.knn = require 'knn'
    self.knn_data_path = '/home/tom/git/graying_the_box/data/seaquest/120k/knn/'
    -- load activations
    local myFile = hdf5.open(self.knn_data_path .. '../global_activations.h5', 'r')
    self.knn_activations = myFile:read('data'):partial({1, self.num_knn_points}, {1, 512})
    myFile:close()
    -- load screens
    local myFile = hdf5.open(self.knn_data_path .. '../states.h5', 'r')
    self.knn_states = myFile:read('data'):partial({1, self.num_knn_points}, {1, 84*84})
    myFile:close()

    print('second read done')
    self.knn_k = 10

    self.buf_ind = 0
    self.buf_size = 512
    local s_size = self.state_dim
    self.buf_a      	= torch.LongTensor(self.buf_size):fill(0)
    self.buf_r      	= torch.zeros(self.buf_size)
    self.buf_term   	= torch.ByteTensor(self.buf_size):fill(0)
    self.buf_s      	= torch.ByteTensor(self.buf_size, s_size):fill(0)
    self.buf_activation = torch.FloatTensor(self.bufferSize+1, 512):fill(0)
-- end
    -- check whether there is a network file
    local network_function
    if not (type(self.network) == 'string') then
        error("The type of the network provided in NeuralQLearner" ..
              " is not a string!")
    end

    local msg, err = pcall(require, self.network)
    if not msg then
        -- try to load saved agent
        local err_msg, exp = pcall(torch.load, self.network)
        if not err_msg then
            error("Could not find network file " .. self.network)
        end
        if self.best and exp.best_model then
            self.network = exp.best_model
        else
            self.network = exp.model
        end
    else
        print('Creating Agent Network from ' .. self.network)
        self.network = err
        self.network = self:network()
    end

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
    else
        self.network:float()
    end

    -- Load preprocessing network.
    if not (type(self.preproc == 'string')) then
        error('The preprocessing is not a string')
    end
    msg, err = pcall(require, self.preproc)
    if not msg then
        error("Error loading preprocessing net")
    end
    self.preproc = err
    self.preproc = self:preproc()
    self.preproc:float()

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
        self.tensor_type = torch.CudaTensor
    else
        self.network:float()
        self.tensor_type = torch.FloatTensor
    end

    -- Create transition table.
    ---- assuming the transition table always gets floating point input
    ---- (Foat or Cuda tensors) and always returns one of the two, as required
    ---- internally it always uses ByteTensors for states, scaling and
    ---- converting accordingly
    local transition_args = {
        stateDim = self.state_dim, numActions = self.n_actions,
        histLen = self.hist_len, gpu = self.gpu,
        maxSize = self.replay_memory, histType = self.histType,
        histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
        bufferSize = self.bufferSize,
        num_knn_points = self.num_knn_points,
        knn_data_path = self.knn_data_path
    }

    self.transitions = dqn.TransitionTable(transition_args)

    self.numSteps = 0 -- Number of perceived states.
    self.lastState = nil
    self.lastAction = nil
    self.last_activation = nil
    self.v_avg = 0 -- V running average.
    self.tderr_avg = 0 -- TD error running average.

    self.q_max = 1
    self.r_max = 1

    self.w, self.dw = self.network:getParameters()
    self.dw:zero()

    self.deltas = self.dw:clone():fill(0)

    self.tmp= self.dw:clone():fill(0)
    self.g  = self.dw:clone():fill(0)
    self.g2 = self.dw:clone():fill(0)

    if self.target_q then
        self.target_network = self.network:clone()
    end
end


function nql:reset(state)
    if not state then
        return
    end
    self.best_network = state.best_network
    self.network = state.model
    self.w, self.dw = self.network:getParameters()
    self.dw:zero()
    self.numSteps = 0
    print("RESET STATE SUCCESFULLY")
end


function nql:preprocess(rawstate)
    if self.preproc then
        return self.preproc:forward(rawstate:float())
                    :clone():reshape(self.state_dim)
    end

    return rawstate
end


function nql:getQUpdate(args)
    local s, a, r, s2, term, delta, target_zero
    local q, q2, q2_max

    s = args.s
    a = args.a
    r = args.r
    s2 = args.s2
    term = args.term
    target_zero = args.target_zero
    -- The order of calls to forward is a bit odd in order
    -- to avoid unnecessary calls (we only need 2).

    -- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
    term = term:clone():float():mul(-1):add(1)

    local target_q_net
    if self.target_q then
        target_q_net = self.target_network
    else
        target_q_net = self.network
    end

    -- Compute max_a Q(s_2, a).

    q2_max = target_q_net:forward(s2):float():max(2)

    for i = 1,self.minibatch_size do
 	   if target_zero[i] == 1 then q2_max[i] = 0 end
    end

    -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
    q2 = q2_max:clone():mul(self.discount):cmul(term)

    delta = r:clone():float()

    if self.rescale_r then
        delta:div(self.r_max)
    end
    delta:add(q2)

    -- q = Q(s,a)
    local q_all = self.network:forward(s):float()
    q = torch.FloatTensor(q_all:size(1))
    for i=1,q_all:size(1) do
        q[i] = q_all[i][a[i]]
    end
    delta:add(-1, q)

    if self.clip_delta then
        delta[delta:ge(self.clip_delta)] = self.clip_delta
        delta[delta:le(-self.clip_delta)] = -self.clip_delta
    end

    local targets = torch.zeros(self.minibatch_size, self.n_actions):float()
    for i=1,math.min(self.minibatch_size,a:size(1)) do
        targets[i][a[i]] = delta[i]
    end

    if self.gpu >= 0 then targets = targets:cuda() end

    return targets, delta, q2_max
end

function nql:pertubateState(args)
    local eps, eta, s ,a,term,SNR,printProb
    local p
    s = args.s:clone()
    a = args.a
    term = args.term
    local first_s = s:select(1,1)
    local stateSize = first_s:size(1)
    local new_s = torch.zeros(self.minibatch_size,stateSize)
    eps = 0.00001
    SNR  = 0.1
    printProb = 0.002
    local noiseType    = "addversial" -- addversial random
    local noise_func   = "inverse_abs" -- inverse_abs inverse_thresh
    local noise_init   = "rand" -- rand ones
    local noise

    if noiseType == "addversial" then
        printProb = 0.2	
	-- Init noise
	if noise_init == "ones"	then	
		noise = torch.ones(self.minibatch_size,stateSize)
	elseif noise_init == "rand" then
		noise = torch.rand(self.minibatch_size,stateSize)*2-1
	end
    	if self.gpu >= 0 then 
		new_s  = new_s:cuda()
		noise  = noise:cuda()
	end
	-- Create targets
	local targets = torch.zeros(self.minibatch_size, self.n_actions):float()
	for i=1,math.min(self.minibatch_size,a:size(1)) do
	   targets[i][a[i]] = 1
	end  
	targets = targets:float()
	if self.gpu >= 0 then targets = targets:cuda() end
	-- Compute inputs gradients
	--target_q_net:forward(s)
	--target_q_net:zeroGradParameters()
	local gradInput = self.network:backward(s,targets )
	-- Compute addversial noise from samples
	if noise_func == "inverse_thresh" then
		gradInput[gradInput:le(eps) and gradInput:ge(0)] = eps
		gradInput[gradInput:ge(-eps) and gradInput:le(0)] = -eps
		noise:cdiv(gradInput)
	elseif noise_func == "inverse_abs" then
		gradInput:abs()
		gradInput[gradInput:le(eps)] = eps
		noise:cdiv(gradInput)
	end

	-- Fix SNR per sample inside batch
	for i = 1,self.minibatch_size do
		local im_s = torch.reshape(s[i],torch.LongStorage{4,84,84})
		local im_noise = torch.reshape(noise[i],torch.LongStorage{4,84,84})
		local im_added_noise = torch.reshape(new_s[i],torch.LongStorage{4,84,84})
		for j = 1,4 do 
			local im_norm = torch.norm(im_s[j])
			local noise_norm = torch.norm(noise[j])
			if im_norm > 1 and noise_norm > 1 then
				im_added_noise[j]:add(im_s[j],SNR*im_norm/noise_norm,im_noise[j])
			else
				im_added_noise[j] = im_s[j]
			end
		end
		new_s[i] = torch.reshape(im_added_noise,torch.LongStorage{4*84*84}) 	
	end
	-- print
	local printFlag = torch.bernoulli(printProb)
	if printFlag == 1 then
		local im1 = torch.reshape(s[1],torch.LongStorage{4,84,84})
		local im2 = torch.reshape(new_s[1],torch.LongStorage{4,84,84})
		local im3 = im1-im2
		win1 = image.display({image=im1[1], win=win1})
		win2 = image.display({image=im2[1], win=win2})
		win3 = image.display({image=im3[1], win=win3})
		--print("noise",im3[1][10][10])
        end  		
    elseif noiseType == "random" then
	printProb = 0.02
	noise  = torch.randn(self.minibatch_size,stateSize)
	if self.gpu >= 0 then noise = noise:cuda() end
		for i = 1,self.minibatch_size do
		    local v = torch.bernoulli(p)
		    if v == 1 then
		    	new_s[i]:add(s[i],SNR*torch.norm(s[i])/torch.norm(noise[i]),noise[i])
			local printFlag = torch.bernoulli(printProb)
			if printFlag == 1 then
				local im1 = torch.reshape(s[i],torch.LongStorage{4,84,84})
				local im2 = torch.reshape(new_s[i],torch.LongStorage{4,84,84})
				local im3 = im1-im2
				win1 = image.display({image=im1[1], win=win1})
				win2 = image.display({image=im2[1], win=win2})
				win3 = image.display({image=im3[1], win=win3})
				--print("noise",im3[1][10][10])
        		end  
		    end
		      
		end
    end
   
   return new_s
end

function nql:AdversarialPertubate(args)
    local intensity = 0.0003
    local s = args.s:clone()
    local targets = args.targets:clone()
    local printProb = 0.2
    local gradInput = self.network:backward(s,targets )
    local noise = gradInput:clone():sign():mul(intensity)
    if self.gpu >= 0 then noise  = noise:cuda() end
    local new_s = s:clone():add(noise)	
    -- print
    local printFlag = torch.bernoulli(printProb)
    if printFlag == 1 then
	local im1 = torch.reshape(s[1],torch.LongStorage{4,84,84})
	local im2 = torch.reshape(new_s[1],torch.LongStorage{4,84,84})
	local im3 = im1-im2
	win1 = image.display({image=im1[1], win=win1})
	win2 = image.display({image=im2[1], win=win2})
	win3 = image.display({image=im3[1], win=win3})
	--print("noise",im3[1])
    end  	
    return new_s
end

function nql:qLearnMinibatch()
    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
    -- assert(self.transitions:size() > self.minibatch_size)

    local s, a, r, s2, term, target_zero
    local new_s 
    local p = 0.1

    local targets, delta, q2_max
   -- local v = torch.bernoulli(p)
    local v = 0
    if v == 1 then	-- old pertubation
	s, a, r, s2, term      = self.transitions:samplePolicy(self.minibatch_size)
         -- local tmp_targets, tmp_delta, tmp_q2_max = self:getQUpdate{s=s, a=a, r=r, s2=s2,term=term, update_qmax=true} -- forward target(s2) forward net(s)
	self.network:forward(s)
        self.dw:zero()
	new_s          = self:pertubateState{s=s,a=a,term=term}	-- backward net(s)
        local output   = torch.exp(self.network:forward(new_s)) -- forward net(new_s)
	local z        = torch.sum(output,2):float()
    	targets        = -output:float()
    	for i=1,self.minibatch_size do
            for j=1,output:size(2) do     
		targets[i][j]= targets[i][j]/z[i][1]
 	        if a[i]==j then
		    targets[i][j] = targets[i][j] + 1
		end
            end
	end
	if self.gpu >= 0 then targets = targets:cuda() end
    elseif v == 2 then -- addversial
	s, a, r, s2, term = self.transitions:sample(self.minibatch_size)
        targets, delta, q2_max = self:getQUpdate{s=s, a=a, r=r, s2=s2,term=term, update_qmax=true}
	self.dw:zero()
	new_s = self:AdversarialPertubate{targets=targets,s=s}
        targets, delta, q2_max = self:getQUpdate{s=new_s, a=a, r=r, s2=s2,term=term, update_qmax=true}
    else
	new_s, a, r, s2, term, target_zero = self.transitions:sample(self.minibatch_size)
        targets, delta, q2_max = self:getQUpdate{s=new_s, a=a, r=r, s2=s2,term=term, target_zero=target_zero, update_qmax=true}
    end
    -- zero gradients of parameters
    self.dw:zero()

    -- get new gradient
    self.network:backward(new_s, targets)

    -- add weight cost to gradient
    self.dw:add(-self.wc, self.w)

    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
                self.lr_end
    self.lr = math.max(self.lr, self.lr_end)

    -- use gradients
    self.g:mul(0.95):add(0.05, self.dw)
    self.tmp:cmul(self.dw, self.dw)
    self.g2:mul(0.95):add(0.05, self.tmp)
    self.tmp:cmul(self.g, self.g)
    self.tmp:mul(-1)
    self.tmp:add(self.g2)
    self.tmp:add(0.01)
    self.tmp:sqrt()

    -- accumulate update
    self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp)
    --print("v",v)
    --print("Gradnorm",torch.norm(self.deltas))
    if self.numSteps > self.learn_start then
    	self.w:add(self.deltas)
    end
end


function nql:sample_validation_data()
    local s, a, r, s2, term, target_zero = self.transitions:sample(self.valid_size)
    self.valid_s    = s:clone()
    self.valid_a    = a:clone()
    self.valid_r    = r:clone()
    self.valid_s2   = s2:clone()
    self.valid_term = term:clone()
    self.valid_target_zero = term:clone()
end


function nql:compute_validation_statistics()
    local targets, delta, q2_max = self:getQUpdate{s=self.valid_s,
        a=self.valid_a, r=self.valid_r, s2=self.valid_s2, term=self.valid_term, target_zero = self.valid_target_zero}

    self.v_avg = self.q_max * q2_max:mean()
    self.tderr_avg = delta:clone():abs():mean()
end


function nql:perceive(reward, rawstate, terminal, testing, testing_ep)
    -- Preprocess state (will be set to nil if terminal)
    local state = self:preprocess(rawstate):float()
    local curState

    if self.max_reward then
        reward = math.min(reward, self.max_reward)
    end
    if self.min_reward then
        reward = math.max(reward, self.min_reward)
    end
    if self.rescale_r then
        self.r_max = math.max(self.r_max, reward)
    end

    self.transitions:add_recent_state(state, terminal)

    local currentFullState = self.transitions:get_recent()

    --Store transition s, a, r, s'
    if self.lastState and not testing then
        self:add(self.lastState,self.lastAction,reward,self.lastTerminal,self.last_activation)
    end

    if self.numSteps == self.learn_start+1 and not testing then
        self:sample_validation_data()
    end

    curState= self.transitions:get_recent()
    curState = curState:resize(1, unpack(self.input_dims))

    -- Select action
    local actionIndex = 1
    local activation = torch.zeros(1,512):float()
    if not terminal then
        actionIndex,activation = self:eGreedy(curState, testing_ep)
    end
    self.transitions:add_recent_action(actionIndex)

    -- Tom Debug knn
    --local dists, indices = self.knn.knn(self.knn_activations, activation, self.knn_k)
    --print(self.transitions.knn_cluster_ids[indices[1][1]])
    -- end debug

    -- Tom - fix rmsprop for loading
    --Do some Q-learning updates
    --[[if self.numSteps > self.learn_start and not testing and
        self.numSteps % self.update_freq == 0 then--]]
     if self.transitions.numEntries > (self.transitions.recentMemSize + self.transitions.bufferSize) and not testing and self.numSteps % self.update_freq == 0 then
        for i = 1, self.n_replay do
            self:qLearnMinibatch()
        end
    end

    if not testing then
        self.numSteps = self.numSteps + 1
    end

    self.lastState = state:clone()
    self.lastAction = actionIndex
    self.lastTerminal = terminal
    self.last_activation = activation
    if self.target_q and self.numSteps % self.target_q == 1 then
        self.target_network = self.network:clone()
    end

    if not terminal then
        return actionIndex
    else
        return 0
    end
end


function nql:eGreedy(state, testing_ep)
    self.ep = testing_ep or (self.ep_end +
                math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                math.max(0, self.numSteps - self.learn_start))/self.ep_endt))
    -- Epsilon greedy

    if state:dim() == 2 then
        assert(false, 'Input must be at least 3D')
        state = state:resize(1, state:size(1), state:size(2))
    end

    if self.gpu >= 0 then state = state:cuda() end

    local q = self.network:forward(state):float():squeeze()
    local nodes = self.network:findModules('nn.Linear')
    local activation = nodes[1].output:clone():float()

    if torch.uniform() < self.ep then
        return torch.random(1, self.n_actions),activation
    else
        return self:greedy(q),activation
    end
end


function nql:greedy(q)
    -- Turn single state into minibatch.  Needed for convolutional nets.
    
    local maxq = q[1]
    local besta = {1}

    -- Evaluate all other actions (with random tie-breaking)
    for a = 2, self.n_actions do
        if q[a] > maxq then
            besta = { a }
            maxq = q[a]
        elseif q[a] == maxq then
            besta[#besta+1] = a
        end
    end
    self.bestq = maxq

    local r = torch.random(1, #besta)

    self.lastAction = besta[r]

    return besta[r]
end


function nql:createNetwork()
    local n_hid = 128
    local mlp = nn.Sequential()
    mlp:add(nn.Reshape(self.hist_len*self.ncols*self.state_dim))
    mlp:add(nn.Linear(self.hist_len*self.ncols*self.state_dim, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, self.n_actions))

    return mlp
end


function nql:_loadNet()
    local net = self.network
    if self.gpu then
        net:cuda()
    else
        net:float()
    end
    return net
end


function nql:init(arg)
    self.actions = arg.actions
    self.n_actions = #self.actions
    self.network = self:_loadNet()
    -- Generate targets.
    self.transitions:empty()
end


function nql:report()
    print(get_weight_norms(self.network))
    print(get_grad_norms(self.network))
end

-- Tom
function nql:UpdateEpisodeData(index,time)

    self.transitions:Update_Episode_Data(index,time)
    -- Debug print('index',index)
    -- Debug print('time',time)

end
function nql:UpdateEpisodeReward(index,reward)
    
    self.transitions:Update_Max_Episode_Reward(reward)

    self.transitions:Update_Episode_Reward(index,reward)
    self.transitions:Update_Episode_Reward(index-1,reward)
    self.transitions:Update_Episode_Reward(index-2,reward)
    self.transitions:Update_Episode_Reward(index-3,reward)
    self.transitions:Update_Episode_Reward(index-4,reward)
    -- Debug print('index',index)
    -- Debug print('time',time)

end

function nql:add(s,a,r,term,activation)
    -- 0. Buffer is full. find clusters and push to ER
    if self.buf_ind == self.buf_size then
        
        --update activations using current weights
--        local knn_states_ = zeros(self.num_knn_points,84*84*4)
--        for i=1:self.num_knn_points:
--            knn_states_[i,1:84*84] = self.knn_states[i]
--            if term[i-1]==1:
--                continue
--            if i>2:
--                knn_states_[i,1:84*84+1:2*84*84] = self.knn_states[i-1]
--            if i>3:
--                knn_states_[i,2:84*84+1:3*84*84] = self.knn_states[i-2]
--            if i>4:
--                knn_states_[i,3:84*84+1:4*84*84] = self.knn_states[i-3]

        local q = self.network:forward(knn_states_):float():squeeze()
        local nodes = self.network:findModules('nn.Linear')
        self.knn_activations = nodes[1].output:clone():float()
        
        -- push activation n+1 so we'll have a target activation for example n
        self.buf_activation[self.buf_size+1]:copy(activation)
        -- knn on n+1 exapmles
        -- indices: (0,20000)
	    local dists, indices = self.knn.knn(self.knn_activations, self.buf_activation, self.knn_k)
        --print('Time elapsed : ' .. timer:time().real .. ' seconds')
        -- loop over first n examples
        for i=1,self.buf_size do
            --cluster ind (1,21)
            local cluster_ind =  self.transitions.knn_cluster_ids[indices[i][1]]
            local next_cluster_ind =  self.transitions.knn_cluster_ids[indices[i+1][1]]
            --raise 'target_zero' flag if consequtive example belongs to a termination cluster
            local target_zero = 0
            if next_cluster_ind == 10 or next_cluster_ind == 12 or next_cluster_ind == 14 or next_cluster_ind == 15 then
                target_zero = 1
            end
            -- skip refuel clusters
            -- local dont_skip_me = (cluster_ind < 3) or (cluster_ind > 10 and cluster_ind ~= 13)
            local take_me = (cluster_ind==1) or
                            (cluster_ind==2) or
                            --(cluster_ind==3) or
                            --(cluster_ind==4) or
                            --(cluster_ind==5) or
                            --(cluster_ind==6) or
                            --(cluster_ind==7) or
                            --(cluster_ind==8) or
                            --(cluster_ind==9) or
                            --(cluster_ind==10) or
                            (cluster_ind==11) or
                            --(cluster_ind==12) or
                            --(cluster_ind==13) or
                            --(cluster_ind==14) or
                            --(cluster_ind==15) or
                            (cluster_ind==16) or
                            (cluster_ind==17) or
                            (cluster_ind==18) or
                            (cluster_ind==19) or
                            (cluster_ind==20) or
                            (cluster_ind==21)

            if take_me then
                self.transitions:add(self.buf_s[i],self.buf_a[i],self.buf_r[i],self.buf_term[i],cluster_ind,target_zero)
            end
        end
        -- reset index
        self.buf_ind = 0
    end

 -- 1. Buffer is not full. Push current example
    -- increment index
    self.buf_ind = self.buf_ind+1
    --insert s, a, r, termination, activation
    self.buf_s[self.buf_ind]=  s:clone():float():mul(255)
    self.buf_a[self.buf_ind] = a
    self.buf_r[self.buf_ind] = r

    if term then
        self.buf_term[self.buf_ind] = 1
    else
        self.buf_term[self.buf_ind] = 0
    end

    self.buf_activation[self.buf_ind]:copy(activation)

end
-- End Tom
