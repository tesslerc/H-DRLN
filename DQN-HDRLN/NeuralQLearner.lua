--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

if not dqn then
    require 'initenv'
end

require 'hdf5'

local nql = torch.class('dqn.NeuralQLearner')


function nql:__init(args)
	self.tmp_counter = 0
	self.optimization_distance = 0
    self.state_dim  = args.state_dim -- State dimensionality.
    self.actions    = args.actions
    self.n_actions  = #self.actions

    self.finished = false -- has agent ever finished the task??
    if args.options then
        self.options    = args.options
        self.optionActions = args.optionsActions
        self.n_options  = self.options
    else
        self.options    = {}
        self.n_options  = 0
        self.optionActions = {}
    end

    self.option_actions_left = 0
    self.option_length = args.option_length
    self.option_accumulated_reward = 0

    self.main_agent = false
    self.distill = args.distill
    self.distilled_network = args.distilled_network
    self.distill_index = 1

    if args.skill_agent then
	     print("This is the main agent!")
       self.main_agent = true
       self.skill_agent = args.skill_agent
       self.primitive_actions = args.primitive_actions
       self.n_primitive_actions = #self.primitive_actions
    end
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

    if args.ddqn ~= 0 then
        self.ddqn	= true
        print("DDQN = true")
    else
        self.ddqn	= false
        print("DDQN = false")
    end


    self.gpu            = args.gpu

    self.ncols          = args.ncols or 1  -- number of color channels in input
    self.input_dims     = args.input_dims or {self.hist_len*self.ncols, 84, 84}
    self.preproc        = args.preproc  -- name of preprocessing network
    self.histType       = args.histType or "linear"  -- history type to use
    self.histSpacing    = args.histSpacing or 1
    self.nonTermProb    = args.nonTermProb or 1
    self.bufferSize     = args.bufferSize or 512

    self.qvals = torch.zeros(self.replay_memory)
    self.gameActions = torch.zeros(self.replay_memory)
    self.buf_activation = torch.FloatTensor(self.replay_memory+1, 512):fill(0)

    self.transition_params = args.transition_params or {}

    self.network    = args.network or self:createNetwork()

    -- check whether there is a network file
    local network_function
    if not (type(self.network) == 'string') then
        error("The type of the network provided in NeuralQLearner" ..
            " is not a string!")
    end

    local msg, err = pcall(require, self.network)
   --print("ERROR")
   --print(msg)
   --print(self.network)
   --print(err)
    if not msg then
        -- try to load saved agent
        local err_msg, exp = pcall(torch.load, self.network)
        if not err_msg then
            error("Could not find network file ")
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

    if self.main_agent ~= true and args.supervised_skills then
        self.network:remove(11)

        local myFile = hdf5.open(args.supervised_file, 'r')

      	local hiddenSize = myFile:read('hiddenWidth'):all()
        local numSkills = myFile:read('numberSkills'):all()

      	concat = nn.Concat(2)

        -- this is for utilizing hidden layer
        for i = 1, numSkills
        do
        	skill = nn.Sequential()
        	skill:add(nn.Linear(512, hiddenSize))
        	skill:add(nn.ReLU())
        	skill:add(nn.Linear(hiddenSize, 5))
        	concat:add(skill)
        end
      	self.network:add(concat)

      	print(self.network)
        -- load weights into split network
        for i = 0, (numSkills-1)
        do
        	self.network.modules[11].modules[i + 1].modules[1].weight[{{1, hiddenSize}, {1, 512}}] = torch.Tensor(512,hiddenSize):copy(myFile:read('W_hidden_' .. i):all()):t()
        	self.network.modules[11].modules[i + 1].modules[1].bias[{{1, hiddenSize}}] = torch.Tensor(hiddenSize):copy(myFile:read('b_hidden_' .. i):all())
        	self.network.modules[11].modules[i + 1].modules[3].weight[{{1, 5}, {1, hiddenSize}}] = torch.Tensor(hiddenSize,5):copy(myFile:read('W_output_' .. i):all()):t()
        	self.network.modules[11].modules[i + 1].modules[3].bias[{{1, 5}}] = torch.Tensor(5):copy(myFile:read('b_output_' .. i):all())
        end
      	myFile:close()
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
        bufferSize = self.bufferSize, option_length = self.option_length, options = self.options -- need to do this manually (maybe pass list of options?)
    }

    self.transitions = dqn.TransitionTable(transition_args)

    self.numSteps = 0 -- Number of perceived states.
    self.lastState = nil
    self.lastAction = nil
    self.v_avg = 0 -- V running average.
    self.tderr_avg = 0 -- TD error running average.

    self.q_max = 1
    self.r_max = 1

    if self.main_agent then
        self.w, self.dw = self.network:getParameters()
        self.dw:zero()

        self.deltas = self.dw:clone():fill(0)

        self.tmp= self.dw:clone():fill(0)
        self.g  = self.dw:clone():fill(0)
        self.g2 = self.dw:clone():fill(0)
    end
    if self.target_q then
        self.target_network = self.network:clone()
    end
    if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      cudnn.convert(self.network, cudnn)
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
    local s, a, r, s2, term, delta
    local q, q2, q2_max
    local network_q, i, j

    s = args.s
    a = args.a
    r = args.r
    s2 = args.s2
    term = args.term

    -- now we insert the discount into the reward itself (via perceive)
    temp_discount = self.discount
    --for optionCount = 1, self.n_options do
    --	if a == self.options[optionCount] then
    --		temp_discount = self.discount^self.option_length --number of steps
    --	end
    --end

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

    -- Double Q Learning - use online network to evaluate best action, use target network to evaluate the expected reward of such action
    local best_index
    if self.ddqn == true then
        local max_value
        local dq2_max = target_q_net:forward(s2):float()
        network_q = self.network:forward(s2):float():squeeze()
        for j=1,q2_max:size(1) do
            best_index = 1
            q2_max[j] = dq2_max[j][1]
            for i=1,network_q:size(2) do
                --print("Iter value")
                --print(network_q[j][i])
                if network_q[j][i] > network_q[j][best_index] then
                    q2_max[j] = dq2_max[j][i]
                    best_index = i
                end
            end
            --print("Max Q")
            --print(network_q[j][best_index])
        end
    end

    -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
    q2 = q2_max:clone():mul(temp_discount):cmul(term)

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

function nql:getQUpdateDistill(args) -- this is used for policy distillation, irrelevant for standard run of the algorithm
    --local s, a, r, s2, term, delta
    local s, term
    local q, q2, q2_max
    local i,j
    s = args.s
    --a = args.a
    --r = args.r
    --s2 = args.s2
    term = args.term

    --self.distill_index -- index to distill on
    -- now we insert the discount into the reward itself (via perceive)
    --temp_discount = self.discount
    --for optionCount = 1, self.n_options do
    --	if a == self.options[optionCount] then
    --		temp_discount = self.discount^self.option_length --number of steps
    --	end
    --end

    -- The order of calls to forward is a bit odd in order
    -- to avoid unnecessary calls (we only need 2).

    -- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)

    -- find indice where option begins (out of N indices of main network)
    local opt_begin = 1
    for i=1,(self.distill_index-1) do
        opt_begin = opt_begin + #(self.optionActions[i])
    end

    -- set delta = expected reward based on skill agent output
    --delta = self.skill_agent[self.distill_index].network:forward(s):float():clone():max(2)
    delta = self.network:forward(s):float():clone()
    --   print(delta)
    -- q = Q(s,a)
    local q_all = self.skill_agent[self.distill_index].network:forward(s):float():clone()
    --q = torch.FloatTensor(q_all:size(1))
    q = torch.FloatTensor(q_all:size()) -- [state][action]

    for i=1,q_all:size(1) do
        local action_reward = q_all[{ {i} ,{} }]
        local max_r = action_reward:clone():abs():max()
        local sum_reward = 0
        for j=1,q_all:size(2) do
            sum_reward = sum_reward + math.exp((action_reward[1][j] + max_r) / 0.1) -- 0.01 = tao
        end

        for j=1,q_all:size(2) do
            q[i][j] = math.exp((action_reward[1][j] + max_r) / 0.1) / sum_reward - 1 -- softmax .. 0.01 = tao
        end
    end
    local targets

    local total_actions = self.n_actions

    total_actions = 0
    for i=1, #self.optionActions do
        total_actions = total_actions + #(self.optionActions[i])
    end

    targets = torch.zeros(self.minibatch_size, total_actions):float()
   --print("opt_begin "..opt_begin.." total_actions "..total_actions.." opt end "..(#(self.optionActions[self.distill_index])+opt_begin-1))
    --print("TARGETS")
    for i=1,q_all:size(1) do
        for j=opt_begin,(#(self.optionActions[self.distill_index])+opt_begin-1) do
            targets[i][j] = - delta[i][j] + q[i][j-opt_begin+1]
            if targets[i][j] > 1 then
               targets[i][j] = 1
            elseif targets[i][j] < -1 then
               targets[i][j] = -1
            end
            --if self.tmp_counter > (4*4999000) then
            --   print(i.." "..j.." "..targets[i][j].." "..delta[i][j].." "..q[i][j-opt_begin+1])
            --end
            self.optimization_distance = self.optimization_distance + (- delta[i][j] + q[i][j-opt_begin+1])^2
        end
    end

    if self.tmp_counter == 3999 then
    	print('opt_dist: '..(self.optimization_distance/(4000*(3+3+4+4))))
    	self.tmp_counter = -1
    	self.optimization_distance = 0
    end

    --print("TARGETS")
    --   print(targets)
    if self.gpu >= 0 then targets = targets:cuda() end

    return targets, delta
end

function nql:qLearnMinibatch()
    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw

    --if self.main_agent == true then -- in case we want to allow skill to learn also
    --skill_agent:qLearnMinibatch()
    --end
    assert(self.transitions:size() > self.minibatch_size)
    local i
    local s, a, r, s2, term = self.transitions:sample(self.minibatch_size)

    local targets, delta, q2_max = self:getQUpdate{s=s, a=a, r=r, s2=s2,
        term=term, update_qmax=true}

    -- zero gradients of parameters
    self.dw:zero()

    -- get new gradient
    self.network:backward(s, targets)

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
    self.w:add(self.deltas)
end

function nql:qLearnMinibatchDistill(args)
    local s, t, agent
    s = args.s
    t = args.t
    --   distill_index = args.distill_index
    --print("distill_index = "..self.distill_index)
    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw

    local i
    local targets, delta = self:getQUpdateDistill{s=s,
        term=term, update_qmax=true}

    -- zero gradients of parameters
    self.dw:zero()

    -- get new gradient
    -- TODO: Add kullback leibler divergence here (?)
    self.network:backward(s, targets)

    -- add weight cost to gradient
    self.dw:add(-self.wc, self.w)

    -- compute linearly annealed learning rate
    local t = 0--math.max(0, self.numSteps - self.learn_start)
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
    self.w:add(self.deltas)

   self.distill_index = self.distill_index%4+ 1
   self.tmp_counter = self.tmp_counter + 1
end


function nql:sample_validation_data()
    local s, a, r, s2, term = self.transitions:sample(self.valid_size)
    self.valid_s    = s:clone()
    self.valid_a    = a:clone()
    self.valid_r    = r:clone()
    self.valid_s2   = s2:clone()
    self.valid_term = term:clone()
end


function nql:compute_validation_statistics()
    local targets, delta, q2_max = self:getQUpdate{s=self.valid_s,
        a=self.valid_a, r=self.valid_r, s2=self.valid_s2, term=self.valid_term}

    self.v_avg = self.q_max * q2_max:mean()
    self.tderr_avg = delta:clone():abs():mean()
end


function nql:perceive(reward, rawstate, terminal, testing, testing_ep)
    if tonumber(reward) == 0 then -- this is for greedy exploration while not finished even once
	self.finished = true
    end

    -- Preprocess state (will be set to nil if terminal)
    local state = self:preprocess(rawstate):float()
    local curState
    local realReward = reward
    if self.lastAction ~= nil then
        if self.main_agent == true and #self.options > 0 then
            for i = 1, #self.options do
                if self.actions[self.lastAction] == self.options[i] or self.lastAction > self.n_actions then
                    self.option_accumulated_reward = self.option_accumulated_reward + reward*(self.discount^(self.option_length - self.option_actions_left-1))
                    reward = self.option_accumulated_reward
                    break
                end
            end
        end
    end

    if self.main_agent then -- this allows us to view what the agent receives as input
        local img = torch.FloatTensor(3,84,84)
        for i = 1,84 do
            for j = 1,84 do
                img[1][i][j]=state[(i-1)*84+j]
                img[2][i][j]=0
                img[3][i][j]=0
            end
        end
        win = image.display({image=img, win=win})
    end

    if self.max_reward then
        reward = math.min(reward, self.max_reward)
    end
    if self.min_reward then
        reward = math.max(reward, self.min_reward)
    end
    if self.rescale_r then
        self.r_max = math.max(self.r_max, math.abs(reward))
    end

    local previousFullState
    if self.lastState then
        previousFullState = self.transitions:get_recent()
    end

    self.transitions:add_recent_state(state, terminal)

    local currentFullState = self.transitions:get_recent()

    --Store transition s, a, r, s'
    if self.lastState and not testing then
        self.transitions:add(self.lastState, self.lastAction, reward, realReward,
            self.lastTerminal, priority)

		  -- this is for extracting data
	    local lastState = previousFullState:resize(1, unpack(self.input_dims))
	    if self.gpu >= 0 then
		      lastState = lastState:cuda()
	    end

	    local q = self.network:forward(lastState):float():squeeze()

	    local nodes = self.network:findModules('nn.Linear')
	    local activation = nodes[1].output:clone():float()

	    local maxq = q[1]

	    for a = 2, self.n_actions do
    		if q[a] > maxq then
    		    maxq = q[a]
    		end
	    end
	    print(self.transitions.insertIndex)
	    self.qvals[self.transitions.insertIndex] = maxq
    	self.buf_activation[self.transitions.insertIndex]:copy(activation)
    end
    -- end extract data section
    if self.numSteps == self.learn_start+1 and not testing then
        self:sample_validation_data()
    end

    curState = self.transitions:get_recent()
    curState = curState:resize(1, unpack(self.input_dims))

    -- Select action
    local actionIndex = 1
    local gameAction = 1
    local skillAction = {}

    if self.main_agent == true then
        if self.gpu >= 0 then
            state = state:cuda()
        end

	      local skill_state = curState:clone():cuda()
        if self.distilled_network then
          local q = self.skill_agent.network:forward(skill_state):float():squeeze()
          local optionStartIndex = 1
          for i=1, #self.options do
              local maxq = q[optionStartIndex]
              local besta = {optionStartIndex}

              -- Evaluate all other actions (with random tie-breaking)
              for a = optionStartIndex+1, optionStartIndex+(#self.optionActions[i])-1 do
                  if q[a] > maxq then
                      besta = { a }
                      maxq = q[a]
                  elseif q[a] == maxq then
                      besta[#besta+1] = a
                  end
              end
              self.bestq = maxq

              local r = torch.random(1, #besta)

              skillAction[i] = besta[r] - optionStartIndex + 1
              optionStartIndex = optionStartIndex + #self.optionActions[i]
          end
        else
          for i=1, #self.options do
              skillAction[i] = self.skill_agent[i]:perceive(reward, rawstate, terminal, true, 0) -- when needed, forward on the skill_agent
          end
        end
    end

    if not terminal then
        if self.option_actions_left == 0 then -- our code
            actionIndex = self:eGreedy(curState, testing_ep)
            gameAction = actionIndex
            self.lastOption = 0
	      end
        -- our code
        local isOption = false
        if self.main_agent then
            if actionIndex > #self.primitive_actions then
                isOption = true
            end
        end

        if isOption and self.main_agent == true and self.option_actions_left == 0 then -- adjust to multiple options
            self.lastOption = actionIndex
            self.option_actions_left = self.option_length
            self.option_accumulated_reward = 0
        end
	      print("Option actions left "..self.option_actions_left)
        if self.option_actions_left > 0 and self.main_agent == true then
            gameAction = skillAction[self.lastOption-self.n_primitive_actions]

            if self.option_actions_left ~= self.option_length then
                actionIndex = self.n_actions + 1 -- non possible action = during option
            end
            self.option_actions_left = self.option_actions_left - 1
        end
        -- end our code
    else
        self.option_actions_left = 0
        self.lastOption = 0
    end
    -- %%%%% store option index and option-policy index(index for actions chosen from the option policy)
    self.transitions:add_recent_action(actionIndex)

    -- This section disabled, we run it in parallel while simulator is playing
    --Do some Q-learning updates
    --if self.numSteps > self.learn_start and not testing and
    --self.numSteps % self.update_freq == 0 and actionIndex <= self.n_actions then
    --for i = 1, self.n_replay do
    --self:qLearnMinibatch()
    --end
    --end

    if not testing and actionIndex <= self.n_actions then -- added so only main network will forward numSteps
        self.numSteps = self.numSteps + 1
    end
    if self.main_agent == true then
        print("MainActionIndex: "..actionIndex.." Selected Skill: "..(self.lastOption-self.n_primitive_actions).." GameAction: "..gameAction)
    end

    self.lastState = state:clone()
    self.lastAction = actionIndex
    self.lastTerminal = terminal
    if self.main_agent == true and testing == true then
        print("EvalAction: actionIndex: "..actionIndex)
    end
    if self.target_q and self.numSteps % self.target_q == 1 and actionIndex <= self.n_actions then
        self.target_network = self.network:clone()
    end


    if not terminal then
        if not testing then
            --self.gameActions[self.transitions.numEntries + 1] = gameAction
        end
        return gameAction, actionIndex
    else
        if not testing then
            --self.gameActions[self.transitions.numEntries + 1] = 0
        end
        return 0, 0
    end
end


function nql:eGreedy(state, testing_ep)
    self.ep = testing_ep or (self.ep_end +
        math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
            math.max(0, self.numSteps - self.learn_start))/self.ep_endt))
    if not testing_ep and self.finished == false then
	     self.ep = math.max(self.ep, 0.5)
    end
    if torch.uniform() < self.ep then
        return torch.random(1, self.n_actions)
    else
        return self:greedy(state)
    end
end


function nql:greedy(state)
    -- Turn single state into minibatch.  Needed for convolutional nets.
    if state:dim() == 2 then
        assert(false, 'Input must be at least 3D')
        state = state:resize(1, state:size(1), state:size(2))
    end

    if self.gpu >= 0 then
        state = state:cuda()
    end

    local q = self.network:forward(state):float():squeeze()

    local maxq = q[1]
    local besta = {1}

    -- Controller
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
