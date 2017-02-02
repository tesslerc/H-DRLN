--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require 'image'
require 'hdf5'

local trans = torch.class('dqn.TransitionTable') 

function trans:__init(args)
    self.stateDim = args.stateDim
    self.numActions = args.numActions
    self.histLen = args.histLen
    self.knn_data_path = args.knn_data_path
    self.maxSize = args.maxSize or 1024^2
    self.bufferSize = args.bufferSize or 1024
    self.histType = args.histType or "linear"
    self.histSpacing = args.histSpacing or 1
    self.zeroFrames = args.zeroFrames or 1
    self.nonTermProb = args.nonTermProb or 1
    self.nonEventProb = args.nonEventProb or 1
    self.gpu = args.gpu
    self.numEntries = 0
    self.insertIndex = 0
    self.MaxEpisodeReward = 0
-- Tom
    print('first read done')
    self.number_clusters = 21
    self.num_knn_data = args.num_knn_data
    local myFile = hdf5.open(self.knn_data_path .. 'cluster_ids.h5', 'r')
    self.knn_cluster_ids = myFile:read('data'):partial({1, self.num_knn_data})
    self.knn_cluster_ids:apply(function(x) if x == 0 then return self.number_clusters end end) 
    myFile:close()
    self.firstfull = 0 -- Tom - indicate if the ER was full
    
    self.ER_Cluster_ids = {}
    self.ER_Cluster_counter = torch.zeros(self.number_clusters)
    for i=1,self.number_clusters do
	self.ER_Cluster_ids[i] = torch.zeros(1)
    end
    self.Cluster_prob = torch.zeros(self.number_clusters)
    self.Cluster_prob[1] = 3
    self.Cluster_prob[2] = 3
    self.Cluster_prob[11] = 1
    self.Cluster_prob[16] = 1
    self.Cluster_prob[17] = 1
    self.Cluster_prob[18] = 2
    self.Cluster_prob[19] = 2
    self.Cluster_prob[20] = 2
    self.Cluster_prob[21] = 1

-- end
    self.histIndices = {}
    local histLen = self.histLen
    if self.histType == "linear" then
        -- History is the last histLen frames.
        self.recentMemSize = self.histSpacing*histLen
        for i=1,histLen do
            self.histIndices[i] = i*self.histSpacing
        end
    elseif self.histType == "exp2" then
        -- The ith history frame is from 2^(i-1) frames ago.
        self.recentMemSize = 2^(histLen-1)
        self.histIndices[1] = 1
        for i=1,histLen-1 do
            self.histIndices[i+1] = self.histIndices[i] + 2^(7-i)
        end
    elseif self.histType == "exp1.25" then
        -- The ith history frame is from 1.25^(i-1) frames ago.
        self.histIndices[histLen] = 1
        for i=histLen-1,1,-1 do
            self.histIndices[i] = math.ceil(1.25*self.histIndices[i+1])+1
        end
        self.recentMemSize = self.histIndices[1]
        for i=1,histLen do
            self.histIndices[i] = self.recentMemSize - self.histIndices[i] + 1
        end
    end

-- Tom
    --self.EpisodeData = torch.LongTensor(self.maxSize,2):fill(0) -- stores trajectory time and reward , for a treajectory indexed by episode_index
    --self.InEpisodeTime = torch.LongTensor(self.maxSize):fill(0)
    --self.EpisodeIndex = torch.LongTensor(self.maxSize):fill(0)
    self.p = torch.LongTensor(self.maxSize):fill(0)
-- End Tom 
    self.s = torch.ByteTensor(self.maxSize, self.stateDim):fill(0)
    self.a = torch.LongTensor(self.maxSize):fill(0)
    self.target_zero = torch.LongTensor(self.maxSize):fill(0)
    self.r = torch.zeros(self.maxSize)
    self.t = torch.ByteTensor(self.maxSize):fill(0)
    self.action_encodings = torch.eye(self.numActions)

    -- Tables for storing the last histLen states.  They are used for
    -- constructing the most recent agent state more easily.
    self.recent_s = {}
    self.recent_a = {}
    self.recent_t = {}

    local s_size = self.stateDim*histLen
    self.buf_a      = torch.LongTensor(self.bufferSize):fill(0)
    self.buf_target_zero = torch.LongTensor(self.bufferSize):fill(0)
    self.buf_r      = torch.zeros(self.bufferSize)
    self.buf_term   = torch.ByteTensor(self.bufferSize):fill(0)
    self.buf_s      = torch.ByteTensor(self.bufferSize, s_size):fill(0)
    self.buf_s2     = torch.ByteTensor(self.bufferSize, s_size):fill(0)

    if self.gpu and self.gpu >= 0 then
        self.gpu_s  = self.buf_s:float():cuda()
        self.gpu_s2 = self.buf_s2:float():cuda()
    end
end


function trans:reset()
    self.numEntries = 0
    self.insertIndex = 0
end


function trans:size()
    return self.numEntries
end


function trans:empty()
    return self.numEntries == 0
end


function trans:fill_buffer()
    assert(self.numEntries >= self.bufferSize)
    -- clear CPU buffers
    self.buf_ind = 1
    local ind
    for buf_ind=1,self.bufferSize do
        local s, a, r, s2, term, target_zero = self:sample_one(1)
        self.buf_s[buf_ind]:copy(s)
        self.buf_a[buf_ind] = a
        self.buf_target_zero[buf_ind] = target_zero
        self.buf_r[buf_ind] = r
        self.buf_s2[buf_ind]:copy(s2)
        self.buf_term[buf_ind] = term
    end
    self.buf_s  = self.buf_s:float():div(255)
    self.buf_s2 = self.buf_s2:float():div(255)
    if self.gpu and self.gpu >= 0 then
        self.gpu_s:copy(self.buf_s)
        self.gpu_s2:copy(self.buf_s2)
    end
end


function trans:sample_one()
    assert(self.numEntries > 1)
    local index
    local valid = false
    while not valid do
        -- start at 2 because of previous action
        index = torch.random(2, self.numEntries-self.recentMemSize)
        --	index = self:get_Index()
        if self.t[index+self.recentMemSize-1] == 0 then
            valid = true
        end
        if self.nonTermProb < 1 and self.t[index+self.recentMemSize] == 0 and
            torch.uniform() > self.nonTermProb then
            -- Discard non-terminal states with probability (1-nonTermProb).
            -- Note that this is the terminal flag for s_{t+1}.
            valid = false
        end
        if self.nonEventProb < 1 and self.t[index+self.recentMemSize] == 0 and
            self.r[index+self.recentMemSize-1] == 0 and
            torch.uniform() > self.nonTermProb then
            -- Discard non-terminal or non-reward states with
            -- probability (1-nonTermProb).
            valid = false
        end
    end

    return self:get(index)
end


function trans:sample(batch_size)
    local batch_size = batch_size or 1
    assert(batch_size < self.bufferSize)

    if not self.buf_ind or self.buf_ind + batch_size - 1 > self.bufferSize then
        self:fill_buffer()
    end

    local index = self.buf_ind

    self.buf_ind = self.buf_ind+batch_size
    local range = {{index, index+batch_size-1}}

    local buf_s, buf_s2, buf_a, buf_r, buf_term, buf_target_zero = self.buf_s, self.buf_s2,
        self.buf_a, self.buf_r, self.buf_term, self.buf_target_zero
    if self.gpu and self.gpu >=0  then
        buf_s = self.gpu_s
        buf_s2 = self.gpu_s2
    end

    return buf_s[range], buf_a[range], buf_r[range], buf_s2[range], buf_term[range], buf_target_zero[range]
end


function trans:concatFrames(index, use_recent)
    if use_recent then
        s, t = self.recent_s, self.recent_t
    else
        s, t = self.s, self.t
    end

    local fullstate = s[1].new()
    fullstate:resize(self.histLen, unpack(s[1]:size():totable()))

    -- Zero out frames from all but the most recent episode.
    local zero_out = false
    local episode_start = self.histLen

    for i=self.histLen-1,1,-1 do
        if not zero_out then
            for j=index+self.histIndices[i]-1,index+self.histIndices[i+1]-2 do
                if t[j] == 1 then
                    zero_out = true
                    break
                end
            end
        end

        if zero_out then
            fullstate[i]:zero()
        else
            episode_start = i
        end
    end

    if self.zeroFrames == 0 then
        episode_start = 1
    end

    -- Copy frames from the current episode.
    for i=episode_start,self.histLen do
        fullstate[i]:copy(s[index+self.histIndices[i]-1])
    end

    return fullstate
end


function trans:concatActions(index, use_recent)
    local act_hist = torch.FloatTensor(self.histLen, self.numActions)
    if use_recent then
        a, t = self.recent_a, self.recent_t
    else
        a, t = self.a, self.t
    end

    -- Zero out frames from all but the most recent episode.
    local zero_out = false
    local episode_start = self.histLen

    for i=self.histLen-1,1,-1 do
        if not zero_out then
            for j=index+self.histIndices[i]-1,index+self.histIndices[i+1]-2 do
                if t[j] == 1 then
                    zero_out = true
                    break
                end
            end
        end

        if zero_out then
            act_hist[i]:zero()
        else
            episode_start = i
        end
    end

    if self.zeroFrames == 0 then
        episode_start = 1
    end

    -- Copy frames from the current episode.
    for i=episode_start,self.histLen do
        act_hist[i]:copy(self.action_encodings[a[index+self.histIndices[i]-1]])
    end

    return act_hist
end


function trans:get_recent()
    -- Assumes that the most recent state has been added, but the action has not
    return self:concatFrames(1, true):float():div(255)
end


function trans:get(index)
    local s = self:concatFrames(index)
    local s2 = self:concatFrames(index+1)
    local ar_index = index+self.recentMemSize-1
    --if self.a[ar_index]==0 then
        --assert( (self.a[ar_index] ~= 0),"action is zero, ar_index= "..ar_index.. "state is:: "..s)
    --end
    return s, self.a[ar_index], self.r[ar_index], s2, self.t[ar_index+1], self.target_zero[ar_index]
end


--function trans:add(s, a, r, term,in_episode_time,episode_index)
function trans:add(s, a, r, term, cluster_id, target_zero)
    --maxSize - size of the ER buffer
    -- numEntries - number of examples in the ER buffer (increases to maxSize and stays there)
    -- insertIndex - index to push new example. Always insert at next index, then wrap around

    assert(s, 'State cannot be nil')
    assert(a, 'Action cannot be nil')
    assert(r, 'Reward cannot be nil')

    -- Incremenet until at full capacity
    if self.numEntries < self.maxSize then
        self.numEntries = self.numEntries + 1
    end

    -- Always insert at next index, then wrap around
    self.insertIndex = self.insertIndex + 1

    -- Overwrite oldest experience once at capacity
    if self.insertIndex > self.maxSize then
	    self.firstfull   = 1 -- tom
        self.insertIndex = 1
    end

    if a == 0 then print(self.insertIndex) end
    assert(a~=0, 'Error, a==0')

    -- Overwrite (s,a,r,t) at insertIndex
    self.s[self.insertIndex] = s:clone():float()
    self.a[self.insertIndex] = a
    self.r[self.insertIndex] = r
    self.target_zero[self.insertIndex] = target_zero

-- Tom Knn

    
    -- once an example is removed from the ER remove it from the ER clusters strcuture
    if self.firstfull > 0  then
        if self.ER_Cluster_counter[self.p[self.insertIndex]] == 1 then
	    self.ER_Cluster_ids[self.p[self.insertIndex]][1] = 0
	else
	     self.ER_Cluster_ids[self.p[self.insertIndex]] = self.ER_Cluster_ids[self.p[self.insertIndex]]:narrow(1, 2, self.ER_Cluster_ids[self.p[self.insertIndex]]:size()[1]-1)
        end
        self.ER_Cluster_counter[self.p[self.insertIndex]]=self.ER_Cluster_counter[self.p[self.insertIndex]]-1
    end
    -- add the cluster index of the new example in the ER
    assert(cluster_id>=1,"index not in range(<1) ".. cluster_id)
    assert(cluster_id<=21,"index not in range(>21) ".. cluster_id)
    self.p[self.insertIndex] = cluster_id -- assert a cluster index for every example in the ER
    -- add the cluster index of the new example to the cluster ids data structure
    local ind = torch.zeros(1):fill(self.insertIndex)

    if self.ER_Cluster_ids[self.p[self.insertIndex]][1] == 0 then
	self.ER_Cluster_ids[self.p[self.insertIndex]] = ind
        self.ER_Cluster_counter[self.p[self.insertIndex]]= self.ER_Cluster_counter[self.p[self.insertIndex]]+1
    else
        self.ER_Cluster_ids[self.p[self.insertIndex]] = torch.cat(self.ER_Cluster_ids[self.p[self.insertIndex]],ind:clone())
        self.ER_Cluster_counter[self.p[self.insertIndex]]= self.ER_Cluster_counter[self.p[self.insertIndex]]+1
    end
    if (self.ER_Cluster_ids[self.p[self.insertIndex]]:size()[1] ~= self.ER_Cluster_counter[self.p[self.insertIndex]]) then
	print(self.ER_Cluster_ids[self.p[self.insertIndex]]:size()[1],' ', self.ER_Cluster_counter[self.p[self.insertIndex]])
    end


-- end
    self.t[self.insertIndex] = term

end


function trans:add_recent_state(s, term)
    local s = s:clone():float():mul(255):byte()
    if #self.recent_s == 0 then
        for i=1,self.recentMemSize do
            table.insert(self.recent_s, s:clone():zero())
            table.insert(self.recent_t, 1)
        end
    end

    table.insert(self.recent_s, s)
    if term then
        table.insert(self.recent_t, 1)
    else
        table.insert(self.recent_t, 0)
    end

    -- Keep recentMemSize states.
    if #self.recent_s > self.recentMemSize then
        table.remove(self.recent_s, 1)
        table.remove(self.recent_t, 1)
    end
end


function trans:add_recent_action(a)
    if #self.recent_a == 0 then
        for i=1,self.recentMemSize do
            table.insert(self.recent_a, 1)
        end
    end

    table.insert(self.recent_a, a)

    -- Keep recentMemSize steps.
    if #self.recent_a > self.recentMemSize then
        table.remove(self.recent_a, 1)
    end
end


--[[
Override the write function to serialize this class into a file.
We do not want to store anything into the file, just the necessary info
to create an empty transition table.

@param file (FILE object ) @see torch.DiskFile
--]]
function trans:write(file)
    file:writeObject({self.stateDim,
                      self.numActions,
                      self.histLen,
                      self.maxSize,
                      self.bufferSize,
                      self.numEntries,
                      self.insertIndex,
                      self.recentMemSize,
                      self.histIndices})
end


--[[
Override the read function to desearialize this class from file.
Recreates an empty table.

@param file (FILE object ) @see torch.DiskFile
--]]
function trans:read(file)
    local stateDim, numActions, histLen, maxSize, bufferSize, numEntries, insertIndex, recentMemSize, histIndices = unpack(file:readObject())
    self.stateDim = stateDim
    self.numActions = numActions
    self.histLen = histLen
    self.maxSize = maxSize
    self.bufferSize = bufferSize
    self.recentMemSize = recentMemSize
    self.histIndices = histIndices
    self.numEntries = 0
    self.insertIndex = 0

    self.s = torch.ByteTensor(self.maxSize, self.stateDim):fill(0)
    self.a = torch.LongTensor(self.maxSize):fill(0)
    self.target_zero = torch.LongTensor(self.maxSize):fill(0)    
    self.r = torch.zeros(self.maxSize)
    self.t = torch.ByteTensor(self.maxSize):fill(0)
    self.action_encodings = torch.eye(self.numActions)

    -- Tables for storing the last histLen states.  They are used for
    -- constructing the most recent agent state more easily.
    self.recent_s = {}
    self.recent_a = {}
    self.recent_t = {}

    self.buf_a      = torch.LongTensor(self.bufferSize):fill(0)
    self.buf_target_zero      = torch.LongTensor(self.bufferSize):fill(0)
    self.buf_r      = torch.zeros(self.bufferSize)
    self.buf_term   = torch.ByteTensor(self.bufferSize):fill(0)
    self.buf_s      = torch.ByteTensor(self.bufferSize, self.stateDim * self.histLen):fill(0)
    self.buf_s2     = torch.ByteTensor(self.bufferSize, self.stateDim * self.histLen):fill(0)

    if self.gpu and self.gpu >= 0 then
        self.gpu_s  = self.buf_s:float():cuda()
        self.gpu_s2 = self.buf_s2:float():cuda()
    end
end

-- Tom
function trans:Update_Episode_Data(index,time)
    self.EpisodeData[index][1] = time
    -- Debug print('self.EpisodeData[index][1]',self.EpisodeData[index][1])
end

function trans:Update_Episode_Reward(index,reward)
    self.EpisodeData[index][2] = reward
    --print('reward: self.EpisodeData[index][2]',self.EpisodeData[index][2])
    --print('time: self.EpisodeData[index][1]',self.EpisodeData[index][1])
end

function trans:Update_Max_Episode_Reward(reward)
    if reward>self.MaxEpisodeReward then
	self.MaxEpisodeReward = reward
    end
end

--[[

function trans:samplePolicy(batch_size)
    -- Same as sample, but rejects "near termination" states
    local batch_size = batch_size or 1
    assert(batch_size < self.bufferSize)

    if not self.buf_ind or self.buf_ind + batch_size - 1 > self.bufferSize then
        self:fill_bufferPolicy()
    end

    local index = self.buf_ind

    self.buf_ind = self.buf_ind+batch_size
    local range = {{index, index+batch_size-1}}

    local buf_s, buf_s2, buf_a, buf_r, buf_term = self.buf_s, self.buf_s2,
        self.buf_a, self.buf_r, self.buf_term
    if self.gpu and self.gpu >=0  then
        buf_s = self.gpu_s
        buf_s2 = self.gpu_s2
    end

    return buf_s[range], buf_a[range], buf_r[range], buf_s2[range], buf_term[range]
end
function trans:fill_bufferPolicy()
    assert(self.numEntries >= self.bufferSize)
    -- clear CPU buffers
    self.buf_ind = 1
    local ind
    for buf_ind=1,self.bufferSize do
        local s, a, r, s2, term = self:sample_onePolicy(1)
        self.buf_s[buf_ind]:copy(s)
        self.buf_a[buf_ind] = a
        self.buf_r[buf_ind] = r
        self.buf_s2[buf_ind]:copy(s2)
        self.buf_term[buf_ind] = term
    end
    self.buf_s  = self.buf_s:float():div(255)
    self.buf_s2 = self.buf_s2:float():div(255)
    if self.gpu and self.gpu >= 0 then
        self.gpu_s:copy(self.buf_s)
        self.gpu_s2:copy(self.buf_s2)
    end
end

function trans:sample_onePolicy()
    assert(self.numEntries > 1)
    local index
    local valid = false
    while not valid do
        -- start at 2 because of previous action
        index = self:get_Index()
        if self.t[index+self.recentMemSize-1] == 0 then
            valid = true
        end
        if self.nonTermProb < 1 and self.t[index+self.recentMemSize] == 0 and
            torch.uniform() > self.nonTermProb then
            -- Discard non-terminal states with probability (1-nonTermProb).
            -- Note that this is the terminal flag for s_{t+1}.
            valid = false
        end
        if self.nonEventProb < 1 and self.t[index+self.recentMemSize] == 0 and
            self.r[index+self.recentMemSize-1] == 0 and
            torch.uniform() > self.nonTermProb then
            -- Discard non-terminal or non-reward states with
            -- probability (1-nonTermProb).
            valid = false
        end
    end

    return self:get(index)
end

function trans:get_PolicyIndex()
    local flag = true
    --local TimeDiff = 30
    --local StartTime = 5
    local count = 0
    local EpisodeTime, InEpisodeTime, EpisodeReward
    while flag do
	count = count + 1
	index = torch.random(2, self.numEntries-self.recentMemSize) -- orig call
        local ar_index = index+self.recentMemSize-1
   	--[[InEpisodeTime = self.InEpisodeTime[ar_index]
   	local epIndex = self.EpisodeIndex[ar_index]
	EpisodeTime = self.EpisodeData[epIndex][1]
	EpisodeReward = self.EpisodeData[epIndex][2]
        
	local v = torch.bernoulli(EpisodeReward/self.MaxEpisodeReward)
	if ((EpisodeTime - InEpisodeTime) > TimeDiff ) and (InEpisodeTime>StartTime) and (v==1) then

	local v = torch.bernoulli(self.p[ar_index])
	if (v==1) then
	    flag = false
        end
    end
    return index
end
--]]
function trans:get_Index()
    --a = torch.multinomial(self.Cluster_probs, 1, true)
    --local Cluster_prob = torch.ones(self.number_clusters):div(self.number_clusters) 

    local Cluster_size = self.ER_Cluster_counter:clone()
    Cluster_size:cmul(self.Cluster_prob)
    Cluster_size:div(torch.sum(Cluster_size))
    assert(Cluster_size:size()[1] == 21 , "cluster_size:size(): " .. Cluster_size:size()[1] .. " error")
    local index = 0
    while index < 2 or index > self.numEntries-self.recentMemSize do
	    local cluster_index = torch.multinomial(Cluster_size, 1, true)[1]
            assert(cluster_index >= 1 , "cluster index: " .. cluster_index .. " out of bounds")
    	    assert(cluster_index <= 21 , "cluster index: " .. cluster_index .. " out of bounds")
	    local index_in_cluster_ids = torch.random(1, self.ER_Cluster_ids[cluster_index]:size()[1])
	    assert(index_in_cluster_ids >= 1 and index_in_cluster_ids<=self.ER_Cluster_ids[cluster_index]:size()[1], 'index out of bounds')

	    index = self.ER_Cluster_ids[cluster_index][index_in_cluster_ids]
	    assert(index >= 1  , 'index out of bounds')
	    assert(index<=self.numEntries  , 'index out of bounds')
    end
    --print('index: ',index)

    --[[local flag = true
    local count = 0
    local gamma = 0.995
    local p
    local EpisodeTime, InEpisodeTime, EpisodeReward
    while flag do
	count = count + 1
	index = torch.random(2, self.numEntries-self.recentMemSize) -- orig call
        local ar_index = index+self.recentMemSize-1
   	--[[InEpisodeTime = self.InEpisodeTime[ar_index]
   	local epIndex = self.EpisodeIndex[ar_index]
	EpisodeTime = self.EpisodeData[epIndex][1]
	EpisodeReward = self.EpisodeData[epIndex][2]
	p = torch.pow(gamma,self.MaxEpisodeReward - EpisodeReward)
	local v = torch.bernoulli()
	
	local v = torch.bernoulli(self.p[ar_index])
	if (v==1) then
	    flag = false
        end
    end
	--]]

    return index
end

-- End Tom
