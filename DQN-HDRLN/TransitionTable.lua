--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require 'image'

--[[
  We introduce a new concept called 'Success Memory'. This concept is a
  variation of the 'Prioritised Experience Replay', in which we give
  higher priority to trajectories which result in successfuly completing
  the task.
  The reason behind this is that we are working in a domain with sparse
  reward signals, and it allows the agent to 'learn from rare events' such as
  completing the task (which has low probability during initial training).
]]
priorityMemProbability = 0.2
maxPriorityMemDepth = 100 -- how much of the trajectory do we remember? (counted from the end)
local trans = torch.class('dqn.TransitionTable')


function trans:__init(args)
    self.stateDim = args.stateDim
    self.numActions = args.numActions
    self.histLen = args.histLen
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

    self.successIndex = 0
    self.successEntries = 0
    self.successMemSize = self.maxSize / 100 -- success memory is a smaller memory unit

    self.option_length = 0
    self.option = args.numActions + 1
    self.numOptions = 0
    if args.option_length then -- insert support for options
        self.option_length = args.option_length
        self.numOptions = #args.options
    end

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

    self.s = torch.ByteTensor(self.maxSize, self.stateDim):fill(0)
    self.a = torch.LongTensor(self.maxSize):fill(0)
    self.r = torch.zeros(self.maxSize)
    self.real_r = torch.zeros(self.maxSize)
    self.t = torch.ByteTensor(self.maxSize):fill(0)
    self.action_encodings = torch.eye(self.numActions)

    self.success_s = torch.ByteTensor(self.successMemSize, self.stateDim):fill(0)
    self.success_a = torch.LongTensor(self.successMemSize):fill(0)
    self.success_r = torch.zeros(self.successMemSize)
    self.success_real_r = torch.zeros(self.successMemSize)
    self.success_t = torch.ByteTensor(self.successMemSize):fill(0)

    -- Tables for storing the last histLen states.  They are used for
    -- constructing the most recent agent state more easily.
    self.recent_s = {}
    self.recent_a = {}
    self.recent_t = {}

    local s_size = self.stateDim*histLen
    self.buf_a      = torch.LongTensor(self.bufferSize):fill(0)
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
        local s, a, r, s2, term = self:sample_one(1)
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


function trans:sample_one()
    assert(self.numEntries > 1)
    local index
    local valid = false
    local prioritized = false
    if torch.uniform() < priorityMemProbability and (self.successEntries-self.recentMemSize-self.option_length) > self.histLen then
        prioritized = true
        while not valid do
            -- start at 2 because of previous action
            index = torch.random(2, self.successEntries-self.recentMemSize-self.option_length)

            -- if selected state received while skill in control, backtrack until state where HDRLN selected this option.
            while self.success_a[index+self.recentMemSize-1] > self.numActions do
                index = index - 1
                if index <= 2 then -- we need also to have at least 1 action before this
                    break
                end
            end
            if index > 1 then
                if self.success_t[index+self.recentMemSize-1] == 0 then
                    valid = true
                end
                if self.success_a[index+self.recentMemSize] > self.numActions and (self.success_a[index+self.recentMemSize-1] > self.numActions
                    or self.success_a[index+self.recentMemSize-1] < (self.numActions-self.numOptions)) then
                    -- Discard states where a[index] is skill based, but a[index-1] isn't skill initiator
                    valid = false
                end
                if self.nonTermProb < 1 and self.success_t[index+self.recentMemSize] == 0 and
                    torch.uniform() > self.nonTermProb then
                    -- Discard non-terminal states with probability (1-nonTermProb).
                    -- Note that this is the terminal flag for s_{t+1}.
                    valid = false
                end
                if self.nonEventProb < 1 and self.success_t[index+self.recentMemSize] == 0 and
                    self.success_real_r[index+self.recentMemSize-1] == 0 and
                    torch.uniform() > self.nonTermProb then
                    -- Discard non-terminal or non-reward states with
                    -- probability (1-nonTermProb).
                    valid = false
                end
            end
        end
    else
        while not valid do
            -- start at 2 because of previous action
            index = torch.random(2, self.numEntries-self.recentMemSize-self.option_length)

            -- if selected state received while skill in control, backtrack until state where HDRLN selected this option.
            while self.a[index+self.recentMemSize-1] > self.numActions do
                index = index - 1
                if index <= 1 then -- we need also to have at least 1 action before this
                    break
                end
            end
            if index > 1 then
                if self.t[index+self.recentMemSize-1] == 0 then
                    valid = true
                end
                if self.a[index+self.recentMemSize] > self.numActions and (self.a[index+self.recentMemSize-1] > self.numActions
                    or self.a[index+self.recentMemSize-1] < (self.numActions-self.numOptions)) then
                    -- Discard states where a[index+1] is skill based, but a[index] isn't skill initiator
                    valid = false
                end
                if self.nonTermProb < 1 and self.t[index+self.recentMemSize] == 0 and
                    torch.uniform() > self.nonTermProb then
                    -- Discard non-terminal states with probability (1-nonTermProb).
                    -- Note that this is the terminal flag for s_{t+1}.
                    valid = false
                end
                if self.nonEventProb < 1 and self.t[index+self.recentMemSize] == 0 and
                    self.real_r[index+self.recentMemSize-1] == 0 and
                    torch.uniform() > self.nonTermProb then
                    -- Discard non-terminal or non-reward states with
                    -- probability (1-nonTermProb).
                    valid = false
                end
            end
        end
    end
    --print("prioritized =  " .. tostring(prioritized) .. " index = " .. index .. " successEntries = " .. self.successEntries)
    local s, a, r, real_r, s2, t = self:get(index, prioritized)
    if real_r ~= 0 then
        -- Terminal state is due to 'timeout', this isn't a real metric the agent can real, and doesn't teach us about the MDP.
        -- Since when we terminate is arbitrary, we set terminal to false.
        t = 0
    end
    return s, a, r, s2, t
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

    local buf_s, buf_s2, buf_a, buf_r, buf_term = self.buf_s, self.buf_s2,
        self.buf_a, self.buf_r, self.buf_term
    if self.gpu and self.gpu >=0  then
        buf_s = self.gpu_s
        buf_s2 = self.gpu_s2
    end

    return buf_s[range], buf_a[range], buf_r[range], buf_s2[range], buf_term[range]
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

function trans:concatPrioFrames(index, use_recent)
    if use_recent then
        s, t = self.recent_s, self.recent_t
    else
        s, t = self.success_s, self.success_t
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


function trans:get(index, prioritized)
    local s
    -- if self.a[ar_index] == option - then get sk by changing index+1 to index+k
    local s2
    local t_index
    local ar_index = index+self.recentMemSize-1

    -- if selected from prioritized memory
    if prioritized ~= nil and prioritized == true then
        s = self:concatPrioFrames(index)
        -- if we selected an action, we need to look forward for the 'exit state' of the skill (after N steps or when terminal reached).
        if self.success_a[ar_index] > (self.numActions - self.numOptions) and self.success_t[ar_index] ~= 1 then
            local option_end = 0
            while (ar_index+option_end+1) <= self.successEntries do
                option_end = option_end + 1
                if self.success_a[ar_index+option_end] <= self.numActions or self.success_t[ar_index+option_end] == 1 then -- search forward until skill ends
                    break
                end
            end
            t_index = option_end
        else
            t_index = 1
        end
        s2 = self:concatPrioFrames(index+t_index)
        --print("Success - Action1: "..self.success_a[ar_index]..", Action2: "..self.success_a[ar_index+t_index]..", IndexDiff: "..t_index.." Reward: "..self.success_r[ar_index+(t_index-1)].." RealReward: "..self.success_real_r[ar_index+(t_index-1)].." Terminal: "..self.success_t[ar_index+t_index])
        return s, self.success_a[ar_index], self.success_r[ar_index+(t_index-1)], self.success_real_r[ar_index+(t_index-1)], s2, self.success_t[ar_index+t_index]
    else -- not prio memory
        s = self:concatFrames(index)
        -- if we selected an action, we need to look forward for the 'exit state' of the skill (after N steps or when terminal reached).
        if self.a[ar_index] > (self.numActions - self.numOptions) and self.t[ar_index] ~= 1 then
            local option_end = 0
            while (ar_index+option_end+1) <= self.numEntries do
                option_end = option_end + 1
                if self.a[ar_index+option_end] <= self.numActions or self.t[ar_index+option_end] == 1 then -- search forward until skill ends
                    break
                end
            end
            t_index = option_end
        else
            t_index = 1
        end
        s2 = self:concatFrames(index+t_index)
	      --print("Action1: "..self.a[ar_index]..", Action2: "..self.a[ar_index+t_index]..", IndexDiff: "..t_index.." Reward: "..self.r[ar_index+(t_index-1)].." RealReward: "..self.real_r[ar_index+(t_index-1)].." Terminal: "..self.t[ar_index+t_index])
        return s, self.a[ar_index], self.r[ar_index+(t_index-1)], self.real_r[ar_index+(t_index-1)], s2, self.t[ar_index+t_index]
    end
end


function trans:add(s, a, r, real_r, term)
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
        self.insertIndex = 1
    end

    -- Overwrite (s,a,r,t) at insertIndex
    self.s[self.insertIndex] = s:clone():float():mul(255)
    self.a[self.insertIndex] = a
    self.r[self.insertIndex] = r
    self.real_r[self.insertIndex] = real_r

    if term then
        self.t[self.insertIndex] = 1
    else
        self.t[self.insertIndex] = 0
    end

    -- For success memory, we want to store whole trajectory (or last S steps), so need to search backwards for first state of trajectory
    if self.insertIndex > 2 and self.real_r[self.insertIndex-1] == 0 and self.t[self.insertIndex] == 1 then -- if previous s,a ended in terminal gave us success reward
        local index = 1
        while self.t[self.insertIndex-index-1] ~= 1 and (self.insertIndex - index) > 2 and index < maxPriorityMemDepth do
            index = index + 1
        end
        index = index - 1
        while (index >= 0) do
            self.successIndex = self.successIndex + 1
            if self.successIndex > self.successMemSize then
                self.successIndex = 1
            end
            if self.successEntries < self.successMemSize then
                self.successEntries = self.successEntries + 1
            end

            self.success_s[self.successIndex] = self.s[self.insertIndex-index]:clone()
            self.success_a[self.successIndex] = self.a[self.insertIndex-index]
            self.success_r[self.successIndex] = self.r[self.insertIndex-index]
            self.success_real_r[self.successIndex] = self.real_r[self.insertIndex-index]
            self.success_t[self.successIndex] = self.t[self.insertIndex-index]
            index = index - 1
            --print("successIndex: " .. self.successIndex)
        end
    end
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

    self.successIndex = 0
    self.successEntries = 0
    self.successMemSize = self.maxSize / 100

    self.s = torch.ByteTensor(self.maxSize, self.stateDim):fill(0)
    self.a = torch.LongTensor(self.maxSize):fill(0)
    self.r = torch.zeros(self.maxSize)
    self.real_r = torch.zeros(self.maxSize)
    self.t = torch.ByteTensor(self.maxSize):fill(0)
    self.action_encodings = torch.eye(self.numActions)

    self.success_s = torch.ByteTensor(self.successMemSize, self.stateDim):fill(0)
    self.success_a = torch.LongTensor(self.successMemSize):fill(0)
    self.success_r = torch.zeros(self.successMemSize)
    self.success_t = torch.ByteTensor(self.successMemSize):fill(0)

    -- Tables for storing the last histLen states.  They are used for
    -- constructing the most recent agent state more easily.
    self.recent_s = {}
    self.recent_a = {}
    self.recent_t = {}

    self.buf_a      = torch.LongTensor(self.bufferSize):fill(0)
    self.buf_r      = torch.zeros(self.bufferSize)
    self.buf_term   = torch.ByteTensor(self.bufferSize):fill(0)
    self.buf_s      = torch.ByteTensor(self.bufferSize, self.stateDim * self.histLen):fill(0)
    self.buf_s2     = torch.ByteTensor(self.bufferSize, self.stateDim * self.histLen):fill(0)

    if self.gpu and self.gpu >= 0 then
        self.gpu_s  = self.buf_s:float():cuda()
        self.gpu_s2 = self.buf_s2:float():cuda()
    end
end
