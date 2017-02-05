--[[
Copyright (c) 2014 Google Inc.
See LICENSE file for full terms of limited license.
]]

function make_step(action, socketnum)
    --print("Sending command")
    sendCommand(action, socketnum)
    return getState(socketnum, action)
end

function sendCommand(action, socketnum)
    local s = outSocket
    local err
    while not s do
        outSocket = require("socket")
        socket.select(nil, nil, 1) -- sleep for 1 sec

        s, err = outSocket.connect("127.0.0.1",tonumber(socketnum)+100)
        if s then
            outClient = s
            outClient:settimeout(1)
        end
        if err then
            print(err)
            print(socketnum)
        end
    end
    outClient:send(tostring(action))
end

function getState(socketnum, action)
    local s = inSocket
    local err
    local num_err = 0
    while not s do
        inSocket = require("socket")
        socket.select(nil, nil, 1) -- sleep for 1 sec

        s, err = inSocket.connect("127.0.0.1",tonumber(socketnum))
        if s then
            inClient = s
            inClient:settimeout(1)
        end
        if err then
            print(err)
            print(socketnum)
        end
    end
    local line=''
    local err,rest, line2
    local height=''
    local width=''
    local length=''
    local r,g,b = nil
    local reward,terminal = nil
    local rgbvector = ''
    local rgbtable = {}
    local client = inClient

    local failCount = 0

    width = 84 --tonumber(width)
    height = 84 --tonumber(height)

    local rgb = torch.FloatTensor(3,height,width) -- create matrix 3XheightXwidth

    length = tonumber(width*height*3)
    line =''
    local lengthremaining = length
    local readsize

    --if not err then
    repeat
        readsize = lengthremaining
        line,err,rest = client:receive(readsize) -- read RGB vector from socket
        if (line ~= nil and string.len(line) > 0) then
            rgbtable[#rgbtable +1] = line
            lengthremaining = lengthremaining - string.len(line)
        end
        if err then
            line = ''
            print(err)
            socket.select(nil, nil, 1) -- sleep for 1 second
            --break
            num_err = num_err + 1
            if (num_err > 10) then
                num_err = 0
                sendCommand(action, socketnum)
                failCount = failCount + 1
            end
            if failCount == 10 then
                inSocket = nil
                outSocket = nil
                rgb, reward, terminal = make_step(0, socketnum)
                terminal = true
                return rgb, reward, terminal
            end
        end
    until lengthremaining <= 0
    rgbvector = table.concat(rgbtable, "")
    --end
    -- parse RGB vector. Built like this: Red_int+'a'+Green_int+'a'+Blue_int+'a'... so on
    local received = 0 -- this runs over the whole matrix
    local arrayiterator = 1 -- this runs over the whole vector

    --if not err then
    repeat
        r = rgbvector:sub(arrayiterator,arrayiterator)
        arrayiterator = arrayiterator+1

        g = rgbvector:sub(arrayiterator,arrayiterator)
        arrayiterator = arrayiterator+1

        b = rgbvector:sub(arrayiterator,arrayiterator)
        arrayiterator = arrayiterator+1

        rgb[1][math.floor(received/width)+1][(received)%width+1] = string.byte(r)/255 --tonumber(r)/255 -- since we are using float.tensor, each RGB value is between 0 and 1 and not between 0 and 255
        rgb[2][math.floor(received/width)+1][(received)%width+1] = string.byte(g)/255 --tonumber(g)/255
        rgb[3][math.floor(received/width)+1][(received)%width+1] = string.byte(b)/255 --tonumber(b)/255

        received = received+1
    until received == (height*width)
    -- end parse RGB vector
    --end

    line = ''
    reward = ''
    --if not err then
    num_err = 0
    repeat
        reward = reward .. line
        line,err,rest = client:receive(1) -- receive reward value
        if err then
            line = ''
            print(err)
            --break
            num_err = num_err + 1
            if (num_err > 10) then
                num_err = 0
                sendCommand(action, socketnum)
                failCount = failCount + 1
            end
            if failCount == 10 then
                inSocket = nil
                outSocket = nil
                rgb, reward, terminal = make_step(0, socketnum)
                terminal = true
                return rgb, reward, terminal
            end
        end
    until line == 'a'
    --end
    line = ''
    terminal = ''
    --if not err then
    num_err = 0
    repeat
        terminal = terminal .. line
        line,err,rest = client:receive(1) -- receive terminal state (should be single byte, 0 or 1)
        if err then
            line = ''
            print(err)
            --break
            num_err = num_err + 1
            if (num_err > 10) then
                num_err = 0
                sendCommand(action, socketnum)
                failCount = failCount + 1
            end
            if failCount == 10 then
                inSocket = nil
                outSocket = nil
                rgb, reward, terminal = make_step(0, socketnum)
                terminal = true
                return rgb, reward, terminal
            end
        end
    until line == 'a'
    --end
    print("Reward: "..reward.." Terminal: "..terminal)
    if terminal == '0' then -- turn char into boolean, nicer for later on if statements
        terminal = false
    else
        terminal = true
    end

    --if not err then

    return rgb, reward, terminal
        --else
        --	print("Ran into an error, re-sending NOP for new state\n")
        --	--client:close()
        --	return make_step(0, socketnum)
        --end
end

if not dqn then
    require "initenv"
end

outSocket = nil -- global variables for minecraft
inSocket = nil -- global variables for minecraft
outClient = nil
inClient = nil

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-framework', '', 'name of training framework')
cmd:option('-env', '', 'name of environment to use')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
    'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
    'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', true,
    'saves the agent network in a separate file')
cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-steps', 10^5, 'number of training steps to perform')
cmd:option('-eval_steps', 10^5, 'number of evaluation steps')

cmd:option('-verbose', 2,
    'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')

cmd:option('-break_network','', 'Network used for room break skill')
cmd:option('-pickup_network','', 'Network used for pickup skill')
cmd:option('-place_network','', 'Network used for place block skill')
cmd:option('-navigate_network','', 'Network used for room solve skill')
cmd:option('-distilled_network','', 'Network used for room solve skill')

cmd:option('-skill_agent_params', '', 'string of skill agent parameters')

cmd:option('-socket','', 'Socket we connect to')
cmd:option('-reward_shaping','','Allow or not reward shaping')

cmd:option('-gif_file', '', 'GIF path to write session screens')
cmd:option('-csv_file', '', 'CSV path to write session data')

cmd:text()

local opt = cmd:parse(arg)
--.[[
--- General setup.
local game_env, game_actions, agent, opt = setup(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

-- file names from command line
--local gif_filename = opt.gif_file

-- start a new game
--
local screen, reward, terminal = make_step(0, opt.socket)

-- compress screen to JPEG with 100% quality
--local jpg = image.compressJPG(screen:squeeze(), 100)
-- create gd image from JPEG string
--local im = --gd.createFromJpegStr(jpg:storage():string())

-- remember the image and show it first
--local previm = im

local win = image.display({image=screen})

print("Started playing...")
for i = 0, 1000 do
--while agent.transitions.numEntries < agent.transitions.maxSize do
    --gif_filename = opt.gif_file .. i .. ".gif"
    -- convert truecolor to palette
    --im:trueColorToPalette(false, 256)

    -- write GIF header, use global palette and infinite looping
    --im:gifAnimBegin(gif_filename, true, 0)
    -- write first frame
    --im:gifAnimAdd(gif_filename, false, 0, 0, 50, --gd.DISPOSAL_NONE)
    -- play one episode (game)
    while not terminal do
        --print(agent.transitions.numEntries.." "..agent.transitions.maxSize)
        -- if action was chosen randomly, Q-value is 0
        agent.bestq = 0

        -- choose the best action
        local action_index = agent:perceive(reward, screen, terminal, false, 0) --0.05)

        -- play game in test mode (episodes don't end when losing a life)
        screen, reward, terminal = make_step(game_actions[action_index], opt.socket)

        -- display screen
        image.display({image=screen, win=win})

        -- create gd image from tensor
        --jpg = image.compressJPG(screen:squeeze(), 100)
        --im = --gd.createFromJpegStr(jpg:storage():string())

        -- use palette from previous (first) image
        --im:trueColorToPalette(false, 256)
        --im:paletteCopy(previm)

        -- write new GIF frame, no local palette, starting from left-top, 7ms delay
        --im:gifAnimAdd(gif_filename, false, 0, 0, 50, --gd.DISPOSAL_NONE)
        -- remember previous screen for optimal compression
        --previm = im
    end
    local action_index = agent:perceive(reward, screen, terminal, false, 0)
    screen, reward, terminal = make_step(9, opt.socket)
end
--]]
--[[
file = torch.DiskFile(opt.name .. '_statespace.t7', 'w')
file:writeObject(agent.transitions.s[{ {1,agent.transitions.numEntries} ,{} }])
file:close() -- make sure the data is written

file = torch.DiskFile(opt.name .. '_terminal.t7', 'w')
file:writeObject(agent.transitions.t[{{1,agent.transitions.numEntries}}])
file:close() -- make sure the data is written

print("done write")
--]]


local s_file = torch.DiskFile('navigate_statespace.t7', 'r')
local t_file = torch.DiskFile('navigate_terminal.t7', 'r')
local nav_s = s_file:readObject()
local nav_t = t_file:readObject()

local s_file = torch.DiskFile('pickup_statespace.t7', 'r')
local t_file = torch.DiskFile('pickup_terminal.t7', 'r')
local pickup_s = s_file:readObject()
local pickup_t = t_file:readObject()

local s_file = torch.DiskFile('break_statespace.t7', 'r')
local t_file = torch.DiskFile('break_terminal.t7', 'r')
local break_s = s_file:readObject()
local break_t = t_file:readObject()

--[[
local img = torch.FloatTensor(3,84,84)
term = true
require "socket"
for a = 1, 1000 do
        for i = 1,84 do
            for j = 1,84 do
                img[1][i][j]=break_s[a][(i-1)*84+j]
                img[2][i][j]=0
                img[3][i][j]=0
            end
        end
   if break_t[a] == 0 then
      term = false
   else
      term = true
   end
   agent:perceive(-1, img, term, false, 0)
   socket.select(nil,nil,5)
end--]]
--[[

--local s_file = torch.DiskFile('main_statespace.t7', 'r')
--local t_file = torch.DiskFile('main_terminal.t7', 'r')
--local main_s = t_file:readObject()
--local main_t = s_file:readObject()

--{navigate_agent, pickup_agent, break_agent, break_agent}

------------
--- Start load data
------------

agent.skill_agent[1].transitions.s = torch.ByteTensor(nav_t:size(1), agent.skill_agent[1].transitions.stateDim):fill(0)
agent.skill_agent[1].transitions.a = torch.LongTensor(nav_t:size(1)):fill(0)
agent.skill_agent[1].transitions.r = torch.zeros(nav_t:size(1))
agent.skill_agent[1].transitions.t = torch.ByteTensor(nav_t:size(1)):fill(0)

agent.skill_agent[1].transitions.t = nav_t:clone()
agent.skill_agent[1].transitions.numEntries = nav_t:size(1)
agent.skill_agent[1].transitions.s = nav_s:clone()

------------

agent.skill_agent[2].transitions.s = torch.ByteTensor(pickup_t:size(1), agent.skill_agent[1].transitions.stateDim):fill(0)
agent.skill_agent[2].transitions.a = torch.LongTensor(pickup_t:size(1)):fill(0)
agent.skill_agent[2].transitions.r = torch.zeros(pickup_t:size(1))
agent.skill_agent[2].transitions.t = torch.ByteTensor(pickup_t:size(1)):fill(0)

agent.skill_agent[2].transitions.t = pickup_t:clone()
agent.skill_agent[2].transitions.numEntries = pickup_t:size(1)
agent.skill_agent[2].transitions.s = pickup_s:clone()

------------

agent.skill_agent[3].transitions.s = torch.ByteTensor(break_t:size(1), agent.skill_agent[1].transitions.stateDim):fill(0)
agent.skill_agent[3].transitions.a = torch.LongTensor(break_t:size(1)):fill(0)
agent.skill_agent[3].transitions.r = torch.zeros(break_t:size(1))
agent.skill_agent[3].transitions.t = torch.ByteTensor(break_t:size(1)):fill(0)

agent.skill_agent[3].transitions.t = break_t:clone()
agent.skill_agent[3].transitions.numEntries = break_t:size(1)
agent.skill_agent[3].transitions.s = break_s:clone()

-------------

agent.skill_agent[4].transitions.s = torch.ByteTensor(break_t:size(1), agent.skill_agent[1].transitions.stateDim):fill(0)
agent.skill_agent[4].transitions.a = torch.LongTensor(break_t:size(1)):fill(0)
agent.skill_agent[4].transitions.r = torch.zeros(break_t:size(1))
agent.skill_agent[4].transitions.t = torch.ByteTensor(break_t:size(1)):fill(0)

agent.skill_agent[4].transitions.t = break_t:clone()
agent.skill_agent[4].transitions.numEntries = break_t:size(1)
agent.skill_agent[4].transitions.s = break_s:clone()

-------------
-- Done load data
-------------
--navigate_agent, pickup_agent, break_agent, place_agent
print("done read")
--print(states:size())

local img = torch.FloatTensor(3,84,84)
local a,t
for t = 1, 25000000 do
    if t%1000 == 0 then
       print(t)
    end
    if t%1000000 == 0 then
	agent.best_network = agent.network:clone()
	local s, a, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_r,
	   agent.valid_s2, agent.valid_term
	agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
	agent.valid_term = nil, nil, nil, nil, nil, nil, nil
	local w, dw, g, g2, delta, delta2, deltas, tmp = agent.w, agent.dw,
	   agent.g, agent.g2, agent.delta, agent.delta2, agent.deltas, agent.tmp
	agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
	agent.deltas, agent.tmp = nil, nil, nil, nil, nil, nil, nil, nil

	local filename = "distilled_network_0.1temp_batchof1"
	if opt.save_versions > 0 then
	   filename = filename .. "_" .. math.floor(step / opt.save_versions)
	end
	filename = filename
	torch.save(filename .. ".t7", {agent = agent,
	   model = agent.network,
	   best_model = agent.best_network,
	   reward_history = reward_history,
	   reward_counts = reward_counts,
	   episode_counts = episode_counts,
	   time_history = time_history,
	   v_history = v_history,
	   td_history = td_history,
	   qmax_history = qmax_history,
	   arguments=opt})
	if opt.saveNetworkParams then
	   local nets = {network=w:clone():float()}
	   torch.save(filename..'.params.t7', nets, 'ascii')
	end
	agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
	agent.valid_term = s, a, r, s2, term
	agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,

	agent.deltas, agent.tmp = w, dw, g, g2, delta, delta2, deltas, tmp
	print('Saved:', filename .. '.t7')
	io.flush()

	collectgarbage()
    end
    for a = 1, 4 do
        --for i = 1,84 do
        --    for j = 1,84 do
        --        img[1][i][j]=statespace[a][t][(i-1)*84+j]
        --        img[2][i][j]=0
        --        img[3][i][j]=0
        --    end
        --end
        --if t == 1 then
        --    win = image.display({image=img})
        --else
        --    win = image.display({image=img, win=win})
        --end

        local s,aa,r,s2,term = agent.skill_agent[a].transitions:sample(agent.minibatch_size)
        agent:qLearnMinibatchDistill{s=s, t=term}
    end
end


agent.best_network = agent.network:clone()
local s, a, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_r,
   agent.valid_s2, agent.valid_term
agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
agent.valid_term = nil, nil, nil, nil, nil, nil, nil
local w, dw, g, g2, delta, delta2, deltas, tmp = agent.w, agent.dw,
   agent.g, agent.g2, agent.delta, agent.delta2, agent.deltas, agent.tmp
agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
agent.deltas, agent.tmp = nil, nil, nil, nil, nil, nil, nil, nil

local filename = "distilled_network_0.1temp_batchof1"
if opt.save_versions > 0 then
   filename = filename .. "_" .. math.floor(step / opt.save_versions)
end
filename = filename
torch.save(filename .. ".t7", {agent = agent,
   model = agent.network,
   best_model = agent.best_network,
   reward_history = reward_history,
   reward_counts = reward_counts,
   episode_counts = episode_counts,
   time_history = time_history,
   v_history = v_history,
   td_history = td_history,
   qmax_history = qmax_history,
   arguments=opt})
if opt.saveNetworkParams then
   local nets = {network=w:clone():float()}
   torch.save(filename..'.params.t7', nets, 'ascii')
end
agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
agent.valid_term = s, a, r, s2, term
agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
agent.deltas, agent.tmp = w, dw, g, g2, delta, delta2, deltas, tmp
print('Saved:', filename .. '.t7')
io.flush()
collectgarbage()
--]]

-- end GIF animation and close CSV file
--gd.gifAnimEnd(gif_filename)

print("Finished playing, close window to exit!")
