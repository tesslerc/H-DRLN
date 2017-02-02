--[[
Copyright (c) 2014 Google Inc.
See LICENSE file for full terms of limited license.
]]
previous_learning_step = 0

function make_step(action, socketnum, agent, isOption)
    --print("Sending command")
    sendCommand(action, socketnum)
    if (agent ~= nil and isOption == false) then
        if (agent.numSteps > agent.learn_start and agent.numSteps > previous_learning_step) then
            previous_learning_step = agent.numSteps
            if ((agent.n_replay / agent.update_freq) > 1) then
                --print("Running "..math.floor(agent.n_replay / agent.update_freq).." qLearnMinibatch operations")
                for i = 1, math.floor(agent.n_replay / agent.update_freq) do
                    agent:qLearnMinibatch()
                end
            elseif (agent.numSteps % agent.update_freq == 0) then
                for i = 1, agent.n_replay do
                    agent:qLearnMinibatch()
                end
            end
        end
    end
    --print("Receiving state")
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

cmd:option('-verbose', 2,
    'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')
cmd:option('-gif_file', '', 'GIF path to write session screens')
cmd:option('-csv_file', '', 'CSV path to write session data')

cmd:option('-skill_agent_params', '', 'string of skill agent parameters')
cmd:option('-distilled_network','', 'Network used for room solve skill')
cmd:option('-socket','', 'Socket we connect to')
cmd:option('-reward_shaping','','Allow or not reward shaping')

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

states = torch.ByteTensor(agent.transitions.maxSize,84,84,3):fill(0)


local screen, reward, terminal = make_step(0, opt.socket)

-- compress screen to JPEG with 100% quality
--local jpg = image.compressJPG(screen:squeeze(), 100)
-- create gd image from JPEG string
--local im = --gd.createFromJpegStr(jpg:storage():string())

-- remember the image and show it first
--local previm = im
local win = image.display({image=screen})

print("Started playing...")
--for i = 0, 10 do
while (agent.transitions.numEntries < (agent.transitions.maxSize-10)) do
    --gif_filename = opt.gif_file .. i .. ".gif"
    -- convert truecolor to palette
    --im:trueColorToPalette(false, 256)

    -- write GIF header, use global palette and infinite looping
    --im:gifAnimBegin(gif_filename, true, 0)
    -- write first frame
    --im:gifAnimAdd(gif_filename, false, 0, 0, 50, --gd.DISPOSAL_NONE)
    -- play one episode (game)
    while not terminal do
        print(agent.transitions.numEntries.." "..agent.transitions.maxSize)
        -- if action was chosen randomly, Q-value is 0
        agent.bestq = 0

        -- choose the best action
        local action_index = agent:perceive(reward, screen, terminal, false, 0) --0.05)

        -- play game in test mode (episodes don't end when losing a life)
        screen, reward, terminal = make_step(game_actions[action_index], opt.socket)
	if (agent.transitions.numEntries > 0) then
		states[agent.transitions.numEntries]:copy(screen)
	end
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
	if (agent.transitions.numEntries >= (agent.transitions.maxSize-10)) then
		break
	end
    end
    local action_index = agent:perceive(reward, screen, terminal, false, 0)
    states[agent.transitions.numEntries]:copy(screen)
    screen, reward, terminal = make_step(9, opt.socket)
end

--
file = torch.DiskFile('hdrln_statespace.t7', 'w')
file:writeObject(agent.transitions.s[{ {1,agent.transitions.numEntries} ,{} }])
file:close() -- make sure the data is written

file = torch.DiskFile('hdrln_terminal.t7', 'w')
file:writeObject(agent.transitions.t[{{1,agent.transitions.numEntries}}])
file:close() -- make sure the data is written

file = torch.DiskFile('hdrln_fullstates.t7', 'w')
file:writeObject(states[{ {1,agent.transitions.numEntries}, {}, {}, {} }])
file:close()

file = torch.DiskFile('hdrln_actions.t7', 'w')
file:writeObject(agent.gameActions[{ { 1,agent.transitions.numEntries} }])
file:close()

file = torch.DiskFile('hdrln_rewards.t7', 'w')
file:writeObject(agent.transitions.real_r[{ { 1,agent.transitions.numEntries} }])
file:close()

file = torch.DiskFile('hdrln_qvals.t7', 'w')
file:writeObject(agent.qvals[{ { 1,agent.transitions.numEntries} }])
file:close()

file = torch.DiskFile('hdrln_activations.t7', 'w')
file:writeObject(agent.buf_activation[{ { 1,agent.transitions.numEntries} }])
file:close()

print("done write")
--]]

--[[
file = torch.DiskFile(opt.name .. '_statespace.t7', 'r')
states = file:readObject()
print("done read")
print(states:size())
local img = torch.FloatTensor(3,84,84)
for t = 1,states:size(1) do
    for i = 1,84 do
        for j = 1,84 do
            img[1][i][j]=states[t][(i-1)*84+j]
            img[2][i][j]=0
            img[3][i][j]=0
        end
    end
    print(t)
    if t == 1 then
        win = image.display({image=img})
    else
        win = image.display({image=img, win=win})
    end
end
--]]
-- end GIF animation and close CSV file
--gd.gifAnimEnd(gif_filename)

print("Finished playing, close window to exit!")
