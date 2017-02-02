--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]
--for optionCount = 1, self.n_options do
--if a == self.options[optionCount] then
--temp_discount = self.discount^30
--end
--end

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

cmd:text()

local opt = cmd:parse(arg)

--- General setup.
local game_env, game_actions, agent, opt = setup(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local learn_start = agent.learn_start
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
local step = 0
local total_steps = 0
time_history[1] = 0

local total_reward
local nrewards
local nepisodes
local episode_reward
local last_eval = 0
--[[
local s_file = torch.DiskFile('break_statespace.t7', 'r')
local t_file = torch.DiskFile('break_terminal.t7', 'r')
local break_s = s_file:readObject()
local break_t = t_file:readObject()
--]]
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
   socket.select(nil,nil,1)
end--]]


--
local screen, reward, terminal = make_step(0, opt.socket)
--win = image.display({image=screen, win=win})

print("Iteration ..", step)
local win = nil
while step < opt.steps do
    print('iteration: ' .. total_steps .. ' main steps: '..step)
    total_steps = total_steps + 1

    if tonumber(opt.reward_shaping) == 0 and tonumber(reward) ~= 0 then
        reward = -1
    end

    local action_index, inner_action_index = agent:perceive(reward, screen, terminal)
    --print("action_index: "..action_index..", inner_action_index: "..inner_action_index)
    if inner_action_index <= (agent.n_actions) then -- +1 due to 1 skill
        step = step + 1
    end

    -- game over? get next game!
    if not terminal then
        screen, reward, terminal = make_step(game_actions[action_index], opt.socket, agent, inner_action_index > agent.n_actions)
    else
        screen, reward, terminal = make_step(9, opt.socket, agent, false) -- new game
    end

    -- display screen
    --win = image.display({image=screen, win=win})

    if step % opt.prog_freq == 0 then
        assert(step==agent.numSteps, 'trainer step: ' .. step ..
            ' & agent.numSteps: ' .. agent.numSteps)
        print("Steps: ", step)
        agent:report()
        collectgarbage()
    end

    if step%1000 == 0 then collectgarbage() end


    if step % opt.eval_freq == 0 and step > learn_start and step > last_eval then
        last_eval = step
        screen, reward, terminal = make_step(9, opt.socket) -- 9 = new game

        total_reward = 0
        nrewards = 0
        nepisodes = 0
        episode_reward = 0

        local reached_first = false
        local num_reached_first, num_reached_second = 0, 0

        local eval_time = sys.clock()
        local estep=1
        while estep < opt.eval_steps do
            local action_index = agent:perceive(reward, screen, terminal, true, 0) --0.05)

            -- Play game in test mode (episodes don't end when losing a life)
            screen, reward, terminal = make_step(game_actions[action_index], opt.socket)
            estep = estep + 1

            -- display screen
            --win = image.display({image=screen, win=win})

            if estep%1000 == 0 then collectgarbage() end

            -- record every reward
            tmp_reward = reward
            if tonumber(opt.reward_shaping) == 0 and tonumber(reward) ~= 0 then
                tmp_reward = -1
            end
            episode_reward = episode_reward + tmp_reward

            if tonumber(reward) ~= 0 then
                nrewards = nrewards + 1
            end

            if tonumber(reward) == 20 then
                if reached_first then
                    num_reached_second = num_reached_second + 1
                else
                    num_reached_first = num_reached_first + 1
                end
                reached_first = true
            end

            if terminal then
                reached_first = false

                print('Evaluation ended in: ' .. estep .. ' steps, last reward was: ' .. reward)
                total_reward = total_reward + episode_reward
                episode_reward = 0
                nepisodes = nepisodes + 1

                screen, reward, terminal = make_step(9, opt.socket)
                --break
            end
        end

        print("NumReachedFirst: "..num_reached_first.." NumReachedSecond: "..num_reached_second)

        eval_time = sys.clock() - eval_time
        start_time = start_time + eval_time
        agent:compute_validation_statistics()
        local ind = #reward_history+1
        total_reward = total_reward/math.max(1, nepisodes)

        if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
            agent.best_network = agent.network:clone()
        end

        if agent.v_avg then
            v_history[ind] = agent.v_avg
            td_history[ind] = agent.tderr_avg
            qmax_history[ind] = agent.q_max
        end
        print("V", v_history[ind], "TD error", td_history[ind], "Qmax", qmax_history[ind])

        reward_history[ind] = total_reward
        reward_counts[ind] = nrewards
        episode_counts[ind] = nepisodes

        time_history[ind+1] = sys.clock() - start_time

        local time_dif = time_history[ind+1] - time_history[ind]

        local training_rate = opt.actrep*opt.eval_freq/time_dif

        print(string.format(
            '\nSteps: %d (frames: %d), reward: %.2f, epsilon: %.2f, lr: %G, ' ..
            'training time: %ds, training rate: %dfps, testing time: %ds, ' ..
            'testing rate: %dfps,  num. ep.: %d,  num. rewards: %d',
            step, step*opt.actrep, total_reward, agent.ep, agent.lr, time_dif,
            training_rate, eval_time, opt.actrep*opt.eval_steps/eval_time,
            nepisodes, nrewards))
    end

    if step % opt.save_freq == 0 or step == opt.steps then
        local s, a, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_r,
            agent.valid_s2, agent.valid_term
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
        agent.valid_term = nil, nil, nil, nil, nil, nil, nil
        local w, dw, g, g2, delta, delta2, deltas, tmp = agent.w, agent.dw,
            agent.g, agent.g2, agent.delta, agent.delta2, agent.deltas, agent.tmp
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
        agent.deltas, agent.tmp = nil, nil, nil, nil, nil, nil, nil, nil

        local filename = opt.name
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
end
--]]
