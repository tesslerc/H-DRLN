--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]
dqn = {}

require 'torch'
require 'nn'
require 'nngraph'
require 'nnutils'
require 'image'
require 'Scale'
require 'NeuralQLearner'
require 'TransitionTable'
require 'Rectifier'


function torchSetup(_opt)
    _opt = _opt or {}
    local opt = table.copy(_opt)
    assert(opt)

    -- preprocess options:
    --- convert options strings to tables
    if opt.pool_frms then
        opt.pool_frms = str_to_table(opt.pool_frms)
    end
    if opt.env_params then
        opt.env_params = str_to_table(opt.env_params)
    end
    if opt.agent_params then
        opt.agent_params = str_to_table(opt.agent_params)
        opt.agent_params.gpu       = opt.gpu
        opt.agent_params.best      = opt.best
        opt.agent_params.verbose   = opt.verbose
        if opt.network ~= '' then
            opt.agent_params.network = opt.network
        end
    end

    --- general setup
    opt.tensorType =  opt.tensorType or 'torch.FloatTensor'
    torch.setdefaulttensortype(opt.tensorType)
    if not opt.threads then
        opt.threads = 4
    end
    torch.setnumthreads(opt.threads)
    if not opt.verbose then
        opt.verbose = 10
    end
    if opt.verbose >= 1 then
        print('Torch Threads:', torch.getnumthreads())
    end

    --- set gpu device
    if opt.gpu and opt.gpu >= 0 then
        require 'cutorch'
        require 'cunn'
        if opt.gpu == 0 then
            local gpu_id = tonumber(os.getenv('GPU_ID'))
            if gpu_id then opt.gpu = gpu_id+1 end
        end
        if opt.gpu > 0 then cutorch.setDevice(opt.gpu) end
        opt.gpu = cutorch.getDevice()
        print('Using GPU device id:', opt.gpu-1)
    else
        opt.gpu = -1
        if opt.verbose >= 1 then
            print('Using CPU code only. GPU device id:', opt.gpu)
        end
    end

    --- set up random number generators
    -- removing lua RNG; seeding torch RNG with opt.seed and setting cutorch
    -- RNG seed to the first uniform random int32 from the previous RNG;
    -- this is preferred because using the same seed for both generators
    -- may introduce correlations; we assume that both torch RNGs ensure
    -- adequate dispersion for different seeds.
    math.random = nil
    opt.seed = opt.seed or 1
    torch.manualSeed(opt.seed)
    if opt.verbose >= 1 then
        print('Torch Seed:', torch.initialSeed())
    end
    local firstRandInt = torch.random()
    if opt.gpu >= 0 then
        cutorch.manualSeed(firstRandInt)
        if opt.verbose >= 1 then
            print('CUTorch Seed:', cutorch.initialSeed())
        end
    end

    return opt
end


function setup(_opt)
    assert(_opt)

    --preprocess options:
    --- convert options strings to tables
    _opt.pool_frms = str_to_table(_opt.pool_frms)
    _opt.env_params = str_to_table(_opt.env_params)
    _opt.agent_params = str_to_table(_opt.agent_params)
    _opt.navigate_agent_params = str_to_table(_opt.skill_agent_params)
    _opt.pickup_agent_params = str_to_table(_opt.skill_agent_params)
    _opt.break_agent_params = str_to_table(_opt.skill_agent_params)
    _opt.place_agent_params = str_to_table(_opt.skill_agent_params)
    _opt.distilled_agent_params = str_to_table(_opt.skill_agent_params)
    if _opt.agent_params.transition_params then
        _opt.navigate_agent_params.transition_params =
            str_to_table(_opt.agent_params.transition_params)
        _opt.pickup_agent_params.transition_params =
            str_to_table(_opt.agent_params.transition_params)
        _opt.break_agent_params.transition_params =
            str_to_table(_opt.agent_params.transition_params)
        _opt.place_agent_params.transition_params =
            str_to_table(_opt.agent_params.transition_params)
        _opt.distilled_agent_params.transition_params =
            str_to_table(_opt.agent_params.transition_params)

        _opt.agent_params.transition_params =
            str_to_table(_opt.agent_params.transition_params)
    end

    --- first things first
    local opt = torchSetup(_opt)

    -- load training framework and environment
    --local framework = require(opt.framework)
    --assert(framework)

    --local gameEnv = framework.GameEnvironment(opt)
    --local gameActions = gameEnv:getActions()
    local gameEnv = nil
    local MCgameActions = {1,3,4,5,0,6,7,8,9}--{1,3,4,5,0,6,7,8,9,1,3,4,1,3,4,1,3,4,5,1,3,4,0} -- this is our game actions table
    local MCgameActions_primitive = {1,3,4,0,5} -- this is our game actions table
    local optionsActions = {6,7,8,9} -- these actions are correlated to an OPTION, 20 = solve room (make this struct with max iterations per option and socket port and ip)

    local controlActions = {1,3,4,5,0,6,7,8,9} -- total is 9 -- this is our game actions table
    local navigateActions = {1,3,4} -- total is 3
    local pickupActions = {1,3,4} -- total is 3
    local breakActions = {1,3,4,5} -- total is 4
    local placeActions = {1,3,4,0} -- total is 4

    -- agent options
    _opt.agent_params.actions   = MCgameActions
    _opt.agent_params.options	= optionsActions
    _opt.agent_params.optionsActions = {navigateActions, pickupActions, breakActions, placeActions}
    _opt.agent_params.gpu       = _opt.gpu
    _opt.agent_params.best      = _opt.best
    _opt.agent_params.distilled_network = true
    _opt.agent_params.distill   = false
    if _opt.agent_params.network then
	print(_opt.agent_params.network)
	_opt.agent_params.network = "convnet_atari_main"
    end
    --_opt.agent_params.network = "convnet_atari3"
    if _opt.network ~= '' then
        _opt.agent_params.network = _opt.network
    end

    _opt.agent_params.verbose = _opt.verbose
    if not _opt.agent_params.state_dim then
        _opt.agent_params.state_dim = gameEnv:nObsFeature()
    end
--
    _opt.distilled_agent_params.actions   = navigateActions
    _opt.distilled_agent_params.gpu       = _opt.gpu
    _opt.distilled_agent_params.best      = _opt.best
    _opt.distilled_agent_params.distilled_network = false
    _opt.distilled_agent_params.distill   = false
    if _opt.distilled_network ~= '' then
        _opt.distilled_agent_params.network = _opt.distilled_network
    end
    _opt.distilled_agent_params.verbose = _opt.verbose
    if not _opt.distilled_agent_params.state_dim then
        _opt.distilled_agent_params.state_dim = gameEnv:nObsFeature()
    end
    print("DISTILLED NETWORK")
    local distilled_agent = dqn[_opt.agent](_opt.distilled_agent_params)
    print("DISTILLED NETWORK")
--]]
--[[
    _opt.navigate_agent_params.actions   = navigateActions
    _opt.navigate_agent_params.gpu       = _opt.gpu
    _opt.navigate_agent_params.best      = _opt.best
    _opt.navigate_agent_params.distilled_network = false
    _opt.navigate_agent_params.distill   = false
    if _opt.navigate_network ~= '' then
        _opt.navigate_agent_params.network = _opt.navigate_network
    end
    _opt.navigate_agent_params.verbose = _opt.verbose
    if not _opt.navigate_agent_params.state_dim then
        _opt.navigate_agent_params.state_dim = gameEnv:nObsFeature()
    end
	  local navigate_agent = dqn[_opt.agent](_opt.navigate_agent_params)

    _opt.pickup_agent_params.actions   = pickupActions
    _opt.pickup_agent_params.gpu       = _opt.gpu
    _opt.pickup_agent_params.best      = _opt.best
    _opt.pickup_agent_params.distilled_network = false
    _opt.pickup_agent_params.distill   = false
    if _opt.pickup_network ~= '' then
        _opt.pickup_agent_params.network = _opt.pickup_network
    end
    _opt.pickup_agent_params.verbose = _opt.verbose
    if not _opt.pickup_agent_params.state_dim then
        _opt.pickup_agent_params.state_dim = gameEnv:nObsFeature()
    end
	  local pickup_agent = dqn[_opt.agent](_opt.pickup_agent_params)

    _opt.break_agent_params.actions   = breakActions
    _opt.break_agent_params.gpu       = _opt.gpu
    _opt.break_agent_params.best      = _opt.best
    _opt.break_agent_params.distilled_network = false
    _opt.break_agent_params.distill   = false
    if _opt.break_network ~= '' then
        _opt.break_agent_params.network = _opt.break_network
    end
    _opt.break_agent_params.verbose = _opt.verbose
    if not _opt.break_agent_params.state_dim then
        _opt.break_agent_params.state_dim = gameEnv:nObsFeature()
    end
	  local break_agent = dqn[_opt.agent](_opt.break_agent_params)


    _opt.place_agent_params.actions   = placeActions
    _opt.place_agent_params.gpu       = _opt.gpu
    _opt.place_agent_params.best      = _opt.best
    _opt.place_agent_params.distilled_network = false
    _opt.place_agent_params.distill   = false
    if _opt.place_network ~= '' then
        _opt.place_agent_params.network = _opt.place_network
    end
    _opt.place_agent_params.verbose = _opt.verbose
    if not _opt.place_agent_params.state_dim then
        _opt.place_agent_params.state_dim = gameEnv:nObsFeature()
    end
	  local place_agent = dqn[_opt.agent](_opt.place_agent_params)

--]]
    -- create controller agent instance with its own sub networks
    --[[
    local _place_agent = dqn[_opt.agent](_opt.place_agent_params)
    local _break_agent = dqn[_opt.agent](_opt.break_agent_params)
    local _navigate_agent = dqn[_opt.agent](_opt.navigate_agent_params)
    local _pickup_agent = dqn[_opt.agent](_opt.pickup_agent_params)
    _opt.control_agent_params.actions   = controlActions
    _opt.control_agent_params.gpu       = _opt.gpu
    _opt.control_agent_params.best      = _opt.best
    _opt.control_agent_params.distilled_network = false
    _opt.control_agent_params.distill   = false
    if _opt.control_network ~= '' then
        _opt.control_agent_params.network = _opt.control_network
    end
    _opt.control_agent_params.verbose = _opt.verbose
    if not _opt.control_agent_params.state_dim then
        _opt.control_agent_params.state_dim = gameEnv:nObsFeature()
    end
    _opt.control_agent_params.skill_agent = {_navigate_agent, _pickup_agent, _break_agent, _place_agent} -- set skill_agent inside so we can access from inside agent (inside the perceive)
    _opt.control_agent_params.primitive_actions = MCgameActions_primitive
    local control_agent = dqn[_opt.agent](_opt.control_agent_params) -- control agent instanciation
    --]]

    _opt.agent_params.skill_agent = distilled_agent--{navigate_agent, pickup_agent, break_agent, place_agent} -- set skill_agent inside so we can access from inside agent (inside the perceive)
    --_opt.agent_params.skill_agent = {navigate_agent, pickup_agent, break_agent, place_agent} -- set skill_agent inside
    _opt.agent_params.primitive_actions = MCgameActions_primitive
    print("MAIN AGENT")
    local agent = dqn[_opt.agent](_opt.agent_params)
    print("MAIN AGENT")
    if opt.verbose >= 1 then
        print('Set up Torch using these options:')
        for k, v in pairs(opt) do
            print(k, v)
        end
    end

    return gameEnv, MCgameActions_primitive, agent, opt
end



--- other functions

function str_to_table(str)
    if type(str) == 'table' then
        return str
    end
    if not str or type(str) ~= 'string' then
        if type(str) == 'table' then
            return str
        end
        return {}
    end
    local ttr
    if str ~= '' then
        local ttx=tt
        loadstring('tt = {' .. str .. '}')()
        ttr = tt
        tt = ttx
    else
        ttr = {}
    end
    return ttr
end

function table.copy(t)
    if t == nil then return nil end
    local nt = {}
    for k, v in pairs(t) do
        if type(v) == 'table' then
            nt[k] = table.copy(v)
        else
            nt[k] = v
        end
    end
    setmetatable(nt, table.copy(getmetatable(t)))
    return nt
end
