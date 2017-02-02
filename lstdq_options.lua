-- no clipping
-- gamma search
-- bias
-- more data
-- mm to mv -> tried on the multiplication after the inverse, works the same..

-- requires & loads
require 'torch'
require 'cutorch'
require 'cunn'
require 'initenv'
require 'hdf5'

local skillFileName = 'SkillsData.h5'
local skillFile = hdf5.open(skillFileName, 'r')
--local numberOfSkills = skillFile:read('numberSkills'):all()

--agent     = torch.load('DQN3_0_1_breakout_FULL_Y.t7')
--network   = agent.model
--f         = network:clone()
--orig_w    = f.modules[11].weight
terminals = skillFile:read('termination_1'):all() --torch.load('data/terminals_300000_breakout.t7')
features  = torch.CudaTensor(terminals:size()[1],512):copy(skillFile:read('activations_1'):all()) --torch.load('data/features_300000_breakout.t7')
actions   = skillFile:read('actions_1'):all() --torch.load('data/actions_300000_breakout.t7')
--rewards   = torch.load('data/rewards_300000_breakout.t7')
rewards   = torch.CudaTensor(terminals:size()[1],1):zero() + 1

--require 'cudnn'
--cudnn.benchmark = true
--cudnn.convert(f, cudnn)


-- constants
local n_actions  = 5
local n_features = 512
local gamma      = 0.99
local epsilon    = 0.000000000001--0.0000000001
local delta      = 10
N                = math.floor(features:size()[1]*0.8)

-- flags for options & defining defaults
local include_biases    = false
local reward_clipping   = false
local include_terminals = true
local method            = 'LSTDQgels' --LSTDQinv, LSTDQgels, LSTDQopt
-- inputs
os_include_biases = os.getenv('include_biases')
if (os_include_biases~=nil) then
  if (os_include_biases=='true') then
    include_biases = true
  elseif (os_include_biases=='false') then
    include_biases = false
  else
    print('Illegal argument for include_biases, must be true or false')
    os.exit()
  end
end
os_gamma = os.getenv('gamma')
if (os_gamma~=nil) then
  gamma = tonumber(os_gamma)
end
os_epsilon = os.getenv('epsilon')
if (os_epsilon~=nil) then
  epsilon = tonumber(os_epsilon)
end
os_delta = os.getenv('delta')
if (os_delta~=nil) then
  delta = tonumber(os_delta)
end
os_reward_clipping = os.getenv('reward_clipping')
if (os_reward_clipping~=nil) then
  if (os_reward_clipping=='true') then
    reward_clipping = true
  elseif (os_reward_clipping=='false') then
    reward_clipping = false
  else
    print('Illegal argument for reward_clipping, must be true or false')
    os.exit()
  end
end
os_include_terminals= os.getenv('include_terminals')
if (os_include_terminals~=nil) then
  if (os_include_terminals=='true') then
    include_terminals = true
  elseif (os_include_terminals=='false') then
    include_terminals = false
  else
    print('Illegal argument for include_terminals, must be true or false')
    os.exit()
  end
end
os_method = os.getenv('method')
if (os_method~=nil) then
  if (os_method~='LSTDQinv' and os_method~='LSTDQgels' and os_method~='LSTDQopt') then
    print('Illegal argument for method, must be LSTDQinv, LSTDQgels, or LSTDQopt')
  else
    method=os_method
  end
end

-- input sanity check
--[[print('biases')
print(include_biases)
print('gamma')
print(gamma)
print('epsilon')
print(epsilon)
print('delta')
print(delta)
print('reward_clipping')
print(reward_clipping)
print('include_terminals')
print(include_terminals)
print('method')
print(method)]]--


-- structures
if (include_biases == true) then
  A = torch.CudaTensor((n_features+1)*n_actions,(n_features+1)*n_actions):zero()
  b = torch.CudaTensor((n_features+1)*n_actions,1):zero()
  if (method=='LSTDQopt') then
    A:add(delta*(torch.eye((1+n_features)*n_actions):cuda()))
  end
else
  A = torch.CudaTensor(n_features*n_actions,n_features*n_actions):zero()
  b = torch.CudaTensor(n_features*n_actions,1):zero()
  if (method=='LSTDQopt') then
    A:add(delta*(torch.eye(n_features*n_actions):cuda()))
  end
end

-- algorithm
-- iterations
i = 0
while i < (N-1) do

  -- monitoring progress
  if i%10000 == 0 then
    print(i)
  end
  i=i+1

  -- creating tuple
  action  = actions[i]
  Naction = actions[i+1]
  --reward  = rewards[i][1]
  reward = 0
  if terminals[i + 1] == 0 then
    reward = -1
  end
  -- reward clipping
  if reward_clipping then
    reward = math.min(reward, -1)
    reward = math.max(reward, 1)
  end
  phi_    = features[i]
  Nphi_   = features[i+1]
  if (include_biases == true) then
    phi     = torch.CudaTensor((n_features+1)*n_actions,1):zero()
    Nphi    = torch.CudaTensor((n_features+1)*n_actions,1):zero()
    phi[{{1+(action-1)*n_features,n_features+(action-1)*n_features},1}] = phi_
    phi[{{n_features*n_actions+action},1}] = 1
  else
    phi     = torch.CudaTensor(n_features*n_actions,1):zero()
    Nphi    = torch.CudaTensor(n_features*n_actions,1):zero()
    phi[{{1+(action-1)*n_features,n_features+(action-1)*n_features},1}] = phi_
  end

  -- structures maintenance
  if (include_terminals or (not include_terminals and terminals[i+1]==0)) then
    if (method=='LSTDQinv' or method=='LSTDQgels') then
      b_ = (phi:mul(reward)):div(N)
      b:add(b_)
      if (terminals[i+1]==0) then --not terminal
        if (include_biases == true) then
          Nphi[{{1+(Naction-1)*n_features,n_features+(Naction-1)*n_features},1}] = Nphi_
          Nphi[{{n_features*n_actions+Naction},1}] = 1
        else
          Nphi[{{1+(Naction-1)*n_features,n_features+(Naction-1)*n_features},1}] = Nphi_
        end
        A_ = (torch.mm(phi, ((phi-gamma*Nphi):transpose(1,2)))):div(N)
        A:add(A_)
      else --terminal
        A_ = (torch.mm(phi, ((phi):transpose(1,2)))):div(N)
        A:add(A_)
        i = i+1
      end
    elseif (method=='LSTDQopt') then
      b_ = (reward*phi)
      b:add(b_)
      if (terminals[i+1]==0) then --not terminal
        if (include_biases == true) then
          Nphi[{{1+(Naction-1)*n_features,n_features+(Naction-1)*n_features},1}] = Nphi_
          Nphi[{{n_features*n_actions+Naction},1}] = 1
        else
          Nphi[{{1+(Naction-1)*n_features,n_features+(Naction-1)*n_features},1}] = Nphi_
        end
        A_ = (phi - gamma*Nphi):transpose(1,2)
      else --terminal
        A_ = (phi):transpose(1,2)
        i = i+1
      end
      Acomp = torch.mm(A_, A)
      Aup = torch.mm(A, torch.mm(phi, Acomp))
      Adown = (torch.mm(Acomp, phi)):add(1)
      A:add(-Aup/Adown[1][1])
    else
      print('Got illegal method!!')
      os.exit()
    end
  else
    i = i+1
  end
end


-- garbage collection
--f         = nil
--features  = nil
--actions   = nil
rewards   = nil
--terminals = nil
collectgarbage()

-- solving equations
if (method=='LSTDQinv') then
  if (include_biases == true) then
    A:add(epsilon*(torch.eye((1+n_features)*n_actions):cuda())) --stability
  else
    A:add(epsilon*(torch.eye(n_features*n_actions):cuda())) --stability
  end
  Ainv = torch.inverse(A)
  w = torch.mm(Ainv,b):cuda()
elseif (method=='LSTDQgels') then
  if (include_biases == true) then
    A:add(epsilon*(torch.eye((1+n_features)*n_actions):cuda())) --stability
  else
    A:add(epsilon*(torch.eye(n_features*n_actions):cuda())) --stability
  end
  w = torch.gels(b:double(), A:double()):cuda()
elseif (method=='LSTDQopt') then
  w = torch.mm(A,b)
else
  print('Got illegal method!!')
  os.exit()
end

-- extracting weights & biases
w_copy  = w:clone()
if (include_biases == true) then
  weights = ((w_copy:narrow(1,1,n_features*n_actions)):reshape(n_actions,n_features))
  biases  = ((w_copy:narrow(1,n_features*n_actions+1,n_actions)):reshape(n_actions))
else
  weights = ((w_copy:narrow(1,1,n_features*n_actions)):reshape(n_actions,n_features))
end

-- sanity checks
--[[print('max-min')
print(A:max())
print(A:min())
print('sanity checks')
print(biases)
print(weights:max())
print(weights:min())--]]

-- evaluation
error = 0.0
counter = 0
print('Calculating error')
features_ = torch.CudaTensor(2,512):zero()
features_[{1, {1,512}}] = features[1]

for i = math.floor(features:size()[1]*0.8) + 1, features:size()[1] do
  if terminals[i] == 0 then
    features_ = torch.CudaTensor(2,512):zero()
    features_[{1, {1,512}}] = features[i]
    y = torch.mm(weights, features_:t())[{{1,5}, 1}]
    if (include_biases == true) then
      y = y + biases
    end

    local maxq = y[1]
    local besta = {1}

    -- Evaluate all other actions (with random tie-breaking)
    for a = 2, n_actions do
        if y[a] > maxq then
            besta = { a }
            maxq = y[a]
        elseif y[a] == maxq then
            besta[#besta+1] = a
        end
    end

    local r = torch.random(1, #besta)

    if besta ~= actions[i] then
      error = error + 1.0
    end
    counter = counter + 1
  end
end
print(error)
print(counter)
print('Error is: ' .. (error / counter))

-- saving weights & biases
--torch.save('data/weights.t7', weights)
--if (include_biases == true) then
--  torch.save('data/biases.t7', biases)
--end
