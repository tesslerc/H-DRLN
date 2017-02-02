--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "initenv"

function create_network(args)

    local net = nn.Sequential()
    net:add(nn.Reshape(unpack(args.input_dims)))

    --- first convolutional layer
    local convLayer = nn.SpatialConvolution

    net:add(convLayer(args.hist_len*args.ncols, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],1))
    net:add(args.nl())

    -- Add convolutional layers
    for i=1,(#args.n_units-1) do
        -- second convolutional layer
        net:add(convLayer(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1]))
        net:add(args.nl())
    end

    local nel
    if args.gpu >= 0 then
        nel = net:cuda():forward(torch.zeros(1,unpack(args.input_dims))
                :cuda()):nElement()
    else
        nel = net:forward(torch.zeros(1,unpack(args.input_dims))):nElement()
    end

    -- reshape all feature planes into a vector per example
    net:add(nn.Reshape(nel))

    -- fully connected layer
    net:add(nn.Linear(nel, args.n_hid[1]))
    net:add(args.nl())
    local last_layer_size = args.n_hid[1]

    for i=1,(#args.n_hid-1) do
        -- add Linear layer
        last_layer_size = args.n_hid[i+1]
        net:add(nn.Linear(args.n_hid[i], last_layer_size))
        net:add(args.nl())
    end

    if args.distilled_network == true and false then
    -- add the last fully connected layer (to actions)
	    concat = nn.Concat(2)


	    local MCgameActions = {1,3,4,5,0,6,7,8,9} --,5,6,7,8} -- this is our game actions table
	    local MCgameActions_primitive = {1,3,4,0,5} --,5} -- this is our game actions table
	    local optionsActions = {6,7,8,9} -- these actions are correlated to an OPTION, 20 = solve room (make this struct with max iterations per option and socket port and ip)

	    local navigateActions = {1,3,4}
	    local pickupActions = {1,3,4}
	    local breakActions = {1,3,4,5}
	    local placeActions = {1,3,4,0}


	    --args.skills = {navigateActions, pickupActions, breakActions, placeActions}

	    for i=1,#args.skills do
		print('Added new skill layer '..i..' with '..(args.skills[i])..' actions')
		skill = nn.Sequential()
		skill:add(nn.Linear(last_layer_size, (args.skills[i])))
		concat:add(skill)
	    end

	    net:add(concat)
    else
    -- add the last fully connected layer (to actions)
    	    net:add(nn.Linear(last_layer_size, args.n_actions))
    end

    if args.gpu >=0 then
        net:cuda()
    end
    if args.verbose >= 2 then
        print(net)
        print('Convolutional layers flattened output size:', nel)
    end
    return net
end
