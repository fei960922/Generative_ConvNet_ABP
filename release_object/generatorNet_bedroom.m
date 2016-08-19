function net = generatorNet_new( net, config )
%Construct the generator network as in DCGAN paper
% the input to the network should be 1*100*1*nsamp

    opts.weightDecay = 1 ;
    opts.scale = 2 ;
    opts.weightInitMethod = 'gaussian' ;
    opts.batchNormalization = false;
    opts.addrelu = false;
    opts.leak = 0.0;
    learning_rate = 0.9;

    if (~isfield(config, 'fc_number') || config.fc_number==1) 
        net = add_cnn_block(net, opts, '1', 1, config.z_dim, 1, 64*16*4*4, 1, 0, learning_rate);
    elseif (config.fc_number==2)
        net = add_cnn_block(net, opts, '1', 1, config.z_dim, 1, 64*8, 1, 0, learning_rate);
        net = add_cnn_block(net, opts, '2', 1, 1, 64*8, 64*16*4*4, 1, 0, learning_rate);
    elseif (config.fc_number==3)
        net = add_cnn_block(net, opts, '1', 1, config.z_dim, 1, 64*4, 1, 0, learning_rate);
        net = add_cnn_block(net, opts, '2', 1, 1, 64*4, 64*16*2, 1, 0, learning_rate);
        net = add_cnn_block(net, opts, '3', 1, 1, 1024*2, 1024*4*4, 1, 0, learning_rate);
    end
    %% Reshape layer (reshape to 4*4*512*nsamp)+ batchnorm + relu
    net = addCustomReshapeLayer(net, @reshapeForward_big, @reshapeBackward_big);
    num_out = 1024;
    net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%d','1'), ...
            'weights', {{ones(num_out, 1, 'single'), zeros(num_out, 1, 'single')}}, ...
            'learningRate', 1*[2 1], ...
            'weightDecay', [0 0]) ;
    net.layers{end+1} = struct('type', 'relu', 'leak', opts.leak, 'name', sprintf('relu%s','1'));

    %% first deconv layer output shape: 8*8*256*nsamp 
    opts.batchNormalization = true;
    opts.addrelu = true;
    net = add_deconv_block(net, opts, '2', 5, 5, 64*16, 64*8, 2, [1,2,1,2], learning_rate);
    net = add_deconv_block(net, opts, '3', 5, 5, 64*8, 64*4, 2, [1,2,1,2], learning_rate);
    net = add_deconv_block(net, opts, '4', 5, 5, 64*4, 64*2, 2, [1,2,1,2], learning_rate);

    %% The forth deconv layer output shape: 64*64*3*nsamp
    if (isfield(config,'double_output') && config.double_output)
        net = add_deconv_block(net, opts, '5', 5, 5, 64*2, 64, 2, [1,2,1,2], learning_rate);
        opts.batchNormalization = false;
        opts.addrelu = false;
        net = add_deconv_block(net, opts, '6', 5, 5, 64, 3, 2, [1,2,1,2], learning_rate);
    else
        opts.batchNormalization = false;
        opts.addrelu = false;
        net = add_deconv_block(net, opts, '5', 5, 5, 64*2, 3, 2, [1,2,1,2], learning_rate);
    end
    %% the final tanh layer
    net = addCustomTanhLayer(net, @tanhForward, @tanhBackward);

end