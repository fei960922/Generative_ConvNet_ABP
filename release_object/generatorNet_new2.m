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

net = add_cnn_block(net, opts, '1', 1, config.z_dim, 1, 64*8, 1, 0, learning_rate);
net = add_cnn_block(net, opts, '2', 1, 1, 64*8, 64*8*4*4, 1, 0, learning_rate);
net = addCustomReshapeLayer(net, @reshapeForward, @reshapeBackward);
num_out = 512;
net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%d','1'), ...
        'weights', {{ones(num_out, 1, 'single'), zeros(num_out, 1, 'single')}}, ...
        'learningRate', 1*[2 1], ...
        'weightDecay', [0 0]) ;
net.layers{end+1} = struct('type', 'relu', 'leak', opts.leak, 'name', sprintf('relu%s','1'));
%% first deconv layer output shape: 8*8*256*nsamp 
opts.batchNormalization = true;
opts.addrelu = true;

net = add_deconv_block(net, opts, '2', 5, 5, 64*8, 64*4, 3, [1,2,1,2], learning_rate);
net = add_cnn_block(net, opts, '2c', 2, 2, 64*4, 64*4, 1, 0, learning_rate);
net = add_deconv_block(net, opts, '3', 5, 5, 64*4, 64*2, 2, [1,2,1,2], learning_rate);
net = add_cnn_block(net, opts, '3c', 3, 3, 64*2, 64*2, 1, 0, learning_rate);
net = add_deconv_block(net, opts, '4', 5, 5, 64*2, 64*1, 2, [1,2,1,2], learning_rate);
net = add_cnn_block(net, opts, '4c', 4, 4, 64*1, 64*1, 1, 0, learning_rate);
opts.batchNormalization = false;
opts.addrelu = false;
net = add_deconv_block(net, opts, '5', 5, 5, 64, 3, 2, [1,2,1,2], learning_rate);
net = add_cnn_block(net, opts, '5c', 3, 3, 3, 3, 1, 0, learning_rate);

%% the final tanh layer
net = addCustomTanhLayer(net, @tanhForward, @tanhBackward);

end