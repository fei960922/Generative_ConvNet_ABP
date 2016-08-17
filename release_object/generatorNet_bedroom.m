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
	net = add_cnn_block(net, opts, '1', 1, config.z_dim, 1, 64*8*4*4, 1, 0, learning_rate);
else 
    if (config.fc_number==2)
		%% fc-1 layer 
		net = add_cnn_block(net, opts, '1', 1, config.z_dim, 1, 64*8, 1, 0, learning_rate);
		%% fc-2 layer
		net = add_cnn_block(net, opts, '2', 1, 1, 64*8, 64*8*4*4, 1, 0, learning_rate);
    else
        if (config.fc_number==3)
            net = add_cnn_block(net, opts, '1', 1, config.z_dim, 1, 64*4, 1, 0, learning_rate);
	        net = add_cnn_block(net, opts, '2', 1, 1, 64*4, 64*16*2, 1, 0, learning_rate);
	        net = add_cnn_block(net, opts, '3', 1, 1, 1024*2, 1024*4*4, 1, 0, learning_rate);
        end
    end
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
layer_name = '2';
num_in = 64*16;
num_out = 64*8;
filter_h = 5; %5
filter_w = 5;
upsample = 2; %2, 8, 3
crop= [1,2,1,2];%floor(filter_sz/2); %1 2 1 2
net = add_deconv_block(net, opts, layer_name, filter_h, filter_w, num_in, num_out, upsample, crop, learning_rate);
if (isfield(config,'add_conv_behind') && config.add_conv_behind)
    net = add_cnn_block(net, opts, '2c', 3, 3, num_out, num_out, 1, 0, learning_rate);
end

%% second deconv layer output shape: 16*16*128*nsamp
layer_name = '3';
num_in = 64*8;
num_out = 64*4;
filter_h = 5; %11
filter_w = 5;
upsample = 2; %2, 8, 3
crop= [1,2,1,2];%floor(filter_sz/2); %3
net = add_deconv_block(net, opts, layer_name, filter_h, filter_w, num_in, num_out, upsample, crop, learning_rate);
if (isfield(config,'add_conv_behind') && config.add_conv_behind)
    net = add_cnn_block(net, opts, '3c', 3, 3, num_out, num_out, 1, 0, learning_rate);
end

%% third deconv layer output shape: 32*32*64*nsamp
layer_name = '4';
num_in = 64*4;
num_out = 64*2;
filter_h = 5; %11
filter_w = 5;
upsample = 2; %2, 8, 3
crop= [1,2,1,2];%floor(filter_sz/2); %3
net = add_deconv_block(net, opts, layer_name, filter_h, filter_w, num_in, num_out, upsample, crop, learning_rate);
if (isfield(config,'add_conv_behind') && config.add_conv_behind)
    net = add_cnn_block(net, opts, '4c', 3, 3, num_out, num_out, 1, 0, learning_rate);
    layer_name = '4.5';
    num_in = 64*1;
    num_out = 32;
    filter_h = 5; %11
    filter_w = 5;
    upsample = 2; %2, 8, 3
    crop= [1,2,1,2];%floor(filter_sz/2); %3
    net = add_deconv_block(net, opts, layer_name, filter_h, filter_w, num_in, num_out, upsample, crop, learning_rate);
    net = add_cnn_block(net, opts, '4.c5', 4, 4, num_out, num_out, 1, 0, learning_rate);
end

%% The forth deconv layer output shape: 64*64*3*nsamp
layer_name = '5';
num_in = 64*2;
if  (isfield(config,'add_conv_behind') && config.add_conv_behind)
    num_in = 32;
end
num_out = 3;
filter_h = 5; %11
filter_w = 5;
upsample = 2; %2, 8, 3
crop= [1,2,1,2];%floor(filter_sz/2); %3
opts.batchNormalization = false;
opts.addrelu = false;
net = add_deconv_block(net, opts, layer_name, filter_h, filter_w, num_in, num_out, upsample, crop, learning_rate);
if (isfield(config,'add_conv_behind') && config.add_conv_behind)
    net = add_cnn_block(net, opts, '5c', 3, 3, num_out, num_out, 1, 0, learning_rate);
end

%% the final tanh layer
net = addCustomTanhLayer(net, @tanhForward, @tanhBackward);

end