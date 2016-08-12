function net = generatorNet( net, config )
%Construct the generator network as in DCGAN paper
% the input to the network should be 1*100*1*nsamp
%% fc layer 
opts.weightDecay = 1 ;
opts.scale = 2 ;
opts.weightInitMethod = 'gaussian' ;
opts.batchNormalization = false;
opts.addrelu = false;
opts.leak = 0.0;
layer_name = '1';
num_in = 1;
num_out = 64*8*4*4;
filter_h = 1; 
filter_w = config.z_dim;
stride = 1; %2, 8, 3
pad_sz = 0;%floor(filter_sz/2); %3
pad = ones(1,4)*pad_sz;
net = add_cnn_block(net, opts, layer_name, filter_h, filter_w, num_in, num_out, stride, pad);
%% Reshape layer (reshape to 4*4*512*nsamp)+ batchnorm + relu
net = addCustomReshapeLayer(net, @reshapeForward, @reshapeBackward);
num_out = 512;
net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%d',layer_name), ...
        'weights', {{ones(num_out, 1, 'single'), zeros(num_out, 1, 'single')}}, ...
        'learningRate', 1*[2 1], ...
        'weightDecay', [0 0]) ;

net.layers{end+1} = struct('type', 'relu', 'leak', opts.leak, 'name', sprintf('relu%s',layer_name));
%% first deconv layer output shape: 8*8*256*nsamp 
opts.batchNormalization = true;
opts.addrelu = true;
layer_name = '2';
num_in = 64*8;
num_out = 64*4;
filter_h = 5; %5
filter_w = 5;
upsample = 2; %2, 8, 3
crop= [1,2,1,2];%floor(filter_sz/2); %1 2 1 2
net = add_deconv_block(net, opts, layer_name, filter_h, filter_w, num_in, num_out, upsample, crop);
%% second deconv layer output shape: 16*16*128*nsamp
layer_name = '3';
num_in = 64*4;
num_out = 64*2;
filter_h = 5; %11
filter_w = 5;
upsample = 2; %2, 8, 3
crop= [1,2,1,2];%floor(filter_sz/2); %3
net = add_deconv_block(net, opts, layer_name, filter_h, filter_w, num_in, num_out, upsample, crop);
%% third deconv layer output shape: 32*32*64*nsamp
layer_name = '4';
num_in = 64*2;
num_out = 64*1;
filter_h = 5; %11
filter_w = 5;
upsample = 2; %2, 8, 3
crop= [1,2,1,2];%floor(filter_sz/2); %3
net = add_deconv_block(net, opts, layer_name, filter_h, filter_w, num_in, num_out, upsample, crop);
%% The forth deconv layer output shape: 64*64*3*nsamp
layer_name = '5';
num_in = 64*1;
num_out = 3;
filter_h = 5; %11
filter_w = 5;
upsample = 2; %2, 8, 3
crop= [1,2,1,2];%floor(filter_sz/2); %3
opts.batchNormalization = false;
opts.addrelu = false;
net = add_deconv_block(net, opts, layer_name, filter_h, filter_w, num_in, num_out, upsample, crop);
%% the final tanh layer
net = addCustomTanhLayer(net, @tanhForward, @tanhBackward);

end

