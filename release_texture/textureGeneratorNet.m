function net = textureGeneratorNet( net, config )
%Construct the generator network similar with DCGAN paper
% the input to the network should be z_dim * z_dim * 3 * nbatch
% for texture generator network, we dont include fc layer


%% first deconv layer output shape: 8*8*256*nsamp 
opts.weightDecay = 1 ;
opts.scale = 1 ;
opts.weightInitMethod = 'gaussian' ;

opts.batchNormalization = true;
opts.addrelu = true;
layer_name = '1';
num_in = 5; % now consider the 7*7*5
num_out = 64*8; % 64*4
filter_h = 5; %5
filter_w = 5;
upsample = 2; %2, 8, 3
crop= [1,2,1,2];%floor(filter_sz/2); %3
net = add_deconv_block(net, opts, layer_name, filter_h, filter_w, num_in, num_out, upsample, crop);
%% second deconv layer output shape: 16*16*128*nsamp
layer_name = '2';
num_in = 64*8;
num_out = 64*4;
filter_h = 5; %11
filter_w = 5;
upsample = 2; %2, 8, 3
crop= [1,2,1,2];%floor(filter_sz/2); %3
net = add_deconv_block(net, opts, layer_name, filter_h, filter_w, num_in, num_out, upsample, crop);
%% third deconv layer output shape: 32*32*64*nsamp
layer_name = '3';
num_in = 64*4;
num_out = 64*2;
filter_h = 5; %11
filter_w = 5;
upsample = 2; %2, 8, 3
crop= [1,2,1,2];%floor(filter_sz/2); %3
net = add_deconv_block(net, opts, layer_name, filter_h, filter_w, num_in, num_out, upsample, crop);
%% The forth deconv layer output shape: 64*64*3*nsamp
layer_name = '4';
num_in = 64*2;
num_out = 64; % 3
filter_h = 5; %11
filter_w = 5;
upsample = 2; %2, 8, 3
crop= [1,2,1,2];%floor(filter_sz/2); %3
%opts.batchNormalization = false;
%opts.addrelu = false;
net = add_deconv_block(net, opts, layer_name, filter_h, filter_w, num_in, num_out, upsample, crop);

%% The fifth deconv layer output shape 128*128*3
layer_name = '5';
num_in = 64;
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

