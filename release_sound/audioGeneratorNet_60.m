function net = audioGeneratorNet_60( net, config )
%Construct the generator network similar with DCGAN paper
% the input to the network should be 1* z_dim * 1 * nbatch
% for audio generator network, we dont include fc layer (similar to 2D
% texture)

% the final output size is 1*60000*1
% start from z_dim = 6, then upsampling by 10 (4 layers)

%% first deconv layer output shape: 8*8*256*nsamp 
opts.weightDecay = 1 ;
opts.scale = 2 ;
opts.weightInitMethod = 'gaussian' ;

opts.batchNormalization = true;
opts.addrelu = true;
opts.leak = 0;
layer_name = '1';
num_in = 1;
num_out = 64*4; % 64*4
filter_h = 1; %5
filter_w = 25;
upsample = 10; %2, 8, 3
crop= [0,0,7,8];%floor(filter_sz/2); %3
net = add_deconv_block(net, opts, layer_name, filter_h, filter_w, num_in, num_out, upsample, crop);
%% second deconv layer output shape: 16*16*128*nsamp
layer_name = '2';
num_in = 64*4;
num_out = 64*2;
filter_h = 1; %11
filter_w = 25;
upsample = 10; %2, 8, 3
crop= [0,0,7,8];%floor(filter_sz/2); %3
net = add_deconv_block(net, opts, layer_name, filter_h, filter_w, num_in, num_out, upsample, crop);
%% third deconv layer output shape: 32*32*64*nsamp
layer_name = '3';
num_in = 64*2;
num_out = 1;
filter_h = 1; %11
filter_w = 25;
upsample = 10; %2, 8, 3
crop= [0,0,7,8];%floor(filter_sz/2); %3
opts.batchNormalization = false;
opts.addrelu = false;
net = add_deconv_block(net, opts, layer_name, filter_h, filter_w, num_in, num_out, upsample, crop);


%net = add_deconv_block(net, opts, layer_name, filter_h, filter_w, num_in, num_out, upsample, crop);


%% the final tanh layer
net = addCustomTanhLayer(net, @tanhForward, @tanhBackward);

end

