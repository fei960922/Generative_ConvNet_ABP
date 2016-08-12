function net = add_deconv_block(net, opts, id, h, w, in, out, upsample, crop, learning_rate)
% --------------------------------------------------------------------
% deconv2d + batchnorm+ lrelu
if nargin < 10
    learning_rate = 1;
end


net.layers{end+1} = struct('type', 'convt', 'name', sprintf('deconv%s', id), ...
    'weights', {{gpuArray(init_weight(opts, h, w, in, out, 'single')), gpuArray(zeros(1,1,  out, 'single'))}}, ...
    'upsample', [upsample,upsample], ...
    'crop', crop, ...
    'numGroups', 1, ...
    'learningRate', learning_rate*[1, 2], ...
    'weightDecay', [opts.weightDecay 0]) ;
if opts.batchNormalization
    net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%d',id), ...
        'weights', {{gpuArray(ones(out, 1, 'single')), gpuArray(zeros(out, 1, 'single'))}}, ...
        'learningRate', learning_rate*[2 1], ...
        'weightDecay', [0 0]) ;
end
if opts.addrelu
    net.layers{end+1} = struct('type', 'relu', 'leak', opts.leak, 'name', sprintf('relu%s',id));
end
end


function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.
% update tian: only change the gaussian initializer

switch lower(opts.weightInitMethod)
    case 'gaussian'
        sc = 0.01/opts.scale ;
        
        weights = randn(h, w, out, in, type)*sc; % for test, change to 0
        %weights = zeros(h, w, out, in, type)*sc;
    case 'xavier'
        sc = sqrt(3/(h*w*in)) ;
        weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
    case 'xavierimproved'
        sc = sqrt(2/(h*w*out)) ;
        weights = randn(h, w, in, out, type)*sc ;
    case 'zero'
        weights = zeros(h, w, in, out, type);
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end
end