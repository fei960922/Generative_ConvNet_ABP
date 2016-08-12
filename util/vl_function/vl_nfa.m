function res = vl_nfa(net, x, dzdy, res, varargin)
% based on the slightly modification of vl_simplenn
% More faster than directly use the vl_simplenn in the algorithm
% here, x actually represents the latent factor z. (when we do feedforward,
% we start from latent z and get reconstructed image)
% dzdy represents the im_batch
opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.backPropDepth = +inf ;

opts = vl_argparse(opts, varargin);

n = numel(net.layers) ;

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
else
  doder = true ;
end

if opts.cudnn
  cudnn = {'CuDNN'} ;
else
  cudnn = {'NoCuDNN'} ;
end

gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;
end
res(1).x = x ;

for i=1:n
  l = net.layers{i} ;
  res(i).time = tic ;
  switch l.type
    case 'conv'
      if isfield(l, 'weights')
        res(i+1).x = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                               'pad', l.pad, 'stride', l.stride, ...
                               cudnn{:}) ;
      else
        res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, ...
                               'pad', l.pad, 'stride', l.stride, ...
                               cudnn{:}) ;
      end
    case 'convt'
      if isfield(l, 'weights')
        res(i+1).x = vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
                               'crop', l.crop, 'upsample', l.upsample, ...
                               'numGroups', l.numGroups, cudnn{:}) ;
      else
        res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, ...
                               'crop', l.pad, 'upsample', l.upsample, ...
                               'numGroups', l.numGroups, cudnn{:}) ;
      end
    case 'pool'
      res(i+1).x = vl_nnpool(res(i).x, l.pool, ...
                             'pad', l.pad, 'stride', l.stride, ...
                             'method', l.method, ...
                             cudnn{:}) ;
    case 'normalize'
      res(i+1).x = vl_nnnormalize(res(i).x, l.param) ;
    case 'relu'
      if isfield(l, 'leak'), leak = {'leak', l.leak} ; else leak = {} ; end
      res(i+1).x = vl_nnrelu(res(i).x,[],leak{:}) ;
    case 'bnorm'
      if isfield(l, 'weights')
        res(i+1).x = vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}) ;
      else
        res(i+1).x = vl_nnbnorm(res(i).x, l.filters, l.biases) ;
      end
    case 'custom'
      res(i+1) = l.forward(l, res(i), res(i+1)) ;
    otherwise
      error('Unknown layer type %s', l.type) ;
  end
  % optionally forget intermediate results
  forget = opts.conserveMemory ;
  forget = forget & (~doder || strcmp(l.type, 'relu')) ;
  forget = forget & (~isfield(l, 'rememberOutput') || ~l.rememberOutput) ;
  if forget
    res(i).x = [] ;
  end
  if gpuMode & opts.sync
    % This should make things slower, but on MATLAB 2014a it is necessary
    % for any decent performance.
    wait(gpuDevice) ;
  end
  res(i).time = toc(res(i).time) ;
end


if doder
  %res(n+1).dzdx = dzdy ;
  dzdy = dzdy - res(end).x;
  res(n+1).dzdx = dzdy;
  for i=n:-1:max(1, n-opts.backPropDepth+1)
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    switch l.type
      case 'conv'
        if ~opts.accumulate        
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                          res(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
        else
          dzdw = cell(1,2) ;      
            [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                          res(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;         
          for j=1:2
            res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
          end
          clear dzdw ;
        end

      case 'convt'
        if ~opts.accumulate          
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
                          res(i+1).dzdx, ...
                          'crop', l.crop, 'upsample', l.upsample, ...
                          'numGroups', l.numGroups, cudnn{:}) ;         
        else
          dzdw = cell(1,2) ;     
            [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
                          res(i+1).dzdx, ...
                          'crop', l.crop, 'upsample', l.upsample, ...
                          'numGroups', l.numGroups, cudnn{:}) ;          
          for j=1:2
            res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
          end
          clear dzdw ;
        end

      case 'pool'
        res(i).dzdx = vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
                                'pad', l.pad, 'stride', l.stride, ...
                                'method', l.method, ...
                                cudnn{:}) ;
      case 'normalize'
        res(i).dzdx = vl_nnnormalize(res(i).x, l.param, res(i+1).dzdx) ;
      case 'relu'
        if isfield(l, 'leak'), leak = {'leak', l.leak} ; else leak = {} ; end
        if ~isempty(res(i).x)
          res(i).dzdx = vl_nnrelu(res(i).x, res(i+1).dzdx, leak{:}) ;
        else
          % if res(i).x is empty, it has been optimized away, so we use this
          % hack (which works only for ReLU):
          res(i).dzdx = vl_nnrelu(res(i+1).x, res(i+1).dzdx, leak{:}) ;
        end
 
      case 'bnorm'
        if ~opts.accumulate
          if isfield(l, 'weights')
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
                           res(i+1).dzdx) ;
          else
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnbnorm(res(i).x, l.filters, l.biases, ...
                           res(i+1).dzdx) ;
          end
        else
          dzdw = cell(1,2) ;
          if isfield(l, 'weights')
            [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
                           res(i+1).dzdx) ;
          else
            [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                vl_nnbnorm(res(i).x, l.filters, l.biases, ...
                           res(i+1).dzdx) ;
          end
          for j=1:2
            res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
          end
          clear dzdw ;
        end
      
      case 'custom'
        res(i) = l.backward(l, res(i), res(i+1)) ;
    end
    if opts.conserveMemory
      res(i+1).dzdx = [] ;
    end
    if gpuMode & opts.sync
      wait(gpuDevice) ;
    end
    res(i).backwardTime = toc(res(i).backwardTime) ;
  end
end

