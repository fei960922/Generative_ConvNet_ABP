function  [net_cpu, syn_mats] = process_epoch_nfa(opts, getBatch, epoch, subset, learningRate, imdb, net_cpu, syn_mats, config)
% -------------------------------------------------------------------------
% updating the weights in each epoch.

% move CNN to GPU as needed
numGpus = numel(opts.gpus) ;
net = vl_simplenn_move(net_cpu, 'gpu') ;

% validation mode if learning rate is zero
training = learningRate > 0 ;
if training, mode = 'training' ; else, mode = 'validation' ; end
%if nargout > 2, mpiprofile on ; end


%dydz_syn = gpuArray(zeros(config.dydz_sz, 'single'));
%ydz_syn(net.filterSelected) = net.selectedLambdas;
%dydz_syn = repmat(dydz_syn, 1, 1, 1, config.nTileRow*config.nTileCol);
loss = 0; % used for the reconstruction error SSD
for t=1:opts.batchSize:numel(subset)
    fprintf('%s: epoch %02d: batch %3d/%3d: ', mode, epoch, ...
        fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
    batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
    batchTime = tic ;
    numDone = 0 ;
    res = [] ;
    res_syn = [];
%     stats = [] ;
    %   error = [] ;
    
        
        batchStart = t ;
        batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
        batch = subset(batchStart :  batchEnd) ;
        im = getBatch(imdb, batch) ;
        
        if numGpus >= 1
            im = gpuArray(im) ;
        end
        
        % training images
        %numImages = size(im, 4);
        cell_idx = (ceil(t / opts.batchSize));
        syn_mat = gpuArray(syn_mats{cell_idx});
        if isempty(syn_mat)
           %syn_mat = gpuArray(config.refsig*randn([1, config.z_dim, 1, size(im, 4)], 'single')); 
           syn_mat = gpuArray(config.refsig*zeros([1,config.z_dim,1, size(im, 4)], 'single'));
        end
        switch config.alg_type
            case 'alter_grad'
                syn_mat = alternate_gradient_z(config, net, im, syn_mat);       
                syn_mats{cell_idx} = syn_mat;      
                syn_mat = gpuArray(syn_mat);
                fz = vl_simplenn(net, syn_mat);
                dydz = im - fz(end).x;
                res = vl_simplenn(net, gpuArray(syn_mat), gpuArray(dydz), res, 'conserveMemory', 1, 'cudnn', 1);
                
            case 'joint_grad'
                net = vl_simplenn_move(net, 'gpu');
                fz = vl_simplenn(net, syn_mat);
                dydz = im - fz(end).x;
                res = vl_simplenn(net, gpuArray(syn_mat), gpuArray(dydz), res, 'conserveMemory', 1, 'cudnn', 1);
                syn_mat = syn_mat + config.joint_lambda  /config.s /config.s* res(1).dzdx ...
                          - config.joint_lambda /config.refsig /config.refsig* syn_mat;
                %syn_mat = syn_mat + 0.3 * gpuArray(randn(size(syn_mat), 'single'));
                
                syn_mats{cell_idx} = gather(syn_mat);
               
                
            case 'langevin_sampling'
                [syn_mat]= langevin_dynamic_z(config, net, im, syn_mat);       
                syn_mats{cell_idx} = gather(syn_mat);      
               % syn_mat_g = gpuArray(syn_mat);
                fz = vl_simplenn(net, syn_mat, [], []);
                dydz = im - fz(end).x;
                res = vl_simplenn(net, syn_mat, gpuArray(dydz), res, 'conserveMemory', 1, 'cudnn', 1);
               
        end
                
        numDone = numDone + numel(batch) ;
        

    % gather and accumulate gradients across labs
    if training
        if numGpus <= 1
            [net] = accumulate_gradients(opts, learningRate, batchSize, net, res, config);
             %[net] = accumulate_gradients_adam(opts, learningRate, batchSize, net, res, config, epoch);
        else
            fprintf('Not implement the multi-GPU version yet');
                    
        end
    end
    
    clear res;
    clear res_syn;
    
    % print learning statistics
    batchTime = toc(batchTime) ;
    %   stats = sum([stats,[batchTime ; error]],2); % works even when stats=[]
    speed = batchSize/batchTime ;
    
    fprintf(' %.2f s (%.1f data/s)', batchTime, speed) ;
    fprintf(' [%d/%d]', numDone, batchSize);
    fprintf('\n') ;
end
net_cpu = vl_simplenn_move(net, 'cpu') ;
end


% -------------------------------------------------------------------------
function [net] = accumulate_gradients(opts, lr, batchSize, net, res, config)
% -------------------------------------------------------------------------
%layer_sets = config.layer_sets;
% if nargin < 8
%     layer_sets = numel(net.layers):-1:numel(net.layers)-2;
% end



%res_syn_ref = res_syncell{1}; % just for reference use
for l = numel(net.layers):-1:1
    for j=1:numel(res(l).dzdw)
        thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
        thisLR = lr * net.layers{l}.learningRate(j) ;
        
        % accumualte from multiple labs (GPUs) if needed
       
        
        if isfield(net.layers{l}, 'weights')
            %gradient_dzdw_cell = [];
            %gradient_dzdw_sum = zeros(size(res_syn_ref(l).dzdw{j}));
            %for iImg = 1:num_img
                %res_syn = res_syncell{iImg};
            %    gradient_dzdw_cell{iImg} = (1/num_syn) * (1/config.s / config.s)* res_syn(l).dzdw{j};
            %    gradient_dzdw_sum =gradient_dzdw_sum + gradient_dzdw_cell{iImg};
            %end
            gradient_dzdw = (1/batchSize)* (1 / config.s / config.s)* res(l).dzdw{j};
            
            if max(abs(gradient_dzdw(:))) > 30 %10
                        gradient_dzdw = gradient_dzdw / max(abs(gradient_dzdw(:))) * 30;
            end

            net.layers{l}.momentum{j} = ...
                + opts.momentum * net.layers{l}.momentum{j} ...
                - thisDecay * net.layers{l}.weights{j} ...
                + gradient_dzdw;
            
            %             net.layers{l}.momentum{j} = gradient_dzdw;
            net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR *net.layers{l}.momentum{j};
            
            
            %       net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR * gradient_dzdw;
            if j == 1
                %res_l = min(l+2, length(res));
                %fprintf('\n layer %s:max response is %f, min response is %f.\n', net.layers{l}.name, max(res(res_l).x(:)), min(res(res_l).x(:)));
                fprintf('max gradient is %f, min gradient is %f, learning rate is %f\n', max(gradient_dzdw(:)), min(gradient_dzdw(:)), thisLR);
                                
            end % j==1
        end
    end
end
end


% -------------------------------------------------------------------------
function [net] = accumulate_gradients_adam(opts, lr, batchSize, net, res, config, epoch)
% -------------------------------------------------------------------------
%layer_sets = config.layer_sets;
% if nargin < 8
%     layer_sets = numel(net.layers):-1:numel(net.layers)-2;
% end


for l = numel(net.layers):-1:1
    for j=1:numel(res(l).dzdw)
        thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
        thisLR = lr * net.layers{l}.learningRate(j) ;
        lr_t = lr * sqrt(1.0 - (config.beta2)^epoch) / (1.0 - (config.beta1)^epoch);
        
        % accumualte from multiple labs (GPUs) if needed
       
        
        if isfield(net.layers{l}, 'weights')
            %gradient_dzdw_cell = [];
            %gradient_dzdw_sum = zeros(size(res_syn_ref(l).dzdw{j}));
            %for iImg = 1:num_img
                %res_syn = res_syncell{iImg};
            %    gradient_dzdw_cell{iImg} = (1/num_syn) * (1/config.s / config.s)* res_syn(l).dzdw{j};
            %    gradient_dzdw_sum =gradient_dzdw_sum + gradient_dzdw_cell{iImg};
            %end
            gradient_dzdw = -(1/batchSize)* (1 / config.s / config.s)* res(l).dzdw{j};
            gradient_dzdw2 = gradient_dzdw.^2;
            
            net.layers{l}.avg_first{j} = config.beta1 * net.layers{l}.avg_first{j} ...
                +(1 - config.beta1) * gradient_dzdw;
            net.layers{l}.avg_second{j} = config.beta2 * net.layers{l}.avg_second{j} ...
                +(1 - config.beta2) * gradient_dzdw2;
            
            net.layers{l}.weights{j} = net.layers{l}.weights{j} ...
                - lr_t * net.layers{l}.avg_first{j}./(sqrt(net.layers{l}.avg_second{j}) + config.eps);
  
            %       net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR * gradient_dzdw;
            if j == 1
                %res_l = min(l+2, length(res));
                %fprintf('\n layer %s:max response is %f, min response is %f.\n', net.layers{l}.name, max(res(res_l).x(:)), min(res(res_l).x(:)));
                fprintf('max gradient is %f, min gradient is %f, learning rate is %f\n', max(gradient_dzdw(:)), min(gradient_dzdw(:)), lr_t);
                                
            end % j==1
        end
    end
end
end



function net = accumulate_bias(net, res, config, mmap)
layer_sets = config.layer_sets;
% if nargin < 8
%     layer_sets = numel(net.layers):-1:numel(net.layers)-2;
% endwrite_bias

for l = layer_sets
    % accumualte from multiple labs (GPUs) if needed
    res_l = [];
    if nargin >= 4
        tag = sprintf('l%d',l) ;
        for g = setdiff(1:numel(mmap.Data), labindex)
            res_l = cat(4, res_l, mmap.Data(g).(tag));
        end
    end
    
    if isfield(net.layers{l}, 'weights') ...
            && strcmp(net.layers{l}.name(1:2), 'fc') == 0
        
        res_l = cat(4, res_l, res(l+1).x);
        sz = size(res_l);
        res_l = reshape(reshape(res_l, [], sz(4))', [], sz(3));
        bias = single(prctile(res_l, 100-config.sparse_level(l)));
        net.layers{l}.weights{2} = net.layers{l}.weights{2} - bias;
    end
end
end







 