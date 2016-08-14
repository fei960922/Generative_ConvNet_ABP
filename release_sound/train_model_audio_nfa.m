function net = train_model_audio_nfa(config, net, audiodb, getBatch)
% train non-linear factor analysis model
% the input net is a cpu version.
opts.batchSize = config.BatchSize ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = config.gpus; % which GPU devices to use (none, one, or more)
opts.learningRate = 0.001 ;
opts.continue = false ;
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = true ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.prefetch = false;
opts.numFetchThreads = 8;
opts.cudnn = true ;
opts.weightDecay = 0.0001 ; %0.0001
opts.momentum = 0.5 ;

opts.plotDiagnostics = false ;
opts.memoryMapFile = fullfile(config.working_folder, 'matconvnet.bin') ;
%opts.learningRate = reshape(learningRate_array, 1, []);

if isempty(opts.train), opts.train = find(audiodb.audios.set==1) ; end
if isempty(opts.val), opts.val = find(audiodb.audios.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

opts.batchSize = min(opts.batchSize, numel(opts.train));
opts.numEpochs = config.nIteration;

evaluateMode = isempty(opts.train) ;

if ~evaluateMode
    for i=1:numel(net.layers)
        if isfield(net.layers{i}, 'weights')
            J = numel(net.layers{i}.weights) ;
            for j=1:J
                net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
            end
            if ~isfield(net.layers{i}, 'learningRate')
                net.layers{i}.learningRate = ones(1, J, 'single') ;
            end
            if ~isfield(net.layers{i}, 'weightDecay')
                net.layers{i}.weightDecay = ones(1, J, 'single') ;
            end
        end
    end
end


interval = ceil(opts.numEpochs / 60);

%mean_img = gather(net.normalization.averageImage);

% setup GPUs
numGpus = numel(opts.gpus) ;
%if numGpus > 1
%  if isempty(gcp('nocreate')),
%    parpool('local',numGpus) ;
%    spmd, gpuDevice(opts.gpus(labindex)), end
%  end
%elseif numGpus == 1
%  gpuDevice(opts.gpus)
%end
% Here, only consider the single gpu cases. So don't consider the num_syn
% for now
%num_syns = ceil(numel(opts.train) / opts.batchSize) * numel(opts.gpus);
%syn_mats = zeros([config.sx, config.sy, 3, ...
%    config.nTileRow * config.nTileCol, num_syns], 'single');
% actually a syn matrix for latent factor z. The last dim is the number of
% images. We initialize it to sample from gaussian
num_syns = ceil(numel(opts.train) / opts.batchSize);
syn_mats = cell(1, num_syns);
%syn_mats = randn([1, config.z_dim, 1, ...
%     opts.batchSize], 'single');

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

z = randn(1, config.vis_dim_z, 1, config.nTileRow*config.nTileCol, 'single');
SSD = zeros(config.nIteration, 1);
for epoch=1:opts.numEpochs
    fprintf('iteration %d / %d\n', epoch, opts.numEpochs);
    
    learningRate =  config.Gamma;
    
    % train one epoch and validate
    train = opts.train(randperm(numel(opts.train))) ; % shuffle
    train = opts.train;
    if numGpus <= 1
        [net, syn_mats] = process_epoch_audio_nfa(opts, getBatch, epoch, train, learningRate, audiodb, net, syn_mats, config);
        loss = compute_loss(opts, audiodb, getBatch, train, net, syn_mats);
    else
        fprintf('Not implement multiple GPU version yet');
       % spmd(numGpus)
       %     [net_, syn_mats_] = process_epoch_nfa(opts, getBatch, epoch, train, learningRate, imdb, net, syn_mats, config);
       % end
        
       % net = net_{1};
       % for i = 1:numGpus
       %     tmp = syn_mats_{i};
       %     syn_mats(i:numGpus:num_syns) = tmp(i:numGpus:num_syns);
       % end
        
       % clear net_;
       % clear syn_mats_;
       % clear tmp;
    end
    SSD(epoch) = loss;
    if mod(epoch - 1, 100) == 0 || epoch == opts.numEpochs
        %idx_syn = randi(num_syns, 1);
        %syn_mat = syn_mats(:,:,:,:, 1);
         
        syn_mat = sampler_audio(opts, config, net, epoch, z, audiodb);        
        
        if mod(epoch, inf) == 0 || epoch == opts.numEpochs
            model_file = [config.working_folder, 'iter_',...
                num2str(epoch) ,'_model.mat'];
            save(model_file, 'net', 'syn_mat');
        end
    end
    fprintf('iteration %d, reconstruction error %f \n', epoch, SSD(epoch));  
end % epoch
figure(10);
plot(1:config.nIteration, SSD(1:config.nIteration));
title('reconstruction error');
saveas(gcf,fullfile(config.Synfolder,'SSD.png'));
end

function loss = compute_loss(opts, audiodb, getBatch, subset, net_cpu, syn_mats)
net = vl_simplenn_move(net_cpu, 'gpu') ;
loss = 0;
res = [];
for t=1:opts.batchSize:numel(subset)
        batchStart = t ;
        batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
        batch = subset(batchStart :  batchEnd) ;
        audio = getBatch(audiodb, batch) ;    
 
        audio = gpuArray(audio);
        
        cell_idx = (ceil(t / opts.batchSize));
        syn_mat = gpuArray(syn_mats{cell_idx});
        fz = vl_simplenn(net, syn_mat);
        dydz = audio - fz(end).x;
        res = vl_simplenn(net, syn_mat, dydz, res, ...   
            'disableDropout', true, ...
            'conserveMemory', opts.conserveMemory, ...
            'backPropDepth', opts.backPropDepth, ...
            'sync', opts.sync, ...
            'cudnn', opts.cudnn) ;
        loss = loss + gather( mean(reshape(sqrt((res(end).x - audio).^2), [], 1)));
    
end
end

