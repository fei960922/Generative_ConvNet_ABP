function [re_train, re_test, im_output] = evaluation(config, net, catagory)

%% Init
    
    if nargin<2, load(config);end
    
    main_path = 'C:\Users\fei96\Documents\MATLAB\CSST\ABP\';
    opts.cudnn = false;
    config.Delta = 0.05;
    config.max_img = 10;
    
    for i=1:length(net.layers)
        if strcmp(net.layers{i}.type,'custom')
            if strcmp(net.layers{i}.name,  'reshape')
                net.layers{i}.forward = @rforward;
                net.layers{i}.backward = @rbackward;
            elseif strcmp(net.layers{i}.name, 'tanh')
                net.layers{i}.forward = @tforward;
                net.layers{i}.backward = @tbackward;
            end
        end
    end

%% Image input
    
    config.inPath = [main_path 'Image\' catagory '\'];
    imgCell= read_images(config, net);
    [im, getBatch] = convert2imdb(imgcell2mat(imgCell));
    im = im.images.data;
    syn_mat = config.refsig*randn([1,config.z_dim,1, size(im, 4)], 'single');%config.synz;
    fprintf('%d testing images found in training data......\n', size(im,4));

%% Langevin_dynamic
    tic;
    res = [];
    SSD = [];
    for t = 1:30
        res = vl_nfa(net, syn_mat, im, res, 'conserveMemory', 1);
        syn_mat = syn_mat + config.Delta * config.Delta /2 /config.s /config.s* res(1).dzdx ...
               - config.Delta * config.Delta /2 /config.refsig /config.refsig* syn_mat;
        syn_mat = syn_mat + config.Delta * randn(size(syn_mat), 'single');          
    if mod(t,5)==0
        fz = vl_simplenn(net, syn_mat, [], [], 'accumulate', false, 'disableDropout', true);
        fz = fz(end).x;
        config.nTileRow = 3;
        config.nTileCol = 3;
        [I_syn, syn_mat_norm] = convert_syns_mat(config, fz);
        SSD(t) = computer_error(im, fz);
        fprintf('Reconstruction Error in %d step (Delta=%f): %f\n', t, config.Delta, SSD(t));
        config.Delta = config.Delta / 1.01;
        im_output = I_syn;
    end
    end
    imshow(I_syn);
    re_train = SSD;
    toc;

%% Image input
    
    config.inPath = [main_path 'Image\' catagory '_eva\'];
    imgCell= read_images(config, net);
    [im, getBatch] = convert2imdb(imgcell2mat(imgCell));
    im = im.images.data;
    syn_mat = config.refsig*randn([1,config.z_dim,1, size(im, 4)], 'single');%config.synz;
    fprintf('%d testing images found in testing data......\n', size(im,4));

%% Langevin_dynamic
    tic;
    res = [];
    SSD = [];
    for t = 1:30
        res = vl_nfa(net, syn_mat, im, res, 'conserveMemory', 1);
        syn_mat = syn_mat + config.Delta * config.Delta /2 /config.s /config.s* res(1).dzdx ...
               - config.Delta * config.Delta /2 /config.refsig /config.refsig* syn_mat;
        syn_mat = syn_mat + config.Delta * randn(size(syn_mat), 'single');          
    if mod(t,5)==0
        fz = vl_simplenn(net, syn_mat, [], [], 'accumulate', false, 'disableDropout', true);
        fz = fz(end).x;
        config.nTileRow = 3;
        config.nTileCol = 3;
        [I_syn, syn_mat_norm] = convert_syns_mat(config, fz);
        SSD(t) = computer_error(im, fz);
        fprintf('Reconstruction Error in %d step (Delta=%f): %f\n', t, config.Delta, SSD(t));
        config.Delta = config.Delta / 1.01;
        im_output = I_syn;
    end
    end
    imshow(I_syn);
    re_test = SSD;
    toc;
    
    
%%

    function res_ = rforward(layer, res, res_)
        res_.x = reshapeForward(res.x);
    end

    function res = rbackward(layer, res, res_)
        res.dzdx = reshapeBackward(res.x, res_.dzdx);
    end
    
    function res_ = tforward(layer, res, res_)
        res_.x = tanhForward(res.x);
    end

    function res = tbackward(layer, res, res_)
        res.dzdx = tanhBackward(res.x, res_.dzdx);
    end

end

