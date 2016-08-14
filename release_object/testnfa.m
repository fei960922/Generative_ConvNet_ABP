function [net, syn_mats] = testnfa()

    clear all;

    category = 'test_0814_bed_s_joint_gd';
    pathss = '/media/vclagpu/Data1/JerryXu/Code/ABP/Image/cat';

    [net, config] = nfa_config(category, pathss, true);
    config.fc_number=3;
    config.add_conv_behind=false;
    
    %construct the generator network

    net = generatorNet_new(net, config);
    config.zsyn = randn(1, config.z_dim, 1, config.nTileRow*config.nTileCol, 'single');
    %read image imdb
    imgCell = read_images(config, net);
    disp('Read Finished');
    [imdb, getBatch] = convert2imdb(imgcell2mat(imgCell));
    disp('!');
    %train the model
    learningTime = tic;
    [net, syn_mats] = train_model_nfa(config, net, imdb, getBatch, 0.3);

    learningTime = toc(learningTime);
    hrs = floor(learningTime / 3600);
    learningTime = mod(learningTime, 3600);
    mins = floor(learningTime / 60);
    secds = mod(learningTime, 60);
    fprintf('total learning time is %d hours / %d minutes / %.2f seconds.\n', hrs, mins, secds);

    interpolator(config, net, syn_mats);
end