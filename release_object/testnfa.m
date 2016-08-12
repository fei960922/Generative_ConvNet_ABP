clear all;

category = 'face';
[net, config] = nfa_config(category);
%construct the generator network

net = generatorNet(net, config);
%x = randn(1, 1, 2, 3, 'single');
%read image imdb
imgCell = read_images(config, net);
[imdb, getBatch] = convert2imdb(imgcell2mat(imgCell));

%train the model
learningTime = tic;
[net, syn_mats] = train_model_nfa(config, net, imdb, getBatch);

learningTime = toc(learningTime);
hrs = floor(learningTime / 3600);
learningTime = mod(learningTime, 3600);
mins = floor(learningTime / 60);
secds = mod(learningTime, 60);
fprintf('total learning time is %d hours / %d minutes / %.2f seconds.\n', hrs, mins, secds);

interpolator(config, net, syn_mats);