clear all;
% text texture network
category = 'water5';
[net, config] = nfa_texture_config(category);
%construct the generator network
net = textureGeneratorNet224(net, config);
%read image imdb
imgCell = read_images(config, net);
[imdb, getBatch] = convert2imdb(imgcell2mat(imgCell));

% change z_dim = 4;
%config.z_dim = 7;

%train the model
learningTime = tic;
net = train_model_texture_nfa(config, net, imdb, getBatch);

learningTime = toc(learningTime);
hrs = floor(learningTime / 3600);
learningTime = mod(learningTime, 3600);
mins = floor(learningTime / 60);
secds = mod(learningTime, 60);
fprintf('total learning time is %d hours / %d minutes / %.2f seconds.\n', hrs, mins, secds);