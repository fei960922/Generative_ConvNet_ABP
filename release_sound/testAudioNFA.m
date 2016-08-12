clear all;
% text texture network
category = 'training1';
[net, config] = nfa_sound_config(category);
%construct the generator network
net = audioGeneratorNet_60(net, config);
%read image imdb
[audioCell, Fs] = read_sound(config);
[audiodb, getBatch] = convert2imdb_audio(audiocell2mat(audioCell));

% add frequency to config
config.Fs = Fs;

% change z_dim = 4;
%config.z_dim = 7;

%train the model
learningTime = tic;
net = train_model_audio_nfa(config, net, audiodb, getBatch);

learningTime = toc(learningTime);
hrs = floor(learningTime / 3600);
learningTime = mod(learningTime, 3600);
mins = floor(learningTime / 60);
secds = mod(learningTime, 60);
fprintf('total learning time is %d hours / %d minutes / %.2f seconds.\n', hrs, mins, secds);

