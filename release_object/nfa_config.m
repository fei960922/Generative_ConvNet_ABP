function [ net, config ] = nfa_config(category )

config.gpus = 1;
% config file for the model and training algorithm
config.categoryName = category;
% 3rd parth package, i.e., matconvnet
config.matconvv_path = '../matconvnet-1.0-beta16/';
run(fullfile(config.matconvv_path, 'matlab', 'vl_setupnn.m'));
%location of the deep model
config.model_path = '../model/';

%lcation of input dataset
config.inPath = ['../Image/', config.categoryName '/'];

%location of the result
config.Synfolder = ['./synthesiedImage/',config.categoryName, '/'];
config.working_folder = ['./working/', config.categoryName, '/'];
config.figure_folder = ['./figure/', config.categoryName,  '/'];
if ~exist('./synthesiedImage/', 'dir')
   mkdir('./synthesiedImage/') 
end
if ~exist('./working/', 'dir')
    mkdir('./working/')
end
if ~exist('./figure/', 'dir')
   mkdir('./figure/') 
end
if ~exist(config.Synfolder, 'dir')
   mkdir(config.Synfolder);
end
if ~exist(config.working_folder, 'dir')
    mkdir(config.working_folder);
end
if ~exist(config.figure_folder, 'dir')
    mkdir(config.figure_folder);
end

config.forceLearn = true;
config.z_dim = 10;
%parameter of the input (the ouput of the generator net is 64*64*3, so
%retrict the input to be 64*64
config.sx = 64;
config.sy = 64;
config.BatchSize = 64; % only used for batching training, but is =1

%parameter for the size of the output of the generator network
config.gensz = [64, 64, 3];

%parameter for the reference gaussian model
config.refsig = 1;
%parameter for the noise of the factor analysis
config.s = 0.3; %0.3
%parameter for langevin sampling
config.Lstep = 30; % step of langevin sampling
config.nTileRow = 9; %9
config.nTileCol = 9; % nTileRow*nTileCol is the # of samples we only visulize
%config.nsample = 1; % the number of samples
config.Delta = 0.3; % stepsize

% parameter for alternate gradient 
config.alt_lambda = 0.125; % the stepsize for the gradient evalution of z

% parameter for joint gradient
config.joint_lambda = 0.05; % the stepsize for the gradient evaluation of z

%parameter for SGD learning
config.nIteration = 600;
config.Gamma = 0.0001; % 0.0005 for egret

%parameter for Adam learning
config.beta1 = 0.5;
config.beta2 = 0.999;
config.lr0 = 0.0002;
config.eps = 1e-8;

config.is_texture = false;
config.interp_dim = 9;
config.vis_dim_x = config.sx; % may increase the dimension to do visualize
config.vis_dim_y = config.sy;
%config.vis_dim_z = config.z_dim;

% used for learning rate annealing (linear or exponential)
config.interval = 300;

% next, set up the algorithm type: alternative gradient (alter_grad), joint gradient (joint_grad),
% or Langevin sampling (langevin_sampling)
config.alg_type = 'langevin_sampling';

% parameter for the interpolation (line/ sphere/both)
config.interp_type = 'both';
config.n_pairs = 8;
config.n_parsamp = 8;

% parameter for copping the celebA
config.is_crop = false;
config.cropped_sz = 108;

net = [];
net.layers = {};
end

