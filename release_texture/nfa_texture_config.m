function [ net, config ] = nfa_texture_config(category )

config.gpus = 1;
% config file for the model and training algorithm
config.categoryName = category;
% 3rd parth package, i.e., matconvnet
config.matconvv_path = '../matconvnet-1.0-beta16/'; % previously ../
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
config.z_dim = 7; % if use 14, may use textureGeneratorNet224_large
%parameter of the input (the ouput of the generator net is 64*64*3, so
%retrict the input to be 64*64
config.sx = 224;
config.sy = 224;
config.BatchSize = 64; % only used for batching training, but is =1

%parameter for the size of the output of the generator network
config.gensz = [224, 224, 3];

%parameter for the reference gaussian model
config.refsig = 1;
%parameter for the noise of the factor analysis
config.s = 0.3;
%parameter for langevin sampling
config.Lstep = 10; % step of langevin sampling
config.nTileRow = 4;
config.nTileCol = 4; % nTileRow*nTileCol is the # of samples we only visulize
%config.nsample = 1; % the number of samples
config.Delta = 0.3; % stepsize

%parameter for SGD learning
config.nIteration = 600;
config.Gamma = 0.0001; % 0.0005 for egret

config.is_texture = true;
config.vis_dim_x = config.sx*2 ; % may increase the dimension to do visualize
config.vis_dim_y = config.sy*2 ;
config.vis_dim_z = config.z_dim*2 ;

config.alg_type = 'langevin_sampling';
net = [];
net.layers = {};
end

