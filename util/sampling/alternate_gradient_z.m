function [syn_mat] =alternate_gradient_z( config, net, im, syn_mat )
% alternate gradient of the latent factor z. Here, didn't involve the
% stochstic term. 
%  syn_mats is the 4D matrix:
%  [1]*[config.z_dim]*[1]*[nImages]
%  for each image, should get "1" synthesised matrix(may speed up)
%syn_mat = gpuArray(syn_mat);
%net = vl_simplenn_move(net, 'gpu') ;
res = [];
%fz = vl_simplenn(net, syn_mat, [], []);
%dydz = im - fz(end).x;
%dydz = gpuArray(single(dydz));
%res = vl_simplenn(net, syn_mat, dydz, res, 'conserveMemory', 1, 'cudnn', 1);
for t = 1:config.Lstep
    res = vl_nfa(net, syn_mat, im, res, ...
        'conserveMemory', 1, ...
        'cudnn', 1);
    
    syn_mat = syn_mat + config.alt_lambda /config.s /config.s* res(1).dzdx ...
        - config.alt_lambda /config.refsig /config.refsig* syn_mat;
    
    %fz_tmp = vl_simplenn(net, syn_mat, [], []);
    %dydz = im - fz_tmp(end).x;
    %res = vl_simplenn(net, syn_mat, dydz, res, 'conserveMemory', 1, 'cudnn', 1);
end
clear res;

end

