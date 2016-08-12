function [syn_mat] =langevin_dynamic_z( config, net, im, syn_mat )
% Langevin sampling of the latent factor z. 
%  syn_mats is the 4D matrix:
%  [1]*[1]*[z_dim]*[nImages]
%  for each image, should get "1" synthesised matrix(may speed up)
%syn_mat_g = gpuArray(syn_mat);
%net = vl_simplenn_move(net, 'gpu') ;
res = [];
fz = vl_simplenn(net, syn_mat, [], []);
dydz = im - fz(end).x;
dydz = gpuArray(single(dydz));
res = vl_simplenn(net, syn_mat, dydz, res, 'conserveMemory', 1, 'cudnn', 1);
for t = 1:config.Lstep
    
    syn_mat = syn_mat + config.Delta * config.Delta /2 /config.s /config.s* res(1).dzdx ...
        - config.Delta * config.Delta /2 /config.refsig /config.refsig* syn_mat;
    syn_mat = syn_mat + config.Delta * gpuArray(randn(size(syn_mat), 'single')); % change tian
    fz_tmp = vl_simplenn(net, syn_mat, [], []);
    dydz = im - fz_tmp(end).x;
    res = vl_simplenn(net, syn_mat, dydz, res, 'conserveMemory', 1, 'cudnn', 1);
    
end
clear res;
%syn_mat = gather(syn_mat);

end

