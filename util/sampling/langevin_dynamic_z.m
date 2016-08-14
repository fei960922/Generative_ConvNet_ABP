function [syn_mat] =langevin_dynamic_z( config, net, im, syn_mat )
% Langevin sampling of the latent factor z. 
% different category may starts from the different initial points.
res = [];

for t = 1:config.Lstep
    res = vl_nfa(net, syn_mat, im, res, ...
        'conserveMemory', 1, ...
        'cudnn', 1);
    % reconstruct
    syn_mat = syn_mat + config.Delta * config.Delta /2 /config.s /config.s* res(1).dzdx ...
           - config.Delta * config.Delta /2 /config.refsig /config.refsig* syn_mat;
    % langevin noise
    if ~(isfield(config, 'no_noise_in_langevin')) || ~config.no_noise_in_langevin
        syn_mat = syn_mat + config.Delta * gpuArray(randn(size(syn_mat), 'single'));        
    end
end
end

