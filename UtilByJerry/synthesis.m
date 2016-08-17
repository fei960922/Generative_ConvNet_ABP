
syn_mat = randn([1,config.z_dim,1, 81], 'single');%config.synz;
fz = vl_simplenn(net, syn_mat, [], [], 'accumulate', false, 'disableDropout', true);
fz = fz(end).x;
config.nTileRow = 9;
config.nTileCol = 9;
[I_syn, syn_mat_norm] = convert_syns_mat(config, fz);
imshow(I_syn,'InitialMagnification','fit');