function [] = draw_interpolation_figures( config, net_cpu, syn_Z, interp_type )
net = vl_simplenn_move(net_cpu, 'gpu');
% syn_Z: n_syn * config.z_dim
n_syn = size(syn_Z, 1);
Z = zeros(1,config.z_dim, 1, n_syn, 'single');
for i = 1:n_syn
   Z(:,:,:,i) = syn_Z(i, :); 
end

fz_interp = vl_simplenn(net, gpuArray(Z), [], [], ...
        'accumulate', false, ...
        'disableDropout', true, ...
        'conserveMemory', 1 ...
        );
syn_mat_interp = gather(fz_interp(end).x);
% next, doing the same as convert_syns_mat. 
space = 5;

color = 0;

for i = 1:size(syn_mat_interp, 4)
   % syn_mat(:,:,:,i) = uint8(syn_mat(:,:,:,i));
   % syn_mat(:,:,:,i) = single(syn_mat(:,:,:,i)); % in yang's version, dont have this line
    gLow = min( reshape(syn_mat_interp(:,:,:,i), [],1));
    gHigh = max(reshape(syn_mat_interp(:,:,:,i), [],1));
    syn_mat_interp(:,:,:,i) = (syn_mat_interp(:,:,:,i)-gLow) / (gHigh - gLow);
end

% doing the same as mat2canvas

sx = config.vis_dim_x;
sy = config.vis_dim_y;
dim3 = size(syn_mat_interp, 3);
num_syn = size(syn_mat_interp, 4);

if strcmp(interp_type, 'line')
    nTileRow = config.n_pairs;
    nTileCol = 8;
else
    
    nTileRow = config.n_parsamp;
    
    nTileCol = 8;
end

I_syn = zeros(nTileRow * sy + space * (nTileRow-1), ...
    nTileCol * sx + space * (nTileCol-1), dim3, 'single');
k = 1;
for j = 1:nTileRow
    for i = 1:nTileCol
        I_syn( 1+(j-1)*sy + (j-1) * space : sy+(j-1)*sy + (j-1) * space, 1+(i-1)*sx + (i-1)*space : sx+(i-1)*sx + (i-1) * space, :) = syn_mat_interp(:,:,:,k);
        k = k+1;
        if k > num_syn
            break;
        end
    end
end
% padding the boundary

for row = 1:nTileRow
    I_syn(row * sx + (row-1) * space + 1:row * sx + (row-1) * space + space, :, :) = color;
end

for col = 1:nTileCol
    I_syn(:, col * sy + (col-1) * space + 1:col * sy + (col-1) * space + space, :) = color;
end

imwrite(I_syn,[config.Synfolder,  interp_type, '_interpolation', '.png']);
end

