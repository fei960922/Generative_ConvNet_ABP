function I_syn = output(syn_mat)

[net, config] = nfa_config('');
[I_syn, syn_mat_norm] = convert_syns_mat(config, syn_mat);
imwrite(I_syn, '1.jpg');
im = im2uint8(I_syn);
[imind,cm] = rgb2ind(im,256);
end