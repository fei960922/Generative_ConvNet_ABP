function fz = test_2(net, syn_mats, config, z)
net.layers{4}.forward = @rforward;
net.layers{4}.backward = @rbackward;
net.layers{17}.forward = @tforward;
net.layers{17}.backward = @tbackward;

%% Reconstruct the trainning data
z_image = [];
for i=1:size(syn_mats,2)
    z_image = cat(4, z_image, syn_mats{i});
end
fz = vl_simplenn(net, z_image, [], [], 'accumulate', false, 'disableDropout', true);
fz = fz(end).x;
[I_syn, syn_mat_norm] = convert_syns_mat(config, fz);
imwrite(I_syn, ['output/reconstruct.jpg']);

%% Randomly Sample image from full Field by Normal Distribution
fz = vl_simplenn(net, z, [], [], 'accumulate', false, 'disableDropout', true);
fz = fz(end).x;
[I_syn, syn_mat_norm] = convert_syns_mat(config, fz);
imwrite(I_syn, ['output/synthesis.jpg']);

%% Random Sample image by sum up trainning data.

syn_number = 81;
z_dim = 100;
for use_pic = 50:50
    w = rand(syn_number, use_pic);
    ss = sum(w,2);
    for i = 1:syn_number
        for j = 1:use_pic
            w(i,j) = w(i,j) ./ ss(i);
        end
    end
    z_n = zeros(1, z_dim, 1, syn_number,'single');
    for i = 1:z_dim
        for j = 1:syn_number
            for k = 1:use_pic
                z_n(:, i, :, j) = z_n(:, i, :, j) + w(j,k) .* z_image(:,i,:,k+1);
            end
        end
    end
    fz = vl_simplenn(net, z_n, [], [], 'accumulate', false, 'disableDropout', true);
    fz = fz(end).x;
    [I_syn, syn_mat_norm] = convert_syns_mat(config, fz);
    imwrite(I_syn, ['output/sumup_' num2str(use_pic) '.jpg']);
end

%% From 1 pic

z_n = zeros(1, z_dim, 1, 81,'single');
for pic=1:50
for i = 1:z_dim
    for j = 1:81
        z_n(:, i, :, j) = (j/40-1) .* z_image(:,i,:,pic);
    end
end
fz = vl_simplenn(net, z_n, [], [], 'accumulate', false, 'disableDropout', true);
fz = fz(end).x;
[I_syn, syn_mat_norm] = convert_syns_mat(config, fz);
imwrite(I_syn, ['output/linear_' num2str(pic) '.jpg']);
end
    
%%

    function res_ = rforward(layer, res, res_)
        res_.x = reshapeForward(res.x);
    end

    function res = rbackward(layer, res, res_)
        res.dzdx = reshapeBackward(res.x, res_.dzdx);
    end
    
    function res_ = tforward(layer, res, res_)
        res_.x = tanhForward(res.x);
    end

    function res = tbackward(layer, res, res_)
        res.dzdx = tanhBackward(res.x, res_.dzdx);
    end
end