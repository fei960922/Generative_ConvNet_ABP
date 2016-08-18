function im_test = Shuffle_evaluate(config, net, syn_mats)

%% Init
    opts.cudnn = false;
    for i=1:length(net.layers)
        if strcmp(net.layers{i}.type,'custom')
            if strcmp(net.layers{i}.name,  'reshape')
                net.layers{i}.forward = @rforward;
                net.layers{i}.backward = @rbackward;
            elseif strcmp(net.layers{i}.name, 'tanh')
                net.layers{i}.forward = @tforward;
                net.layers{i}.backward = @tbackward;
            end
        end
    end
    syn_mat = syn_mats{1};
    temp = zeros(1,80,1,81,'single');
    for i=0:80
        for j=1:80
            if j>i
                temp(1,j,1,i+1) = syn_mat(1,j,1,5);
            else
                temp(1,j,1,i+1) = syn_mat(1,j,1,6);
            end
        end
    end
    fz = vl_simplenn(net, temp, [], [], 'accumulate', false, 'disableDropout', true);
    fz = fz(end).x;
    config.nTileRow = 9;
    config.nTileCol = 9;
    [I_syn, syn_mat_norm] = convert_syns_mat(config, fz);
    %SSD(t) = computer_error(im, fz);
    %fprintf('Reconstruction Error in %d step (Delta=%f): %f\n', t, config.Delta, SSD(t));
    %config.Delta = config.Delta / 1.01;
    im_test = I_syn;
    re_test = [];
    
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

