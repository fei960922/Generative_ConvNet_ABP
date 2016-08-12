function net = addCustomTanhLayer( net, fwfun, bwfun )
% implement the custom tanh layer
layer.name = 'tanh';
layer.type = 'custom';
layer.forward = @forward;
layer.backward = @backward;
net.layers{end+1} = layer;

    function res_ = forward(layer, res, res_)
        res_.x = fwfun(res.x);
    end

    function res = backward(layer, res, res_)
        res.dzdx = bwfun(res.x, res_.dzdx);
    end

end

