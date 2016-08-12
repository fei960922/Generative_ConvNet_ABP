function fz = test_1(net,z)
net.layers{2}.forward = @rforward;
net.layers{2}.backward = @rbackward;
net.layers{15}.forward = @tforward;
net.layers{15}.backward = @tbackward;

fz = vl_simplenn(net, z, [], [], 'accumulate', false, 'disableDropout', true);
fz = fz(end).x;
for i=1:size(fz,4)
    imwrite(fz(:,:,:,i), ['output/' num2str(i) '.jpg']);
end

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