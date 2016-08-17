function dx = reshapeBackward_big( x, p )
% implement the backward pass of the custom reshape layer
% similar to forward pass, only reshape from 4*4*512*64 to 
% 1*1*[4*4*512]*64
nsamp = size(p, 4);
dx = reshape(p, [1, 1, 1024*4*4, nsamp]);

end

