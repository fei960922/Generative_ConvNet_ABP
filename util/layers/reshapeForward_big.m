function y = reshapeForward_big( x )
% implement the forward pass of the custom reshape layer
% Right now, only reshape the 1*1*[64*8*4*4]*64 (where the last 64 is the 
% number of samples of z) to 4*4*512*64 (last 64 is the # of samples)
nsamp = size(x, 4);
y = reshape(x, [4, 4, 1024, nsamp]);

end

