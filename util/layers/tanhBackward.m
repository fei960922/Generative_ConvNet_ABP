function dx = tanhBackward( x, p )
%implement backward pass of the tanh function

dx = p.* (1.0 - tanh(x).^2);


end

