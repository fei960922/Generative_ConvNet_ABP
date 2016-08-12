function err = computer_error (a,b)
    
    c = a - b;
    c = 0.2989*c(:,:,1,:) + 0.5870*c(:,:,2,:) + 0.1140*c(:,:,3,:);
    c = sqrt(c .^ 2);
    err = sum(sum(sum(c)));
    
    err = err / size(a,1) / size(a,2) /size(a,4);

end