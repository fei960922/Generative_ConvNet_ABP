function img_mat = imgcell2mat(img_cell)
img_mat = [];
if isempty(img_cell)
    return; 
end
numImages = numel(img_cell);
sz = size(img_cell{1});
if isa(img_cell{1}, 'gpuArray')
    img_mat = gpuArray(zeros([sz, numImages], 'single'));
else
    img_mat = zeros([sz, numImages], 'single');
end
for i = 1:numImages
    %if size(img_cell{i}, 3) == 3
    %  img_mat(:,:,:,end+1) = img_cell{i};
    %end
    img_mat(:,:,:,i) = img_cell{i};
end