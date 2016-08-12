function [imgCell, im_ori] = read_images(config, net)

img_file = [config.working_folder, 'images.mat'];
files = read_image_recursive(config.inPath);
if isfield(config,'max_img') && length(files)>config.max_img
    files = files(1:config.max_img);
end

im_ori = [];
numImages = 0;

if exist(img_file, 'file')
    load(img_file);
    numImages = length(imgCell);
end

if numImages ~= length(files) || config.forceLearn == true
    imgCell = cell(1, length(files));
    for iImg = 1:length(files)
        
        img = single(imread(fullfile(files(iImg).name)));
       
        if config.is_crop == true
            h = round((size(img, 1) - config.cropped_sz)/2);
         
            w = round((size(img, 2) - config.cropped_sz)/2);
        
            cropped_img = img(h:h+config.cropped_sz-1, w:w+config.cropped_sz-1, :);
          
            img = cropped_img;
            
            img = imresize(img, [config.sx,config.sy]);
        elseif config.is_preprocess == true
            % do rescaling 
            h = size(img, 1);
            w = size(img, 2);
            if (h < w)
                rescaled_img = imresize(img, [config.rescale_sz, config.rescale_sz * (w/h)]);
            else
                rescaled_img = imresize(img, [config.rescale_sz * (h/w), config.rescale_sz]);
            end
            
            % do random crop
            rescale_h = size(rescaled_img, 1);
            rescale_w = size(rescaled_img, 2);
            h1 = randi(rescale_h - config.sx, 1);
            w1 = randi(rescale_w - config.sy, 1);
            cropped_img = rescaled_img(h1:h1+config.sx-1, w1:w1+config.sy-1, :);
            img = cropped_img;
        else
            img = imresize(img, [config.sx,config.sy]);
        end
        
        imgCell{iImg} = 2*(img - min(img(:)))/(max(img(:))-min(img(:)))-1 ;
    end
end
end

function files = read_image_recursive(path)
    list = dir(path);
    files = [];
    for i=1:length(list)
        item = list(i);
        if (item.isdir==1) & ~strcmp(item.name,'.') & ~strcmp(item.name,'..')
            files = [files;read_image_recursive([path item.name])]
        else
            if (strfind(item.name,'.jpg'))
                item.name = [path item.name];
                files = [files;item];
            end
        end
    end
end