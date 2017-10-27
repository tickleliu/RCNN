close all; clear all; clc;
%生成训练列表
image_path = 'image/';
mask_path = 'mask/';

train_path = 'train/';
fine_tune_path = 'fine_tune/';

[ mFiles] = RangTraversal( image_path, '.bmp' );

count = 0;
bcount = 0;
for k = 1 : length(mFiles)
    %     for k = 1 : 10
    filepath = cell2mat(mFiles(k));
    filename = regexp(filepath, '[\\/]', 'split');
    filename = cell2mat(filename(end));
    mask_filename = [mask_path filename];
    
    mask = imread(mask_filename);
    img = imread(filepath);

    img = img(10:end-10, 10:end-10,:);
    mask = mask(10:end-10, 10:end-10,:);
    
    image_g = (img(:,:,1)+img(:,:,2)+img(:,:,3)) / 3;
    image_h = histrogram(image_g);
    image_w = whiten(image_h);
    [x, y] = find(mask == 255);
     for i = 1 : length(x)
        img(x(i), y(i), 1) = 255;
    end
    bw = bwlabel(mask(:,:,1), 8);
    figure(1)
    subplot(2,2,1)  
%     image_g(image_g > mean(mean(image_g))) = 250;
    image(img);
    
    subplot(2,2,2)
    imshow(image_h);
    hold on
    [x, y] = find(image_h==min(min(image_h)));
    scatter(y(1:end), x(1:end), 'o');
    hold off
    subplot(2,2,3)
    imshow(image_w);
    image_m = zeros(size(image_g));
    image_m(image_h > mean(mean(image_h)) * 0.8) = 255;
    se=strel('disk',4);
    image_m=imerode(~image_m, se);
    L = bwlabel(image_m, 4);   
    S = regionprops(L, 'Area');
    bw2 = ismember(L, find([S.Area] >= 50)); 
    bw3 = ismember(L, find([S.Area] <= 5000)); 
    bw2 = bw2 & bw3;
    subplot(2,2,4)
    image_w = whiten(image_g);
    imshow(image_w);
%     figure(2)
%     for i = 1 : 5
%         image_m = zeros(size(image_g));
%         image_m(image_g > mean(mean(image_g)) * (i * 0.01 + 0.8)) = 255;
%         subplot(1,5,i)
%         imshow(uint8(image_m));
%     end
    
end


