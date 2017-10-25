close all;
clear all;
clc;

basepath = '8.7-8.28\';
dirpath = ['C:\Users\lml\Desktop\³¬ÉùÍ¼Æ¬\', basepath];
maskdirpath = ['C:\Users\lml\Desktop\mask\mask\255mask', basepath];
[ mFiles] = RangTraversal( dirpath, '.jpg' );

for k = 1 : length(mFiles)
%     for k = 1 : 10
    filepath = cell2mat(mFiles(k));
    filename = regexp(filepath, '[\\/]', 'split');
    filename = cell2mat(filename(end));
    
    img = imread(filepath);
    [x1, y1, x2, y2] = delete_border(img);
    img = double(img(y1:y2, x1:x2,:));
    
%     %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %% cut the mask file
%     maskfilenames = regexp(filename, '[.]', 'split');
%     maskfilename  = [maskdirpath, 'mask', cell2mat(maskfilenames(1)), '.bmp'];
%     smaskfilename  = [basepath, '/mask/', cell2mat(maskfilenames(1)), '.bmp'];
%     if ~exist(maskfilename, 'file')
%         continue
%     end
%     mask_img = imread(maskfilename); 
%     mask_img = mask_img(y1:y2, x1:x2,:);   
%     subplot(1,2,1);
%     image(uint8(img));
%     subplot(1,2,2);
%     image(mask_img);
%     imwrite(uint8(mask_img), smaskfilename);
%     %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    img_ori = img;
    image(uint8(img));
    
    imgT = zeros(size(img(:,:,1)));
    sz = size(img(:,:,1));
    mask = zeros(sz);
    for i = 1 : sz(1)
        for j = 1 : sz(2)
            imgT(i, j) =  max(img(i,j,:)) - min(img(i,j,:));
            if imgT(i, j) > 20
                mask(i, j) = 1;
            end
            if abs(img(i,j,1) - img(i,j,2)) <= 2 && img(i,j,1) - img(i,j,3) > 4
                mask(i, j) = 1;
            end
        end
    end
 
    
    img_t = img(:,:,2) * 2 - img(:,:,1) -img(:,:,3);
    mask = mask + double(img_t > 20);
    b = strel('disk', 1);  
    mask = imdilate(mask, b);
    mask = mask > 0;
    [x, y] = find(mask == 1);
    for j = 1 : length(x)
      img(x(j), y(j), 1) = 0;  
      img(x(j), y(j), 2) = 255;  
      img(x(j), y(j), 3) = 0;  
    end
    

    img_g = img_ori(:,:,3) / 3 + img_ori(:,:,2) / 3 + img_ori(:,:,1) / 3;
    [inpaintedImg,origImg,C,D] = criminisi(img, mask);
%     inpaintedImg(:,:,1) = medfilt2(inpaintedImg(:,:,1), [5 5]);
%     inpaintedImg(:,:,2) = medfilt2(inpaintedImg(:,:,2), [5 5]);
%     inpaintedImg(:,:,3) = medfilt2(inpaintedImg(:,:,3), [5 5]);
    subplot(1,2,1);
    image(uint8(img_ori));
    subplot(1,2,2);
    image(uint8(img));
    
    
    filename = regexp(filename, '[.]', 'split');
    filename = [basepath, cell2mat(filename(1)), '.bmp'];
    img_ipg = (inpaintedImg(:,:,1) + inpaintedImg(:,:,2) + inpaintedImg(:,:,3));
    img_ipg = img_ipg / 3;
    sz_ipg = size(img_ipg);
    img_ipg = imresize(img_ipg, round(sz_ipg / 3));
    imwrite(uint8(inpaintedImg), filename);
end

