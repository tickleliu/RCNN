close all; clear all; clc;
%生成训练列表
train_file_name = '../train_list.txt';
fine_tune_list_name = '../output.txt';

image_path = 'image&mask_delete_the_border/images/';
mask_path = 'image&mask_delete_the_border/masks/';

train_path = '../train/';
train_path1 = 'train/';
fine_tune_path = '../fine_tune/';
fine_tune_path1 = 'fine_tune/';

if ~exist(train_path)
    mkdir(train_path);
end

if ~exist([train_path, '1/'])
    mkdir([train_path, '1/']);
end

if ~exist([train_path, '0/'])
    mkdir([train_path, '0/']);
end

if ~exist(fine_tune_path)
    mkdir(fine_tune_path);
end

if ~exist([fine_tune_path, '0/'])
    mkdir([fine_tune_path, '0/']);
end

if ~exist([fine_tune_path, '1/'])
    mkdir([fine_tune_path, '1/']);
end

[ mFiles] = RangTraversal( image_path, '.bmp' );

train_file = fopen(train_file_name, 'w');
fine_tune_list = fopen(fine_tune_list_name, 'w');

count = 0;
bcount = 0;
for k = 1 : length(mFiles)
    %     for k = 1 : 10
    filepath = cell2mat(mFiles(k));
    filename = regexp(filepath, '[\\/]', 'split');
    filename = cell2mat(filename(end));
    mask_filename = [mask_path filename];
    mask = imread(mask_filename);
    sz = size(mask);
    img = imread(filepath);
    img_ori = img;
    mask_ori = mask;

    
    image_g = (img(:,:,1)+img(:,:,2)+img(:,:,3)) / 3;
%     image_h = histrogram(image_g);
%     image_h = image_g;
%     img(:,:,1) = uint8(image_h);
%     img(:,:,2) = uint8(image_h);
%     img(:,:,3) = uint8(image_h);
    
    subplot(1,2,1)
    imshow(img);
    subplot(1,2,2)
    imshow(mask);
    bw = bwlabel(mask(:,:,1), 8);
    region_count = max(max(bw));
    bb = regionprops(bw, 'BoundingBox');
    bb = [bb.BoundingBox];
    padding = 5;
    for i = 1 : region_count
        bb_ = bb(i,:);
        
        fine_image_file_name = [fine_tune_path, '1/', num2str(count), '.bmp'];
        fine_image_file_name1 = [fine_tune_path1, '1/', num2str(count), '.bmp'];
        imwrite(img, fine_image_file_name);
        fprintf(fine_tune_list, "%s %d %d,%d,%d,%d\n", fine_image_file_name1,...
            1, floor(bb_(1)) + 1, floor(bb_(2)) + 1, bb_(3), bb_(4));
        
        ori_image_file_name = ['../ori/', num2str(count), '.bmp'];
        imwrite(uint8(img_ori), ori_image_file_name);
        ori_mask_file_name = ['../ori_mask/', num2str(count), '.bmp'];
        imwrite(uint8(mask), ori_mask_file_name);
        if floor(bb_(2)) > padding
            bb_(2) = bb_(2) - padding;
        end
        if floor(bb_(1)) > padding
            bb_(1) = bb_(1) - padding;
        end
        if floor(bb_(2)) + bb_(4) + padding < sz(2)
            bb_(4) = bb_(4) + padding;
        end
        if floor(bb_(1)) + bb_(3) + padding < sz(2)
            bb_(3) = bb_(3) + padding;
        end
        image_temp = zeros( bb_(4), bb_(3), 3);
        image_temp(:,:,1) = img(floor(bb_(2)) + 1 : floor(bb_(2)) + bb_(4), ...
            floor(bb_(1)) + 1 : floor(bb_(1)) + bb_(3), 1);
        image_temp(:,:,2) = img(floor(bb_(2)) + 1 : floor(bb_(2)) + bb_(4), ...
            floor(bb_(1)) + 1 : floor(bb_(1)) + bb_(3), 2);
        image_temp(:,:,3) = img(floor(bb_(2)) + 1 : floor(bb_(2)) + bb_(4), ...
            floor(bb_(1)) + 1 : floor(bb_(1)) + bb_(3), 3);

        count = count + 1;
        save_image_file_name = [train_path, '1/', num2str(count), '.bmp'];
        save_image_file_name1 = [train_path1, '1/', num2str(count), '.bmp'];
        fprintf(train_file, "%s %d\n", save_image_file_name1, 1);
        imwrite(uint8(image_temp), save_image_file_name);
        
        count = count + 1;
        save_image_file_name = [train_path, '1/', num2str(count), '.bmp'];
        save_image_file_name1 = [train_path1, '1/', num2str(count), '.bmp'];
        fprintf(train_file, "%s %d\n", save_image_file_name1, 1);
        imwrite(uint8(rot90(image_temp)), save_image_file_name);
        
        count = count + 1;
        save_image_file_name = [train_path, '1/', num2str(count), '.bmp'];
        save_image_file_name1 = [train_path1, '1/', num2str(count), '.bmp'];
        fprintf(train_file, "%s %d\n", save_image_file_name1, 1);
        imwrite(uint8(flipud(image_temp)), save_image_file_name);
        
        count = count + 1;
        save_image_file_name = [train_path, '1/', num2str(count), '.bmp'];
        save_image_file_name1 = [train_path1, '1/', num2str(count), '.bmp'];
        fprintf(train_file, "%s %d\n", save_image_file_name1, 1);
        imwrite(uint8(fliplr(image_temp)), save_image_file_name);
        
    end
    
    
    half_height = floor(sz(1) / 2);
    half_width = floor(sz(2) / 2);
    trip_height = floor(sz(1) / 3);
    trip_width = floor(sz(2) / 3);
    b_loop_count = 0;
    for i = 1 : 100
        height = randi(half_height - trip_height) + trip_height;
        width = randi(half_width - trip_width) + trip_width;
        x = randi(sz(2) - width - 1);
        y = randi(sz(1) - height - 1);
        bb__ = [x y width height];
        r = IOU(bb__,bb_);
        if r < 0.1
            b_loop_count = b_loop_count + 1;
            bcount = bcount + 1;
            save_image_file_name = [train_path, '0/', num2str(bcount), '.bmp'];
            save_image_file_name1 = [train_path1, '0/', num2str(bcount), '.bmp'];
            fprintf(train_file, "%s %d\n", save_image_file_name1, 0);
            image_temp = zeros( height + 1, width + 1, 3);
            image_temp(:,:,1) = img(y : y + height, ...
                x : x + width, 1);
            image_temp(:,:,2) = img(y : y + height, ...
                x : x + width, 2);
            image_temp(:,:,3) = img(y : y + height, ...
                x : x + width, 3);
            imwrite(uint8(image_temp), save_image_file_name);
        end
        if b_loop_count > 4
            break
        end
    end
    if region_count > 2
        break;
    end
end
fclose(train_file);