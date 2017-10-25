close all; clear all; clc;
%生成训练列表
train_file_name = 'train_list.txt';
fine_tune_list_name = 'fine_tune_list.txt';

image_path = 'image/';
mask_path = 'mask/';

train_path = 'train/';
fine_tune_path = 'fine_tune/';

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
    image = imread(filepath);
    bw = bwlabel(mask(:,:,1), 8);
    
    region_count = max(max(bw));
    bb = regionprops(bw, 'BoundingBox');
    bb = [bb.BoundingBox];
    padding = 5;
    for i = 1 : region_count
        bb_ = bb(i,:);
        
        fine_image_file_name = [fine_tune_path, '1/', num2str(count), '.bmp'];
        imwrite(image, fine_image_file_name);
        fprintf(fine_tune_list, "%s %d %d,%d,%d,%d\n\r", fine_image_file_name,...
            1, floor(bb_(1)) + 1, floor(bb_(2)) + 1, bb_(3), bb_(4));
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
        image_temp(:,:,1) = image(floor(bb_(2)) + 1 : floor(bb_(2)) + bb_(4), ...
            floor(bb_(1)) + 1 : floor(bb_(1)) + bb_(3), 1);
        image_temp(:,:,2) = image(floor(bb_(2)) + 1 : floor(bb_(2)) + bb_(4), ...
            floor(bb_(1)) + 1 : floor(bb_(1)) + bb_(3), 2);
        image_temp(:,:,3) = image(floor(bb_(2)) + 1 : floor(bb_(2)) + bb_(4), ...
            floor(bb_(1)) + 1 : floor(bb_(1)) + bb_(3), 3);
        count = count + 1;
        save_image_file_name = [train_path, '1/', num2str(count), '.bmp'];
        fprintf(train_file, "%s %d\n\r", save_image_file_name, 1);
        imwrite(uint8(image_temp), save_image_file_name);
        
        count = count + 1;
        save_image_file_name = [train_path, '1/', num2str(count), '.bmp'];
        fprintf(train_file, "%s %d\n\r", save_image_file_name, 1);
        imwrite(uint8(rot90(image_temp)), save_image_file_name);
        
        count = count + 1;
        save_image_file_name = [train_path, '1/', num2str(count), '.bmp'];
        fprintf(train_file, "%s %d\n\r", save_image_file_name, 1);
        imwrite(uint8(flipud(image_temp)), save_image_file_name);
        
        count = count + 1;
        save_image_file_name = [train_path, '1/', num2str(count), '.bmp'];
        fprintf(train_file, "%s %d\n\r", save_image_file_name, 1);
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
            fprintf(train_file, "%s %d\n\r", save_image_file_name, 0);
            image_temp = zeros( height + 1, width + 1, 3);
            image_temp(:,:,1) = image(y : y + height, ...
                x : x + width, 1);
            image_temp(:,:,2) = image(y : y + height, ...
                x : x + width, 2);
            image_temp(:,:,3) = image(y : y + height, ...
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

function ratio = IOU(Reframe,GTframe)
%Reframe(x,y,w,h) x,y为左上角坐标
x1 = Reframe(1);
y1 = Reframe(2);
width1 = Reframe(3);
height1 = Reframe(4);


x2 = GTframe(1);
y2 = GTframe(2);
width2 = GTframe(3);
height2 = GTframe(4);


endx = max(x1+width1,x2+width2);%x轴最大值
startx = min(x1,x2);%x轴最小值
width = width1+width2-(endx-startx);%重叠矩形宽


endy = max(y1+height1,y2+height2);%y轴最大值
starty = min(y1,y2);%y轴最小值
height = height1+height2-(endy-starty);%重叠矩形宽


if width<=0||height<=0
    ratio = 0;
    Area=0;
else
    Area = width*height;%冲得面积
    Area1 = width1*height1;%第一个Box面积
    Area2 = width2*height2;%第二个Box面积
    ratio = Area/(Area1+Area2-Area);%重叠率
end
end
