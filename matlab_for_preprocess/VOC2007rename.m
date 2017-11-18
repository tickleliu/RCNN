%%
%图片保存路径为：
%E:\image\car
%E:\image\person
%car和person是保存车和行人的文件夹
%这些文件夹还可以有多个，
%放在image文件夹里就行
%maindir=E:\image\
%该代码的作用是将图片名字改成000123.jpg这种形式
%%
clc;
clear;

maindir='I:\RCNN\matlab_for_preprocess\image\';
maskdir = 'I:\RCNN\matlab_for_preprocess\mask\';
output = './output.txt';
output_file = fopen(output, 'w');
name_long=6; %图片名字的长度，如000123.jpg为6,最多9位,可修改
num_begin=1; %图像命名开始的数字如000123.jpg开始的话就是123

subdir = dir(maindir);
n=1;

for i = 1:length(subdir)
    if ~strcmp(subdir(i).name ,'.') && ~strcmp(subdir(i).name,'..')
        subsubdir = dir(strcat(maindir,subdir(i).name));
        for j=1:length(subsubdir)
            if ~strcmp(subsubdir(j).name ,'.') && ~strcmp(subsubdir(j).name,'..')
                img=imread([maindir,subdir(i).name,'\',subsubdir(j).name]);
                filename = regexp(subsubdir(j).name, '[.]', 'split');
                mask = imread([maskdir,subdir(i).name,'\',cell2mat(filename(1)), '.bmp']);
                subplot(2,1,1)
                imshow(img);
                subplot(2,1,2)
                imshow(mask);
                
                
                
                str=num2str(num_begin,'%09d');
                newname=strcat(str,'.jpg');
                newname=newname(end-(name_long+3):end);
                system(['rename ' [maindir,subdir(i).name,'\',subsubdir(j).name] ' ' newname]);

                bw = bwlabel(mask(:,:,1), 8);
                region_count = max(max(bw));
                bb_ = regionprops(bw, 'BoundingBox');
                bb_ = [bb_.BoundingBox];
                fprintf(output_file, "%s %s %d %d %d %d\n", newname,...
            'mass', floor(bb_(1)) + 1, floor(bb_(2)) + 1, bb_(3), bb_(4));
                num_begin=num_begin+1;
                fprintf('当前处理文件夹%s',subdir(i).name);
                fprintf('已经处理%d张图片\n',n);
                n=n+1;
%                 pause(0.1);%可以将暂停去掉
            end
        end
    end
end

close(output_file);