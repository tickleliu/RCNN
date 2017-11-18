%%
%ͼƬ����·��Ϊ��
%E:\image\car
%E:\image\person
%car��person�Ǳ��泵�����˵��ļ���
%��Щ�ļ��л������ж����
%����image�ļ��������
%maindir=E:\image\
%�ô���������ǽ�ͼƬ���ָĳ�000123.jpg������ʽ
%%
clc;
clear;

maindir='I:\RCNN\matlab_for_preprocess\image\';
maskdir = 'I:\RCNN\matlab_for_preprocess\mask\';
output = './output.txt';
output_file = fopen(output, 'w');
name_long=6; %ͼƬ���ֵĳ��ȣ���000123.jpgΪ6,���9λ,���޸�
num_begin=1; %ͼ��������ʼ��������000123.jpg��ʼ�Ļ�����123

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
                fprintf('��ǰ�����ļ���%s',subdir(i).name);
                fprintf('�Ѿ�����%d��ͼƬ\n',n);
                n=n+1;
%                 pause(0.1);%���Խ���ͣȥ��
            end
        end
    end
end

close(output_file);