%%
%�ô��������voc2007���ݼ��е�xml�ļ���
%txt�ļ�ÿ�и�ʽΪ��000002.jpg dog 44 28 132 121
%��ÿ����ͼƬ����Ŀ�����͡���Χ��������ɣ��ո����
%���һ��ͼƬ�ж��Ŀ�꣬���ʽ���£�����������Ŀ�꣩
%000002.jpg dog 44 28 132 121
%000002.jpg car 50 27 140 110
%��Χ������Ϊ���ϽǺ����½�
%���ߣ�С����_
%CSDN:http://blog.csdn.net/sinat_30071459
%%
clc;
clear;
%ע���޸������ĸ�����
imgpath='VOC2007\JPEGImages\';%ͼ�����ļ���
txtpath='VOC2007\output.txt';%txt�ļ�
xmlpath_new='VOC2007\Annotations/';%�޸ĺ��xml�����ļ���
foldername='VOC2007';%xml��folder�ֶ���


fidin=fopen(txtpath,'r');
lastname='begin';

while ~feof(fidin)
    tline=fgetl(fidin);
    str = regexp(tline, ' ','split')
    filepath=[imgpath,str{1}];
    img=imread(filepath);
    filepath
    [str2double(str{3}),str2double(str{4}),abs(str2double(str{5})),abs(str2double(str{6}))]
    [h,w,d]=size(img);
    imshow(img);
    %       rectangle('Position',[str2double(str{3}),str2double(str{4}),abs(str2double(str{5})-str2double(str{3})),abs(str2double(str{6})-str2double(str{4}))],'LineWidth',4,'EdgeColor','r');
    
    
    rectangle('Position',[str2double(str{3}),str2double(str{4}),abs(str2double(str{5})),abs(str2double(str{6}))],'LineWidth',4,'EdgeColor','r');
    pause(0.1);
    
    if strcmp(str{1},lastname)%����ļ�����ȣ�ֻ������object
        object_node=Createnode.createElement('object');
        Root.appendChild(object_node);
        node=Createnode.createElement('name');
        node.appendChild(Createnode.createTextNode(sprintf('%s',str{2})));
        object_node.appendChild(node);
        
        node=Createnode.createElement('pose');
        node.appendChild(Createnode.createTextNode(sprintf('%s','Unspecified')));
        object_node.appendChild(node);
        
        node=Createnode.createElement('truncated');
        node.appendChild(Createnode.createTextNode(sprintf('%s','0')));
        object_node.appendChild(node);
        
        node=Createnode.createElement('difficult');
        node.appendChild(Createnode.createTextNode(sprintf('%s','0')));
        object_node.appendChild(node);
        
        bndbox_node=Createnode.createElement('bndbox');
        object_node.appendChild(bndbox_node);
        
        node=Createnode.createElement('xmin');
        node.appendChild(Createnode.createTextNode(sprintf('%s',num2str(str{3}))));
        bndbox_node.appendChild(node);
        
        node=Createnode.createElement('ymin');
        node.appendChild(Createnode.createTextNode(sprintf('%s',num2str(str{4}))));
        bndbox_node.appendChild(node);
        
        node=Createnode.createElement('xmax');     
        xmax = str2double(str{5}) + str2double(str{3});
        node.appendChild(Createnode.createTextNode(sprintf('%s',num2str(xmax))));
        bndbox_node.appendChild(node);
        
        node=Createnode.createElement('ymax');
        ymax = str2double(str{6}) + str2double(str{5});
        node.appendChild(Createnode.createTextNode(sprintf('%s',num2str(ymax))));
        bndbox_node.appendChild(node);
    else %����ļ������ȣ�����Ҫ�½�xml
        copyfile(filepath, 'JPEGImages');
        %�ȱ�����һ�ε�xml
        if exist('Createnode','var')
            tempname=lastname;
            tempname=strrep(tempname,'.jpg','.xml');
            xmlwrite(tempname,Createnode);
        end
        
        
        Createnode=com.mathworks.xml.XMLUtils.createDocument('annotation');
        Root=Createnode.getDocumentElement;%���ڵ�
        node=Createnode.createElement('folder');
        node.appendChild(Createnode.createTextNode(sprintf('%s',foldername)));
        Root.appendChild(node);
        node=Createnode.createElement('filename');
        node.appendChild(Createnode.createTextNode(sprintf('%s',str{1})));
        Root.appendChild(node);
        source_node=Createnode.createElement('source');
        Root.appendChild(source_node);
        node=Createnode.createElement('database');
        node.appendChild(Createnode.createTextNode(sprintf('My Database')));
        source_node.appendChild(node);
        node=Createnode.createElement('annotation');
        node.appendChild(Createnode.createTextNode(sprintf('VOC2007')));
        source_node.appendChild(node);
        
        node=Createnode.createElement('image');
        node.appendChild(Createnode.createTextNode(sprintf('flickr')));
        source_node.appendChild(node);
        
        node=Createnode.createElement('flickrid');
        node.appendChild(Createnode.createTextNode(sprintf('NULL')));
        source_node.appendChild(node);
        owner_node=Createnode.createElement('owner');
        Root.appendChild(owner_node);
        node=Createnode.createElement('flickrid');
        node.appendChild(Createnode.createTextNode(sprintf('NULL')));
        owner_node.appendChild(node);
        
        node=Createnode.createElement('name');
        node.appendChild(Createnode.createTextNode(sprintf('xiaoxianyu')));
        owner_node.appendChild(node);
        size_node=Createnode.createElement('size');
        Root.appendChild(size_node);
        
        node=Createnode.createElement('width');
        node.appendChild(Createnode.createTextNode(sprintf('%s',num2str(w))));
        size_node.appendChild(node);
        
        node=Createnode.createElement('height');
        node.appendChild(Createnode.createTextNode(sprintf('%s',num2str(h))));
        size_node.appendChild(node);
        
        node=Createnode.createElement('depth');
        node.appendChild(Createnode.createTextNode(sprintf('%s',num2str(d))));
        size_node.appendChild(node);
        
        node=Createnode.createElement('segmented');
        node.appendChild(Createnode.createTextNode(sprintf('%s','0')));
        Root.appendChild(node);
        object_node=Createnode.createElement('object');
        Root.appendChild(object_node);
        node=Createnode.createElement('name');
        node.appendChild(Createnode.createTextNode(sprintf('%s',str{2})));
        object_node.appendChild(node);
        
        node=Createnode.createElement('pose');
        node.appendChild(Createnode.createTextNode(sprintf('%s','Unspecified')));
        object_node.appendChild(node);
        
        node=Createnode.createElement('truncated');
        node.appendChild(Createnode.createTextNode(sprintf('%s','0')));
        object_node.appendChild(node);
        
        node=Createnode.createElement('difficult');
        node.appendChild(Createnode.createTextNode(sprintf('%s','0')));
        object_node.appendChild(node);
        
        bndbox_node=Createnode.createElement('bndbox');
        object_node.appendChild(bndbox_node);
        
        node=Createnode.createElement('xmin');
        node.appendChild(Createnode.createTextNode(sprintf('%s',num2str(str{3}))));
        bndbox_node.appendChild(node);
        
        node=Createnode.createElement('ymin');
        node.appendChild(Createnode.createTextNode(sprintf('%s',num2str(str{4}))));
        bndbox_node.appendChild(node);
        
        node=Createnode.createElement('xmax');
        xmax = str2double(str{5}) + str2double(str{3});
        node.appendChild(Createnode.createTextNode(sprintf('%s',num2str(xmax))));
        bndbox_node.appendChild(node);
        
        node=Createnode.createElement('ymax');
        ymax = str2double(str{6}) + str2double(str{5});
        node.appendChild(Createnode.createTextNode(sprintf('%s',num2str(ymax))));
        bndbox_node.appendChild(node);
        
        lastname=str{1};
    end
    %�������һ��
    if feof(fidin)
        tempname=lastname;
        tempname=strrep(tempname,'.jpg','.xml');
        xmlwrite(tempname,Createnode);
    end
end
fclose(fidin);

file=dir(pwd);
for i=1:length(file)
    if length(file(i).name)>=4 && strcmp(file(i).name(end-3:end),'.xml')
        fold=fopen(file(i).name,'r');
        fnew=fopen([xmlpath_new file(i).name],'w');
        line=1;
        while ~feof(fold)
            tline=fgetl(fold);
            if line==1
                line=2;
                continue;
            end
            expression = '   ';
            replace=char(9);
            newStr=regexprep(tline,expression,replace);
            fprintf(fnew,'%s\n',newStr);
        end
        fprintf('�Ѵ���%s\n',file(i).name);
        fclose(fold);
        fclose(fnew);
        delete(file(i).name);
    end
end