%%
%�ô�����������ɵ�xml������VOC2007���ݼ��е�trainval.txt;train.txt;test.txt��val.txt
%trainvalռ�����ݼ���50%��testռ�����ݼ���50%��trainռtrainval��50%��valռtrainval��50%��
%������ռ�ٷֱȿɸ����Լ������ݼ��޸ģ�������ݼ��Ƚ��٣�test��val����һЩ
%%
%ע���޸���������·��
xmlfilepath='.\Annotations';
txtsavepath='.\ImageSets\Main\';

xmlfile=dir(xmlfilepath);
numOfxml=length(xmlfile)-2;%��ȥ.��..  �ܵ����ݼ���С

trainval=sort(randperm(numOfxml,floor(numOfxml/2)));%trainvalΪ���ݼ���50%
test=sort(setdiff(1:numOfxml,trainval));%testΪʣ��50%

trainvalsize=length(trainval);%trainval�Ĵ�С
train=sort(trainval(randperm(trainvalsize,floor(trainvalsize))));
val=sort(setdiff(trainval,train));

ftrainval=fopen([txtsavepath 'trainval.txt'],'w');
ftest=fopen([txtsavepath 'test.txt'],'w');
ftrain=fopen([txtsavepath 'train.txt'],'w');
fval=fopen([txtsavepath 'val.txt'],'w');

for i=1:numOfxml
    if ismember(i,trainval)
        fprintf(ftrainval,'%s\n',xmlfile(i+2).name(1:end-4));
        if ismember(i,train)
            fprintf(ftrain,'%s\n',xmlfile(i+2).name(1:end-4));
        else
            fprintf(fval,'%s\n',xmlfile(i+2).name(1:end-4));
        end
    else
        fprintf(ftest,'%s\n',xmlfile(i+2).name(1:end-4));
    end
end
fclose(ftrainval);
fclose(ftrain);
fclose(fval);
fclose(ftest);