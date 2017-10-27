function ratio = IOU(Reframe,GTframe)
%Reframe(x,y,w,h) x,yΪ���Ͻ�����
x1 = Reframe(1);
y1 = Reframe(2);
width1 = Reframe(3);
height1 = Reframe(4);


x2 = GTframe(1);
y2 = GTframe(2);
width2 = GTframe(3);
height2 = GTframe(4);


endx = max(x1+width1,x2+width2);%x�����ֵ
startx = min(x1,x2);%x����Сֵ
width = width1+width2-(endx-startx);%�ص����ο�


endy = max(y1+height1,y2+height2);%y�����ֵ
starty = min(y1,y2);%y����Сֵ
height = height1+height2-(endy-starty);%�ص����ο�


if width<=0||height<=0
    ratio = 0;
    Area=0;
else
    Area = width*height;%������
    Area1 = width1*height1;%��һ��Box���
    Area2 = width2*height2;%�ڶ���Box���
    ratio = Area/(Area1+Area2-Area);%�ص���
end
end

