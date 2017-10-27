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

