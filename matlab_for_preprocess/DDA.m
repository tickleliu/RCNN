function [x_r, y_r] = DDA(x1,y1,x2,y2)
%draw line between two points:use dda algorithm
length =abs(x2-x1);
if  abs(y2-y1)>length
    length=abs(y2-y1);
end
dx=(x2-x1)/length;
dy=(y2-y1)/length;
x=x1+0.5*sign(dx);
y=y1+0.5*sign(dy);
x_r = x1 + [0 : length] .* dx;
y_r = y1 + [0 : length] .* dy;
hold on
plot(x1, y1, 'ro');
plot(x2, y2, 'ro');
plot(x_r, y_r);
hold off
x_r = round(x_r);
y_r = round(y_r);
