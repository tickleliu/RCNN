function [PSNR,MSE] = Evaluate(X,Y)
%求峰值信噪比，评价修复效果
if nargin<2
    E = X;
else
    if any(size(X)~=size(Y))
        error('The input size is not equal to each other!');
    end
    E = X-Y;
end
MSE = sum(E(:).*E(:))/prod(size(X));
PSNR = 10*log10(255^2/MSE);
end


