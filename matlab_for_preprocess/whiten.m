function X = whiten(I)
[m,n] = size(I);
P = reshape(I,m*n,1);
P = single(P);
mu = mean(P);
va = var(P);
X = double(I);
for i=1:m
    for j=1:n
        X(i,j)=(double(I(i,j))-mu)/sqrt(va);
    end
end
end
