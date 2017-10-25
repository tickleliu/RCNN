function [x1, y1, x2, y2] = delete_border(img)

img = double(img);
img_g = img(:,:,1) + img(:,:,2) + img(:,:,3);
img_g = img_g / 3;
sz = size(img_g);
paddingh = round(sz(1) * 0.15);
paddingw = round(sz(2) * 0.1);
img_g = img_g(paddingh:end,paddingw:end - paddingw);

e = edge(img_g, 'canny', [0.02]);
[x1, y1, x2, y2] = vot_edge(e, sz * 0.4);
% subplot(1,2,1)
% image(e * 10)
% subplot(1,2,2)
% image(img_g(y1:y2, x1:x2));
y1 = y1 + paddingh;
y2 = y2 + paddingh;
x1 = x1 + paddingw;
x2 = x2 + paddingw;
end

function [x1, y1, x2, y2] = vot_edge(edge, padding)
sz = size(edge);
x1_vote = zeros(1, sz(2));
for i = 1 : sz(1)
    v = find(edge(i, : ) ~= 0);
    if isempty(v) || min(v) > padding(2)
        continue
    end
    x1_vote(min(v)) = x1_vote(min(v)) + 1;
end

[index, x1] = max(x1_vote);
x2_vote = zeros(1, sz(2));
for i = 1 : sz(1)
    v = find(edge(i, : ) ~= 0);
    if isempty(v) || max(v) < sz(2) - padding(2)
        continue
    end
    x2_vote(max(v)) = x2_vote(max(v)) + 1;
end
[index, x2] = max(x2_vote);

y1_vote = zeros(1, sz(1));
for i = 1 : sz(2)
    v = find(edge(:, i ) ~= 0);
    if isempty(v) || min(v) > padding(1)
        continue
    end
    y1_vote(min(v)) = y1_vote(min(v)) + 1;
end
[index, y1] = max(y1_vote);

y2_vote = zeros(1, sz(1));
for i = 1 : sz(2)
    v = find(edge(:, i ) ~= 0);
    if isempty(v) || max(v) < sz(1) - padding(1)
        continue
    end
    y2_vote(max(v)) = y2_vote(max(v)) + 1;
end
[index, y2] = max(y2_vote);
end