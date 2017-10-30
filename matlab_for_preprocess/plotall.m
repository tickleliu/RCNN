[ mFiles] = RangTraversal( 'I:\RCNN\v0.0', '.jpg' );

samp = randi(length(mFiles), [4,4]);
[w, h] = size(samp);
for i = 1 : w
    for j = 1 : h
    %     for k = 1 : 10
    filepath = cell2mat(mFiles(samp(i, j)));
    subplot(w, h, (i - 1) * w + j)
    img = imread(filepath);
    image(img);
    end
end

