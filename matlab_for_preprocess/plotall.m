% This is a simple script to plot some diagrams.
% It assumes you have already run inpaint, storing into the 
% variables i1,i2,i3,c,d, like so:
%
% [i1,i2,i3,c,d]=inpaint('bw0.png','bw2.png',[0 255 0]);

figure;
subplot(131);image(uint8(i2)); title('Original image');
subplot(132);image(uint8(i3)); title('Fill region');
subplot(133);image(uint8(i1)); title('Inpainted image');

figure;
subplot(121);imagesc(c); title('Confidence term');
subplot(122);imagesc(d); title('Data term');


