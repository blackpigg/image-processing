function []= drawdata(img, groundTruth, networkOutput)

bDrawOutput = false;

if nargin > 2
    bDrawOutput = true;
end

% input image pre-processing
% img = imdb1.images.data(:,:,:,index);
img_norm = 1.0 /(max(max(img)) - min(min(img))) * (img - min(min(img)));
img_rgb = repmat(img_norm, [1 1 3]);

GTPoints = reshape(groundTruth, 2, 5);

if bDrawOutput
    NOPoints = reshape(networkOutput, 2, 5);
end
% 
% index = 7021;
% index2 = (((index-7000)-mod((index-7000),100))/100)+1;
% index3 = mod((index-7000),100);

figure; clf;
imshow(img_rgb, 'border', 'tight');
hold on;
% draw points
for i = 1:5
    x = GTPoints(1,i);
    y = GTPoints(2,i);
    rectangle('position', [x, y, 1, 1], 'EdgeColor', [1 0 0], 'FaceColor', [1 0 0]);
    
    if bDrawOutput
        x_ = NOPoints(1,i);
        y_ = NOPoints(2,i);
        rectangle('position', [x_, y_, 1, 1], 'EdgeColor', [0 1 0], 'FaceColor', [0 1 0]);
    end
end
hold off;
drawnow;

% 
% rectangle('position', [imdb1.images.labels(1,index), imdb1.images.labels(2,index), 1, 1], 'EdgeColor', [1 0 0], 'FaceColor', [1 0 0]);
%  rectangle('position', [imdb1.images.labels(3,index), imdb1.images.labels(4,index), 1, 1], 'EdgeColor', [1 0 0], 'FaceColor', [1 0 0]);
% rectangle('position', [imdb1.images.labels(5,index), imdb1.images.labels(6,index), 1, 1], 'EdgeColor', [1 0 0], 'FaceColor', [1 0 0]);
% rectangle('position', [imdb1.images.labels(7,index), imdb1.images.labels(8,index), 1, 1], 'EdgeColor', [1 0 0], 'FaceColor', [1 0 0]);
% rectangle('position', [imdb1.images.labels(9,index), imdb1.images.labels(10,index), 1, 1], 'EdgeColor', [1 0 0], 'FaceColor', [1 0 0]);
% rectangle('position', [labels(1,index3), labels(2,index3), 1, 1], 'EdgeColor', [0 1 0], 'FaceColor', [0 1 0]);
% rectangle('position', [labels(3,index3), labels(4,index3), 1, 1], 'EdgeColor', [0 1 0], 'FaceColor', [0 1 0]);
% rectangle('position', [labels(5,index3), labels(6,index3), 1, 1], 'EdgeColor', [0 1 0], 'FaceColor', [0 1 0]);
% rectangle('position', [labels(7,index3), labels(8,index3), 1, 1], 'EdgeColor', [0 1 0], 'FaceColor', [0 1 0]);
% rectangle('position', [labels(9,index3), labels(10,index3), 1, 1], 'EdgeColor', [0 1 0], 'FaceColor', [0 1 0]);
end