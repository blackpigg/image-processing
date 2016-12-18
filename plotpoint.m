function []= plotpoint(index)

index = 7021;
index2 = (((index-7000)-mod((index-7000),100))/100)+1;
index3 = mod((index-7000),100);
img = imdb1.images.data(:,:,:,index);
img_norm = 1.0 /(max(max(img)) - min(min(img))) * (img - min(min(img)));
img_rgb = repmat(img_norm, [1 1 3]);
imshow(img_rgb)
 rectangle('position', [imdb1.images.labels(1,index), imdb1.images.labels(2,index), 1, 1], 'EdgeColor', [1 0 0], 'FaceColor', [1 0 0]);
 rectangle('position', [imdb1.images.labels(3,index), imdb1.images.labels(4,index), 1, 1], 'EdgeColor', [1 0 0], 'FaceColor', [1 0 0]);
rectangle('position', [imdb1.images.labels(5,index), imdb1.images.labels(6,index), 1, 1], 'EdgeColor', [1 0 0], 'FaceColor', [1 0 0]);
rectangle('position', [imdb1.images.labels(7,index), imdb1.images.labels(8,index), 1, 1], 'EdgeColor', [1 0 0], 'FaceColor', [1 0 0]);
rectangle('position', [imdb1.images.labels(9,index), imdb1.images.labels(10,index), 1, 1], 'EdgeColor', [1 0 0], 'FaceColor', [1 0 0]);
rectangle('position', [result{1,index2}(1,index3), result{1,index2}(2,index3), 1, 1], 'EdgeColor', [0 1 0], 'FaceColor', [0 1 0]);
rectangle('position', [result{1,index2}(3,index3), result{1,index2}(4,index3), 1, 1], 'EdgeColor', [0 1 0], 'FaceColor', [0 1 0]);
rectangle('position', [result{1,index2}(5,index3), result{1,index2}(6,index3), 1, 1], 'EdgeColor', [0 1 0], 'FaceColor', [0 1 0]);
rectangle('position', [result{1,index2}(7,index3), result{1,index2}(8,index3), 1, 1], 'EdgeColor', [0 1 0], 'FaceColor', [0 1 0]);
rectangle('position', [result{1,index2}(9,index3), result{1,index2}(10,index3), 1, 1], 'EdgeColor', [0 1 0], 'FaceColor', [0 1 0]);
% end