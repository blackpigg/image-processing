function imdb = getregressionImdb(opts)
%==========================================================================
if nargin < 1
    opt = [];
end

inputsize = 39;

% Preapre the imdb structure, returns image data with mean image subtracted
load('trainImageList.mat')

data = zeros(inputsize, inputsize, 1, 10000);
temp = zeros(10,10000);
for i = 1 : 10000
    data(:,:,1,i) = cropimgs(i).resizedimg;
    temp(:,i)  = cropimgs(i).ftpoints;
end
dataMean = single(mean(data));

data = single(data);
labels = double(reshape(temp, 10, 10000));

set      = [ones(1,7000) 3*ones(1, 2000) ones(1,1000)];

data     = bsxfun(@minus, data, dataMean) ;

imdb.images.data      = data;
imdb.images.data_mean = dataMean;
imdb.images.labels    = labels;   
imdb.images.set       = set;


% 
imdb.meta.sets    = {'train', 'val', 'test'};
% imdb.meta.classes = ...
%     arrayfun(@(x)sprintf('%d',x), 0:9, 'uniformoutput', false);

save imdb3.mat imdb -v7.3
end

