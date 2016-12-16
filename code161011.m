clear;
MCN_DIR_BASE = 'C:\Workspace\matconvnet-1.0-beta23';
MCN_DIR_PT_MODELS = 'C:\Workspace\image processing';
MCN_PT_MODEL_FILE = 'imagenet-googlenet-dag.mat';
MCN_DIR_IMAGES = 'C:\Workspace\image';
% MCN_IS_COMPILED = true;
MCN_DIR_DATA = { 'images.jpg', 'KakaoTalk_20161010_204226693.jpg'};

cnn_path = fullfile(MCN_DIR_PT_MODELS, MCN_PT_MODEL_FILE);

net = dagnn.DagNN.loadobj(load(cnn_path));
net.mode = 'test';

%분류할 이미지 불러오기

num_images = length(MCN_DIR_DATA);
ims = cell(1, num_images);
ims_= cell(1, num_images);

for i = 1:num_images
    ims{i} = imread(fullfile(MCN_DIR_IMAGES, MCN_DIR_DATA{i}));
    ims_{i} = single(ims{i});
    ims_{i} = imresize(ims_{i}, net.meta.normalization.imageSize(1:2));
    ims_{i} = bsxfun(@minus, ims_{i}, net.meta.normalization.averageImage);
end

bestScore = zeros(1,num_images);
bestClass = zeros(1,num_images);
for i = 1:num_images
    net.eval({'data', ims_{i}});
    scores = net.vars(net.getVarIndex('prob')).value;
    scores = squeeze(gather(scores));
    [bestScore(i), bestClass(i)] = max(scores);
end

for i = 1:num_images
    figure(i); clf; imshow(ims{i});
    title(sprintf('%s (%d), score %.3f',...
        net.meta.classes.description{bestClass(i)},...
        bestClass(i),...
        bestScore(i)));
end