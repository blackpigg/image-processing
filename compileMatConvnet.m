MCN_DIR_BASE = 'C:\Workspace\matconvnet-1.0-beta23';
MCN_DIR_PT_MODELS = 'C:\Workspace\image processing';
MCN_PT_MODEL_FILE = 'imagenet-googlenet-dag.mat';
MCN_DIR_IMAGES = 'C:\Workspace\image';
% MCN_IS_COMPILED = true;
MCN_DIR_DATA = {'KakaoTalk_20161010_204226693.jpg, images.jpg'};

if ~MCN_IS_COMPILED
    run(fullfile(MCN_DIR_BASE, 'matlab/vl_compilenn'))
end

run(fullfile(MCN_DIR_BASE,'matlab/vl_setupnn'))


net = load(fullfile(MCN_DIR_PT_MODELS, MCN_PT_MODEL_FILE));
net = vl_simplenn_tidy(net);

im =imread(fullfile(MCN_DIR_IMAGES,MCN_DIR_DATA));
im_=single(im);
im_=imresize(im_,net.meta.normalization.imageSize(1:2));
im_=im_-net.meta.normalization.averageImage;


