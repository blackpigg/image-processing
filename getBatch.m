function fn = getBatch(opts)
%==========================================================================
switch lower(opts.networkType)
    case 'simplenn'
        fn = @(x, y) getSimpleNNBatch(x, y);
    case 'dagnn'
        bopts = struct('numGpus', numel(opts.train.gpus));
        fn = @(x, y) getDagNNBatch(bopts, x, y);
end
end


%==========================================================================
function [images, labels] = getSimpleNNBatch(imdb, batch)
%==========================================================================
images = imdb.images.data(:, :, :, batch);
labels = imdb.images.labels(1:10, batch);
end


%==========================================================================
function inputs = getDagNNBatch(opts, imdb, batch)
%==========================================================================
images = imdb.images.data(:, :, :, batch);
labels = imdb.images.labels(1, batch);
if opts.numGpus > 0
    images = gpuArray(images);
end
    inputs = {'input', images, 'label', labels};
end