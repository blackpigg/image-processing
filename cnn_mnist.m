function [net, info] = cnn_mnist(varargin)
%==========================================================================

%--------------------------------------------------------------------------
% ��Ʈ��ũ ����
%--------------------------------------------------------------------------
% network arguments: ���� ����
opts.batchNormalization = false;
opts.network            = [] ;
opts.networkType        = 'simplenn';
[opts, varargin]        = vl_argparse(opts, varargin);

% network arguments: ���� ����
sfx = opts.networkType;
if opts.batchNormalization
    sfx = [sfx '-bnorm'];
end
opts.expDir      = fullfile(vl_rootnn, 'data', ['mnist-baseline-' sfx]);
[opts, varargin] = vl_argparse(opts, varargin);
opts.dataDir     = fullfile(vl_rootnn, 'data', 'mnist');
opts.imdbPath    = fullfile(opts.expDir, 'imdb.mat');
opts.train       = struct();
opts             = vl_argparse(opts, varargin);

% network arguments: GPU ��� ����
if ~isfield(opts.train, 'gpus')
    opts.train.gpus = [];
end;

% network ����
if isempty(opts.network)
    net = cnn_mnist_init(...
        'batchNormalization', opts.batchNormalization, ...
        'networkType', opts.networkType);
else
    net = opts.network;
    opts.network = [];
end


%--------------------------------------------------------------------------
% ������ �غ��ϱ�
%--------------------------------------------------------------------------
if exist(opts.imdbPath, 'file') % <= ������ �����ϴ��� Ȯ��
    imdb = load(opts.imdbPath);
else
    imdb = getMnistImdb(opts); % <= ���� download
    mkdir(opts.expDir);
    save(opts.imdbPath, '-struct', 'imdb');
end
net.meta.classes.name = ...
    arrayfun(@(x)sprintf('%d',x), 1:10, 'UniformOutput', false);


%--------------------------------------------------------------------------
% �н�
%--------------------------------------------------------------------------
switch opts.networkType
    case 'simplenn', trainfn = @cnn_train;
    case 'dagnn',    trainfn = @cnn_train_dag;
end

[net, info] = trainfn(...
    net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, ...
    net.meta.trainOpts, ...
    opts.train, ...
    'val', find(imdb.images.set == 3));
end
