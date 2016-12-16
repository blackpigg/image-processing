function [net, info] = cnn_mnist(varargin)
%==========================================================================

%--------------------------------------------------------------------------
% 네트워크 설정
%--------------------------------------------------------------------------
% network arguments: 구조 관련
opts.batchNormalization = false;
opts.network            = [] ;
opts.networkType        = 'simplenn';
[opts, varargin]        = vl_argparse(opts, varargin);

% network arguments: 폴더 관련
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

% network arguments: GPU 사용 여부
if ~isfield(opts.train, 'gpus')
    opts.train.gpus = [];
end;

% network 정의
if isempty(opts.network)
    net = cnn_mnist_init(...
        'batchNormalization', opts.batchNormalization, ...
        'networkType', opts.networkType);
else
    net = opts.network;
    opts.network = [];
end


%--------------------------------------------------------------------------
% 데이터 준비하기
%--------------------------------------------------------------------------
if exist(opts.imdbPath, 'file') % <= 파일이 존재하는지 확인
    imdb = load(opts.imdbPath);
else
    imdb = getMnistImdb(opts); % <= 파일 download
    mkdir(opts.expDir);
    save(opts.imdbPath, '-struct', 'imdb');
end
net.meta.classes.name = ...
    arrayfun(@(x)sprintf('%d',x), 1:10, 'UniformOutput', false);


%--------------------------------------------------------------------------
% 학습
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
