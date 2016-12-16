function [net, info, imdb, opts] = cnn_regression(varargin)
%==========================================================================

%--------------------------------------------------------------------------
% 네트워크 설정
%--------------------------------------------------------------------------
% network arguments: 구조 관련
opts.batchNormalization = false;
opts.network            = [] ;
opts.networkType        = 'simplenn';
[opts, varargin]        = vl_argparse(opts, varargin);

opts.expDir      = fullfile(vl_rootnn, 'data', 'regression' );
[opts, varargin] = vl_argparse(opts, varargin);
opts.dataDir     = fullfile(vl_rootnn, 'data', 'train');
opts.imdbPath    = fullfile(opts.expDir, 'imdb2.mat');
opts.train       = struct();
opts             = vl_argparse(opts, varargin);

% network arguments: GPU 사용 여부
if ~isfield(opts.train, 'gpus')
    opts.train.gpus = [];
end;

% network 정의
if isempty(opts.network)
    [net] = cnn_regression_init(...
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
    load(opts.imdbPath);
else
    imdb = getregressionImdb(opts); % <= 파일 download
    mkdir(opts.expDir);
    save(opts.imdbPath, '-struct', 'imdb');
end
% net.meta.classes.name = ...
%     arrayfun(@(x)sprintf('%d',x), 1:10, 'UniformOutput', false);
trainfn = @cnn_train;
[net, info] = trainfn(...
    net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, ...
    net.meta.trainOpts, ...
    opts.train, ...
    'val', find(imdb.images.set == 3));
end


