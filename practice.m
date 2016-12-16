function [net_fc, info_fc, imdb1, imdb2] = practice()
point = [1,1;16,16];
% start_point,end_point
% start_point : 분할하는 위치의 시작 주소 [y,x]
% end_point : 분할하는 위치의 끝 주소 [y,x]

% 본 코드는 MatConvNet 라이브러리 내 cnn_mnist 예제를 참고하여 만들어졌음.

clear;
% [주목!] 아래 경로는 본인의 환경에 맞춰 설정해주세요.
MCN_DIR_BASE = 'c:/Workspace/matconvnet';

% ManConvNet 셋업
run(fullfile(MCN_DIR_BASE, 'matlab/vl_setupnn'));
disp('vl_setup is completed');

% inputsize = 16;
%--------------------------------------------------------------------------
% 뉴럴 네트워크 생성 (Batch normalization 유/무 총 2개)
%--------------------------------------------------------------------------
[net_fc, info_fc, imdb1] = cnn_mnist(...
    'expDir', 'data/mnist-baseline', ...
    'batchNormalization', false);

[net_bn, info_bn, imdb2] = cnn_mnist(...
    'expDir', 'data/mnist-bnorm', ...
    'batchNormalization', true);
% 
%--------------------------------------------------------------------------
% Validation error 도시
%--------------------------------------------------------------------------
figure(1); clf;
%--------------------
subplot(1, 2, 1);
semilogy([info_fc.val.objective]', 'o-'); hold on;
semilogy([info_bn.val.objective]', '+--');
xlabel('Training samples [x 10^3]');
ylabel('energy');
grid on;
h = legend('BSLN', 'BNORM');
set(h, 'color', 'none');
title('objective');
%--------------------
subplot(1, 2, 2);
plot([info_fc.val.top1err]', 'o-'); hold on;
plot([info_fc.val.top5err]', 's-');
plot([info_bn.val.top1err]', '+--');
plot([info_bn.val.top5err]', '*--');
h = legend('BSLN-val', 'BSLN-val-5', 'BNORM-val', 'BNORM-val-5');
grid on;
xlabel('Training samples [x 10^3]');
ylabel('error');
set(h, 'color', 'none');
title('error');
%--------------------
drawnow;
end


%==========================================================================
function [net, info, imdb] = cnn_mnist(varargin)
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
    [net] = cnn_mnist_init(...
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
    point = [1,1;16,16];
    imdb = load(opts.imdbPath);
    data = zeros(16, 16, 60e3);
    for i = 1 : 60e3
        data(:,:,i) = imdb.images.data(point(1,1):point(2,1),point(1,2):point(2,2),i);
    end
    imdb.images.data = data;
    imdb.images.data_mean = imdb.images.data_mean(point(1,1):point(2,1),point(1,2):point(2,2));
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


%==========================================================================
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
labels = imdb.images.labels(1, batch);
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


%==========================================================================
function imdb = getMnistImdb(opts)
%==========================================================================
inputsize = 16;
point = [1 1; 16 16];

% Preapre the imdb structure, returns image data with mean image subtracted
files = {'train-images-idx3-ubyte', ...
         'train-labels-idx1-ubyte', ...
         't10k-images-idx3-ubyte', ...
         't10k-labels-idx1-ubyte'};

if ~exist(opts.dataDir, 'dir')
    mkdir(opts.dataDir);
end

for i = 1:4
    if ~exist(fullfile(opts.dataDir, files{i}), 'file')
        url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i});
        fprintf('downloading %s\n', url);
        gunzip(url, opts.dataDir);
    end
end

f  = fopen(fullfile(opts.dataDir, 'train-images-idx3-ubyte'), 'r');
x1 = fread(f, inf, 'uint8');    
temp = reshape(x1(17:end), 28, 28, 60e3);
temp_img = zeros(inputsize, inputsize, 60e3);
for i = 1 : 60e3
    temp_img(:,:,i) = temp(point(1,1):point(2,1),point(1,2):point(2,2),i);
end
x1 = permute(temp_img, [2 1 3]) ;
fclose(f);

f  = fopen(fullfile(opts.dataDir, 't10k-images-idx3-ubyte'), 'r');
x2 = fread(f, inf, 'uint8');
temp = reshape(x2(17:end), 28, 28, 60e3);
temp_img = zeros(inputsize, inputsize, 60e3);
for i = 1 : 60e3
    temp_img(:,:,i) = temp(point(1,1):point(2,1),point(1,2):point(2,2),i);
end
x2 = permute(temp_img, [2 1 3]);
fclose(f);

f  = fopen(fullfile(opts.dataDir, 'train-labels-idx1-ubyte'), 'r');
y1 = fread(f,inf,'uint8');
y1 = double(y1(9:end)') + 1;
fclose(f);

f  = fopen(fullfile(opts.dataDir, 't10k-labels-idx1-ubyte'), 'r');
y2 = fread(f,inf,'uint8');
y2 = double(y2(9:end)') + 1;
fclose(f);

set      = [ones(1, numel(y1)) 3*ones(1, numel(y2))];
data     = single(reshape(cat(3, x1, x2), inputsize, inputsize, 1, []));
dataMean = mean(data(:, :, :, set == 1), 4);
data     = bsxfun(@minus, data, dataMean) ;

imdb.images.data      = data;
imdb.images.data_mean = dataMean;
imdb.images.labels    = cat(2, y1, y2);
imdb.images.set       = set;

imdb.meta.sets    = {'train', 'val', 'test'};
imdb.meta.classes = ...
    arrayfun(@(x)sprintf('%d',x), 0:9, 'uniformoutput', false);

end


%==========================================================================
function net = cnn_mnist_init(varargin)
%==========================================================================
inputsize = 16;

% CNN_MNIST_LENET Initialize a CNN similar for MNIST
opts.batchNormalization = true;
opts.networkType        = 'simplenn';
opts                    = vl_argparse(opts, varargin);

% control random number generation (shuffling method and its seed)
rng('default'); rng(0)

% 네트워크 정의
f = 1/100;
net.layers = {};
net.layers{end+1} = struct(...
    'type','conv',...
    'weights', {{f*randn(5, 5, 1, 20, 'single'), zeros(1, 20,'single')}},...
    'stride', 1, ...
    'pad',0);

net.layers{end+1} = struct(...
    'type','pool',...
    'method', 'max',...
    'pool',[2,2],...
    'stride', 2, ...
    'pad',0);

net.layers{end+1} = struct(...
    'type','conv',...
    'weights', {{f*randn(5, 5, 20, 50, 'single'), zeros(1, 50,'single')}},...
    'stride', 1, ...
    'pad',0);

net.layers{end+1} = struct(...
    'type','pool',...
    'method', 'max',...
    'pool',[2,2],...
    'stride', 2, ...
    'pad',0);

net.layers{end+1} = struct(...
    'type','conv',...
    'weights', {{f*randn(4, 4, 50, 500, 'single'), zeros(1, 500,'single')}},...
    'stride', 1, ...
    'pad',0);

net.layers{end+1} = struct(...
    'type', 'relu');

net.layers{end+1} = struct(...
    'type','conv',...
    'weights', {{f*randn(1, 1, 50, 10, 'single'), zeros(1, 10,'single')}},...
    'stride', 1, ...
    'pad',0);

net.layers{end+1} = struct(...
    'type', 'softmaxloss');

% TODO: Convolutional Layer
% TODO: Pooling Layer
% TODO: Convolutional Layer
% TODO: Pooling Layer
% TODO: Convolutional Layer
% TODO: ReLU Layer
% TODO: Convolutional Layer
% TODO: Softmax Loss Layer

% optionally switch to batch normalization
if opts.batchNormalization
    net = insertBnorm(net, 1); % 1, 3, 5번째 layer 뒤에 노멀라이제이션 레이어 삽입
    net = insertBnorm(net, 4);
    net = insertBnorm(net, 7);
end

% TODO: Meta parameters

net.meta.inputSize              = [inputsize inputsize 1];
net.meta.trainOpts.learningRate = 0.001;
net.meta.trainOpts.numEpochs    = 20;
net.meta.trainOpts.batchSize    = 100;

% Fill in defaul values
net = vl_simplenn_tidy(net);

% network type 대응: DagNN로 구조 변경
switch lower(opts.networkType)
    case 'simplenn'
        % done
    case 'dagnn'
        net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true);
        net.addLayer(...
            'top1err', dagnn.Loss('loss', 'classerror'), ...
            {'prediction', 'label'}, 'error');
        net.addLayer(...
            'top5err', dagnn.Loss('loss', 'topkerror', ...
            'opts', {'topk', 5}), {'prediction', 'label'}, 'top5err');
    otherwise
        assert(false);
end
end


%==========================================================================
function net = insertBnorm(net, l)
%==========================================================================
assert(isfield(net.layers{l}, 'weights'));
ndim  = size(net.layers{l}.weights{1}, 4);

% TODO: Batch Noramalization Layer
layer = struct(...
    'type','bnorm',...
    'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1,'single')}},...
    'learningRate', [1 1 0.05], ...
    'weightDecay',[0 0]);

net.layers{l}.biases =[];
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end));
end


%()()
%('')HAANJU.YOO
