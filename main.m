% function [net_fc, info_fc, imdb1, opts] = main()

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
[net_fc, info_fc, imdb1, opts] = cnn_regression(...
    'expDir', 'data/regression', ...
    'batchNormalization', false);
% end
%==========================================================================
