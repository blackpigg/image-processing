% function [net_fc, info_fc, imdb1, opts] = main()

clear;
% [�ָ�!] �Ʒ� ��δ� ������ ȯ�濡 ���� �������ּ���.
MCN_DIR_BASE = 'c:/Workspace/matconvnet';

% ManConvNet �¾�
run(fullfile(MCN_DIR_BASE, 'matlab/vl_setupnn'));
disp('vl_setup is completed');

% inputsize = 16;
%--------------------------------------------------------------------------
% ���� ��Ʈ��ũ ���� (Batch normalization ��/�� �� 2��)
%--------------------------------------------------------------------------
[net_fc, info_fc, imdb1, opts] = cnn_regression(...
    'expDir', 'data/regression', ...
    'batchNormalization', false);
% end
%==========================================================================
