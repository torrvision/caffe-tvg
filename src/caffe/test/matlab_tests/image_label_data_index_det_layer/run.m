addpath ./scripts
caffe.reset_all();
save_dir = './output';
iter = 100;

if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end
%% Test invalid_detection_mode = REMOVE
prototxt = 'data/test_invalid_mode_remove.prototxt';
net = caffe.Net(prototxt, 'train');
load data/Cityscapescolormap.mat
num_successes = 0;
for k = 1:iter
[label, vis_im, success_status] = test_invalid_mode_remove(net, cmap);
label_save_path = fullfile(save_dir, ['remove_', num2str(k), '_label.png']);
imvis_save_path = fullfile(save_dir, ['remove_', num2str(k), '_imvis.png']);
imwrite(label, cmap, label_save_path);
imwrite(vis_im, imvis_save_path);
num_successes = num_successes + success_status;
end
fprintf('Success/Total = %d/%d\n', num_successes, iter);
if num_successes == iter
    fprintf('Test invalid_detection_mode == REMOVE: SUCCESS!\n');
end
caffe.reset_all();

%% Test invalid_detection_mode = ZERO_OUT (Not implemented yet)
% prototxt = 'data/test_invalid_mode_zero_out.prototxt';
% net = caffe.Net(prototxt, 'test');
% 
% caffe.reset_all();