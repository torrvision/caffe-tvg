clear all;
% Test settings
gpu_id = -1; % -1 to test CPU; non-negative integer to test GPU implementation
num_iters = 100; % number of iterations to test forward/backward passes
eps = 1e-5; % allowed absolute difference between the expected and actual output

% Initialisation
try
caffe.reset_all();
catch
    addpath ../../../../../matlab
    try
        caffe.reset_all();
    catch
        error('Unexpected error\n');
    end
end
addpath ./scripts

if gpu_id >=0
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
else
    caffe.set_mode_cpu();
end

definition = 'data/channel_select_layer.prototxt';

net = caffe.Net(definition, 'test');

% --- Test starts ---

% Test forward (constant num of detections)
is_num_dets_fixed = true;
status(1) = test_forward(net, num_iters, is_num_dets_fixed, eps);

% Test forward (varying num of detections)
is_num_dets_fixed = false;
status(2) = test_forward(net, num_iters, is_num_dets_fixed, eps);

% Test backward (constant num of detections)
is_num_dets_fixed = true;
status(3) = test_backward(net, num_iters, is_num_dets_fixed, eps);

% Test backward (varying num of detections)
is_num_dets_fixed = false;
status(4) = test_backward(net, num_iters, is_num_dets_fixed, eps);

% --- Test ends ---

num_total = numel(status);
num_success = num_total - sum(logical(status));
fprintf('%d/%d tests passed.\n', num_success, num_total);
