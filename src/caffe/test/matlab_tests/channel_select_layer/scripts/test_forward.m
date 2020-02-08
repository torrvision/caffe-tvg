function status = test_forward(net, num_iters, is_num_dets_fixed, eps)

status = 0;

% figure out the dimensions of the unary and detections blob
unary_dims = net.blobs('unary').shape();
unary_spatial_dim = unary_dims(1:2);
label_space = unary_dims(3);
unary_num_dets = unary_dims(4) - 1;
dets_dims = net.blobs('detections').shape();
num_dets = dets_dims(3);

assert(num_dets == unary_num_dets);

for j = 1:num_iters
    % if simulate varying num_dets, then we set num_dets to a random number
    % between [1, dets_dim(3)]
    if ~is_num_dets_fixed
        num_dets = randperm(dets_dims(3), 1);
        net.blobs('detections').reshape([6, 1, num_dets, 1]);
        net.blobs('unary').reshape([unary_spatial_dim, label_space, num_dets + 1]);
        net.reshape();
    end
    
    % generate dummy unary
    unary = single(rand(unary_dims(1), unary_dims(2), label_space, num_dets + 1));
    
    % generate dummy detections
    scores = single(rand(num_dets, 1));
    classes = single(floor(rand(num_dets, 1) * (label_space - 1) + 1)); % generates in range [1, 20]
    bboxes = single(bsxfun(@times, rand(num_dets, 4), [unary_spatial_dim, unary_spatial_dim] - 1));
    for k = 1:num_dets
        xmin = bboxes(k, 1);
        ymin = bboxes(k, 2);
        xmax = bboxes(k, 3);
        ymax = bboxes(k, 4);
        if xmax < xmin
            bboxes(k, 1) = xmax;
            bboxes(k, 3) = xmin;
        end
        if ymax < ymin
            bboxes(k, 2) = ymax;
            bboxes(k, 4) = ymin;
        end
    end
    detections = reshape([classes'; bboxes'; scores'], 6, 1, num_dets);
    
    % run network forward
    input = {permute(unary, [2 1 3 4]), detections}; % permute to Caffe's dimension convention
    output = net.forward(input);
    selected_channels = output{1};
    selected_channels = permute(selected_channels, [2 1 3 4]); % permute back to Matlab's convention
    
    % calculate the expected output
    expected_output = single(zeros(unary_dims(1), unary_dims(2), 1, num_dets+1));
    for k = 1:num_dets
        det_label = detections(1, 1, k);
        expected_output(:, :, 1, k+1) = unary(:, :, det_label+1, k+1);
    end
    
    expected_output(:, :, 1, 1) = unary(:, :, 1, 1);
    
    % check against the actual output
    diff = abs(expected_output - selected_channels);
    status = status + nnz(diff > eps);
    
    if mod(j, 10) == 0
        fprintf('[%s] Iter %d/%d: status: %d.\n', char(datetime), j, num_iters, status);
    end
end

end