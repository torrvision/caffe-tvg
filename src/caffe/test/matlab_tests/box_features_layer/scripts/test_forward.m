function status = test_forward(net, num_iters, is_num_dets_fixed, is_background_det, background_det_score, eps)

status = 0;

% figure out the dimensions of the unary and detections blob
unary_dims = net.blobs('unary').shape();
unary_spatial_dim = unary_dims(1:2);
dets_dims = net.blobs('detections').shape();
num_dets = dets_dims(3);

for j = 1:num_iters
    % if simulate varying num_dets, then we set num_dets to a random number
    % between [1, dets_dim(3)]
    if ~is_num_dets_fixed
        num_dets = randperm(dets_dims(3), 1);
        net.blobs('detections').reshape([6, 1, num_dets, 1]);
        net.reshape();
    end
    
    % generate dummy unary
    unary = single(rand(unary_dims));
    
    % generate dummy detections
    scores = single(rand(num_dets, 1));
    classes = single(uint8(rand(num_dets, 1) * 20 + 1));
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
    box_features = output{1};
    box_features = permute(box_features, [2 1 3 4]); % permute back to Matlab's convention
    
    % calculate the expected output
    expected_output = single(zeros(unary_dims(1), unary_dims(2), unary_dims(3), num_dets+1));
    for k = 1:num_dets
        score = detections(6, 1, k);
        confidence = exp(score) / (exp(score) + exp(1-score));
        xmin = fix(detections(2, 1, k)) + 1;
        ymin = fix(detections(3, 1, k)) + 1;
        xmax = fix(detections(4, 1, k)) + 1;
        ymax = fix(detections(5, 1, k)) + 1;
        
        expected_output(ymin:ymax, xmin:xmax, :, k+1) = unary(ymin:ymax, xmin:xmax, :, 1) * confidence;
        
    end
    
    if is_background_det
        confidence = background_det_score;
        xmin = 1;
        xmax = unary_spatial_dim(2);
        ymin = 1;
        ymax = unary_spatial_dim(1);
        
        expected_output(ymin:ymax, xmin:xmax, :, 1) = unary(ymin:ymax, xmin:xmax, :, 1) * confidence;
        
    end
    
    % check against the actual output
    diff = abs(expected_output - box_features);
    status = status + nnz(diff > eps);
    
    if mod(j, 10) == 0
        fprintf('[%s] Iter %d/%d: status: %d.\n', char(datetime), j, num_iters, status);
    end
end

end