function [label, vis_im, success_status] = test_invalid_mode_remove(net, cmap)
output = net.forward({});
data = output{1};
label = output{4};
detections = output{3};
success_status = true;

% 1) check that the indices in the label are consecutive
% by checking whether all integers between the range
% [min_present_id, max_present_id] are present
present_ids = setdiff(unique(label(:)), [0, 255]);
min_present_id = min(present_ids);
max_present_id = max(present_ids);
check1 = all(ismember(min_present_id:max_present_id, present_ids));
success_status = success_status && check1;
if ~check1
    missing_ids = setdiff(min_present_id:max_present_id, present_ids);
    fprintf('Indices are not consecutive; missing %s\n', num2str(missing_ids));
end

% 2) check that the max indice in the label is equal to number of detections
check2 = min_present_id == 1;
success_status = success_status && check2;
if ~check2
    fprintf('First instance id (%d) is not 1\n', min_present_id);
end
check3 = max_present_id == numel(detections)/6;
success_status = success_status && check3;
if ~check3
    fprintf('Max instance id (%d vs %d) is not equal to number of dets\n',...
    max_present_id, numel(detections)/6);
end

% 3) Visualise
vis_im = double(squeeze(data(:,:,:,1)))/255;
vis_im = permute(vis_im, [2 1 3]);
vis_im = vis_im(:,:,[3 2 1]);
label = permute(uint8(label), [2 1 3]);
for j = 1:size(detections, 3)
    
    %det_label = detections(1,1,j,1);
    x_start = detections(2,1,j,1) + 1;
    y_start = detections(3,1,j,1) + 1;
    x_end = detections(4,1,j,1) + 1;
    y_end = detections(5,1,j,1) + 1;
    
    width = x_end - x_start;
    height = y_end - y_start;
    
    vis_im = insertShape(vis_im, 'Rectangle', [x_start, y_start, width, height],...
        'LineWidth', 3, 'Color', cmap(j+1,:));
    
    textstr = sprintf('Ins id: %d', j);
    
    vis_im = insertText(vis_im,[x_start, y_start],textstr,'FontSize',18,'BoxColor',...
        cmap(j+1,:),'BoxOpacity',0.4,'TextColor','white', 'AnchorPoint', 'LeftBottom');
    
end
% figure;
% subplot(1,2,1); imshow(label, cmap);
% subplot(1,2,2); imshow(vis_im);

end