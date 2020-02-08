function test_class_labels_transform_dependent(labels, class_labels, batch_size)

for k = 1:batch_size
    class_label = class_labels(:,:,:,k);
    label = labels(:,:,:,k);
    unique_labels = unique(label(:));
    unique_class_labels = find(class_label) - 1;
    unique_labels_minus_ignore = setdiff(unique_labels, [255]);
    bad_labels = setxor(unique_class_labels, unique_labels_minus_ignore);
    if ~isempty(bad_labels)
        display(bad_labels);
        error('unexpected issue with batch %d\n', k);
    end
end

fprintf('Test for transform dependent class labels passed.\n');

end