addpath ./scripts
caffe.reset_all();

batch_size = 16; % make sure this is the same as the prototxt

%% Test transform-dependent class labels
prototxt = 'data/test_data_class_layer_transform_dependent.prototxt';
net = caffe.Net(prototxt, 'test');
output = net.forward({});
labels = output{3};
class_labels = output{1};
test_class_labels_transform_dependent(labels, class_labels, batch_size);

caffe.reset_all();

%% Test transform-independent class labels
prototxt = 'data/test_data_class_layer_transform_independent.prototxt';
list = 'data/list_label_test_data_class_layer.txt';
label_paths = importdata(list);
net = caffe.Net(prototxt, 'test');
output = net.forward({});
class_labels = output{1};
test_class_labels_transform_independent(class_labels, batch_size, label_paths);

caffe.reset_all();