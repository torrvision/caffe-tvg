import sys
from PIL import Image
import numpy as np

caffe_path = '../../../../python/'
sys.path.insert(0, caffe_path)
import caffe

from voc_colour_map import voc_colour_map

palette = voc_colour_map()

output_dir = 'output/'
import shutil
import os
import errno

import ipdb as pdb

try:
    os.mkdir(output_dir)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise exc
    pass

prototxt = 'test_data_index_layer_three_channel.prototxt'
net = caffe.Net(prototxt, caffe.TRAIN)
shutil.copyfile(prototxt, os.path.join(output_dir, prototxt) ) 
sys.stdout.flush()

print "Starting"
for i in range(20):
   sys.stdout.flush()
   print i,
   sys.stdout.flush()
   out = net.forward()

   image = out['data'][0]
   image = image.transpose( (1,2,0) )
   image = image[:,:, (2,1,0)]  # BGR -> RGB
   pil_image = Image.fromarray(image.astype(np.uint8))

   label_all = out['label'][0]
   for c in range(label_all.shape[0]):
      label = label_all[c]
      label_image = Image.fromarray(label.astype(np.uint8))
      label_map_image = Image.fromarray(label.astype(np.uint8))
      label_map_image.putpalette(palette)
      label_image.save( os.path.join(output_dir,'label', str(i) + '_' + str(c) + '.png') )
      label_map_image.save( os.path.join(output_dir,'label_map', str(i) + '_' + str(c) + '.png') )

   pil_image.save( os.path.join(output_dir,'image', str(i) + '.jpg') )
   
   index = out['index'][0].ravel()
   index = index.astype(np.int32)

   sys.stdout.flush()
   print index
   sys.stdout.flush()
