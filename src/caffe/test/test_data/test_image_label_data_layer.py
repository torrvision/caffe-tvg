import sys
from PIL import Image
import numpy as np

caffe_path = '../../../../python/';
sys.path.insert(0, caffe_path)
import caffe

from voc_colour_map import voc_colour_map

#palette = [[0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],[64,128,0],[192,128,0],[64,0,128],[192,0,128],[64,128,128],[192,128,128],[0,64,0],[128,64,0],[0,192,0],[128,192,0],[0,64,128]]
palette = voc_colour_map()

output_dir = 'output/'
import shutil
import os
import errno
try:
    os.mkdir(output_dir)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise exc
    pass

prototxt = 'test_data_layer.prototxt';
net = caffe.Net(prototxt, caffe.TRAIN);
shutil.copyfile(prototxt, os.path.join(output_dir, prototxt) ) 
sys.stdout.flush()

for i in range(20):
   print i,
   sys.stdout.flush()
   out = net.forward();

   image = out['data'][0]
   image = image.transpose( (1,2,0) )
   image = image[:,:, (2,1,0)]  # BGR -> RGB
   pil_image = Image.fromarray(image.astype(np.uint8))

   label = out['label'][0][0]
   label_image = Image.fromarray(label.astype(np.uint8))
   label_map_image = Image.fromarray(label.astype(np.uint8))
   label_map_image.putpalette(palette)

   pil_image.save( os.path.join(output_dir,'image', str(i) + '.jpg') )
   label_image.save( os.path.join(output_dir,'label', str(i) + '.png') )
   label_map_image.save( os.path.join(output_dir,'label_map', str(i) + '.png') )