import sys
from PIL import Image, ImageDraw
import numpy as np

caffe_path = '../../../../python/'
sys.path.insert(0, caffe_path)
import caffe

import voc

palette = voc.get_colour_map()

import shutil
import os
import errno

import pdb

def make_dir(dirname):
    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise exc
        pass


offsets = np.array([
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1]
])
np.set_printoptions(precision=3)


def draw_rect(draw, coords, colour, offsets):
    for i in range(offsets.shape[0]):
        offset = np.squeeze(offsets[i,:])
        offset = np.tile(offset, 2)
        coords = coords + offset
        draw.rectangle(xy=( (coords[0], coords[1]),(coords[2], coords[3]) ), outline=tuple(colour) )


def main():
    output_dir = 'output/'
    crop_size = 500;
    make_dir(output_dir)

    prototxt = 'test_data_index_detection_layer.prototxt'
    net = caffe.Net(prototxt, caffe.TRAIN)
    shutil.copyfile(prototxt, os.path.join(output_dir, prototxt) ) 
    sys.stdout.flush()

    is_success = True

    print "Starting"
    for i in range(1000):
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
            label_image.save( os.path.join(output_dir,'label_' + str(i) + '_' + str(c) + '.png') )
            label_map_image.save( os.path.join(output_dir,'label_map_' + str(i) + '_' + str(c) + '.png') )

        pil_image.save( os.path.join(output_dir,'image_' + str(i) + '.jpg') )
        
        index = out['index'][0].ravel()
        index = index.astype(np.int32)

        # Now draw the boxes on the output image too
        detections = out['detections'][0]

        draw = ImageDraw.Draw(pil_image)
        for i_det in range(detections.shape[0]):
            det = np.squeeze(detections[i_det]).astype(np.int32)
            if det.size > 1:
                # Single element of [-1] means that no valid detections are in the file
                det_class = det[0]
                coords = det[1:-1]
                colour = palette[det_class*3:(det_class+1)*3]
                draw_rect(draw, coords, colour, offsets)
                if np.sum((coords < 0)) > 0 or np.sum((coords > crop_size - 1)):
                    is_success = False
                    print index, ": No!"
                if coords[2]<coords[0] or coords[3] < coords[1]:
                    is_success = False
                    print index, ": No!"
            else:
                if det != -1:
                    raise AssertionError('det is scalar value. Should be -1 to indicate no detections. Instead, got ' + str(det))

        pil_image.save( os.path.join(output_dir,'image_box_' + str(i) + '.jpg') )

        sys.stdout.flush()
        print "Image:",i
        print index
        print detections
        print "======="
    print 'Test passed: ', is_success

if __name__ == '__main__':
    main()