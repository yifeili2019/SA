import rospy
from sensor_msgs.msg import Image
#from core import CvBridge, CvBridgeError
import colorsys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:

    sys.path.remove(ros_path)

import cv2
import time

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import image_preporcess

from gdp.msg import position_detection
from gdp.msg import multipositions

#from cv_bridge import CvBridge, CvBridgeError
class YOLO(object):
    _defaults = {
        #"model_path": 'logs/trained_weights_final.h5',
        "model_path": 'model_data/yolo-drone_weights.h5',
        "anchors_path": 'model_data/tiny_yolo_anchors.txt',
        "classes_path": 'model_data/class.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "text_size" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes
    
    def calDis(self,top,left,bottom,right,depth_scale,depthImage):
        mid_h = (bottom-top)/2+top
        mid_v = (right-left)/2+left
        
        ratio = 0.3

        length = (right-left)*ratio
        width = (bottom-top)*ratio

        start = [mid_h-width/2,mid_v-length/2]
        end = [start[0]+width,start[1]+length]

        distance = 0 
        count = 1

        for i in range(int(start[1]),int(end[1])):
            for j in range(int(start[0]),int(end[0])):
                depth = depthImage[j,i].astype(float)
                distance_temp = depth * depth_scale
                if distance_temp<20 and distance_temp>0.1:
                    distance += distance_temp
                    count += 1
        
        distance = distance/count
     
        return distance


    def detect_image(self, image,depthImage,depth_scale):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = image_preporcess(np.copy(image), tuple(reversed(self.model_image_size)))
            image_data = boxed_image

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],#[image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        

        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        thickness = (image.shape[0] + image.shape[1]) // 600
        fontScale=1
        ObjectsList = []
        position_list = multipositions()
        #bridge = CvBridge()
        #Image = bridge.cv2_to_imgmsg(image,"bgr8")
        #position_list.image = Image
        for i, c in reversed(list(enumerate(out_classes))):
            id_ = "target"+str(i)

            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            #label = '{}'.format(predicted_class)
            scores = '{:.2f}'.format(score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

            mid_h = (bottom-top)/2+top
            mid_v = (right-left)/2+left

            ratio = 0.3

            length = (right-left)*ratio
            width = (bottom-top)*ratio

            start = [mid_h-width/2,mid_v-length/2]
            #end = [mid_h+width/2,mid_v+length/2]
            end = [start[0]+width,start[1]+length]

            #mid_h1 = mid_h+10
            #mid_

            

            position_temp = position_detection()
            position_temp.id = id_
            position_temp.top = top
            position_temp.left = left
            position_temp.bottom = bottom
            position_temp.right = right

             
            #depth = depthImage[int(mid_h),int(mid_v)].astype(float)
            #distance = depth * depth_scale
            distance = self.calDis(top,left,bottom,right,depth_scale,depthImage)

            position_temp.distance = distance


            distance = round(distance,2)
            distance = str(distance)

            #position_temp.distance = distance

            position_list.lists.append(position_temp)

            #depth = depthImage[int(mid_h),int(mid_v)].astype(float)
            #distance = depth * depth_scale
            #distance = round(distance,2)
            #distance = str(distance)

            #position_temp.real_z = distance
            #print(distance)



            # put object rectangle
            cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], thickness)
            cv2.rectangle(image, (int(start[1]), int(start[0])), (int(end[1]),int(end[0])), self.colors[c], thickness)
            #cv2.rectangle(image, (int(end[0]), int(end[1])), (640,480), self.colors[c], thickness)


            # get text size
            (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, 1)

            # put text rectangle
            cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), self.colors[c], thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label+" "+distance+"m", (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, (0, 0, 0), 1)

            # add everything to list
            ObjectsList.append([top, left, bottom, right, mid_v, mid_h, label, scores])

        return image, ObjectsList, position_list

    def close_session(self):
        self.sess.close()

    def detect_img(self, image,depthImage,depthScale):
        #image = cv2.imread(image, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image_color = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        r_image, ObjectsList, position_list = self.detect_image(original_image_color,depthImage,depthScale)
        return r_image, ObjectsList, position_list
