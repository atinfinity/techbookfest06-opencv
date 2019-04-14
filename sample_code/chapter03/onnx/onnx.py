import cv2
import numpy as np


# preprocessing
def preprocess(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])[::-1]
    stddev_vec = np.array([0.229, 0.224, 0.225])[::-1]
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[2]):
        norm_img_data[:,:,i] = (img_data[:,:,i]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data

# Load names of classes
def load_classfile(classfile):
    classes = None
    #classfile = 'synset.txt'
    with open(classfile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    return classes

# https://github.com/onnx/models/tree/master/shufflenet
model_file = 'shufflenet/model.onnx'

# load a network
net = cv2.dnn.readNetFromONNX(model_file)

# load image
image = cv2.imread('space_shuttle.jpg')
preprocessed = preprocess(image)
blob = cv2.dnn.blobFromImage(preprocessed, size=(224, 224))

# run a model
net.setInput(blob)
pred = net.forward()

# get a class with a highest score.
pred = pred.flatten()
classId = np.argmax(pred)
confidence = pred[classId]

# profiling
t, _ = net.getPerfProfile()
label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
cv2.putText(image, label, (0, 15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))

# print predicted class.
classfile = 'synset.txt'
classes = load_classfile(classfile)
label = '%s: %.4f' % (classes[classId] if classes else 'Class #%d' % classId, confidence)
cv2.putText(image, label, (0, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))

cv2.imshow("inference", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
