from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image  
from keras.models import load_model
import numpy as np
import cv2
import os


# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')


# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')    

#load model
#model = load_model('models/weights.best.ResNet50.hdf5')
def load_resnet_model():
    model = load_model('models/weights.best.ResNet50.hdf5')
    model._make_predict_function() 
    print('model loaded') # just to keep track in your server
    return model

#load dog names
with open('models/dog_names.txt') as f:
    dog_names = f.read().splitlines()

#get bottleneck features ResNet
bottleneck_features_ResNet = np.load('models/DogResnet50Data.npz')
train_ResNet = bottleneck_features_ResNet['train']
valid_ResNet = bottleneck_features_ResNet['valid']
test_ResNet = bottleneck_features_ResNet['test'] 

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def extract_Resnet50(tensor):
    from keras.applications.resnet50 import ResNet50, preprocess_input
    return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def ResNet_predict_breed(img_path):
    model = load_resnet_model()
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = model.predict(bottleneck_feature)
    #print(predicted_vector)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def human_dog_detector(imag_path):
    try:
        contains_dog = dog_detector(imag_path)
        contains_face = face_detector(imag_path)
        contains_either = contains_dog or contains_face

        if not contains_either:
            response ='The image provided does not contain a dog nor a face'
        else:
            prediction = ResNet_predict_breed(imag_path)
            response = 'Looks like a ' + prediction + '!'
        return response
    except Exception as error:
        print(error)
        return "There was an error. Please try again or use another image"
    


print(human_dog_detector('uploads/Pekingese.jpg'))