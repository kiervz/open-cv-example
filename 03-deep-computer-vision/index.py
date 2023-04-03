import os
import caer
import canaro 
import numpy as np
import cv2 as cv
import gc
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler

# The desired size of the images that will be used for training and testing
IMG_SIZE = (80, 80)

# Number of channels in the images, 1 for grayscale and 3 for RGB
channels = 1
# Path to the directory containing the images of the characters
char_path = r'./resources/simpsons_dataset'

char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path, char)))
    
# Sort the dictionary in descending order based on the number of images
char_dict = caer.sort_dict(char_dict, descending=True)

characters = []
count = 0

# Select the top 10 characters with the most images
for i in char_dict:
    characters.append(i[0])
    count += 1
    if count >= 10:
        break
    
# Create the training data
train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True)

# Display the first image from the training data
plt.figure(figsize=(30, 30))
plt.imshow(train[0][0], cmap='gray')
plt.show()

# Separate the feature set and labels from the training data
featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)

# Normalize the featureSet to be in range of (0, 1)
# Note: If you normalize the data, the network will be able to learn
# the features much faster than no normalizing the data
featureSet = caer.normalize(featureSet)
labels = to_categorical(labels, len(characters))

# Splitting the feature set and labels into training sets and 
# validation set with using a particular validation ratio to 20%
# and 80% will go to the training set.
x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio=.2)

# Remove or delete some of the variables that are not going to be use
del train
del featureSet
del labels
gc.collect()

BATCH_SIZE = 32
EPOCHS = 10

# Create an instance of the `ImageDataGenerator` class
datagen = canaro.generators.imageDataGenerator()

# Create a generator for the training data
train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

# Create the model using the 'createSimpsonsModel()' function from the 'canaro' library
# I'm pretty sure you are encounting this kind of error: decay is deprecated in the new Keras optimizer
# Refer to this code: https://github.com/jasmcaus/canaro/blob/7331ff718c6230173e9dac1c57c71b1ad3bd1000/canaro/models/simpsons.py#L18
model = canaro.models.createSimpsonsModel(
    IMG_SIZE=IMG_SIZE, 
    channels=channels, 
    output_dim=len(characters)
)

# Compile the model with SGD optimizer and binary crossentropy loss
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Display the summary of the model
model.summary()

# Create a list of callbacks that includes a learning rate scheduler
callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]

# Train the model using the 'fit()' method
traning = model.fit(
    train_gen, 
    steps_per_epoch=len(x_train)//BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    validation_steps=len(y_val//BATCH_SIZE),
    callbacks = callbacks_list
)

# Load a test image
test_path = r'./resources/simpsons_testset/charles_montgomery_burns_0.jpg'
# Display the test image
img = cv.imread(test_path)
plt.imshow(img)
plt.show()

def prepare(img): 
    # Converts the image from the BGR color space to grayscale using the OpenCV 
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Resizes the image to the specified IMG_SIZE using the OpenCV cv.resize() function.
    img = cv.resize(img, IMG_SIZE)
    # Reshapes the image to have the specified dimensions and 
    # number of channels (1 in this case) using the caer.reshape() function from the Caer library.
    img = caer.reshape(img, IMG_SIZE, 1)
    
    return img

# Prepare the image for prediction
prediction = model.predict(prepare(img))

# Print the predicted character name
print(f'prediction: {characters[np.argmax(prediction[0])]}')