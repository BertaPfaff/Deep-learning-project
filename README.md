# Pneumonia Image Detection
This repository contains a code implementation for detecting pneumonia in X-ray images using deep learning techniques. The code processes the input dataset, generates new images, creates and trains a model to classify the images, and evaluates the performance of the model. The code also supports using a pre-trained InceptionV3 model for transfer learning.

## Functions
The following functions are implemented in the code:

> - `image_exploration`: Creates a list of tuples containing class, path, and size for each image in the given path.
> - `images_creator`: Generates a specified number of new images, creating a certain number of images for each image in the input list of tuples.
> - `image_generator`: Generates new images from a given image using data augmentation techniques.
> - `format_example`: Returns an image that is reshaped to IMG_SIZE.
> - `dataset_creator`: Creates two lists of tuples with class and formatted image arrays, one for normal images and one for pneumonia images.
> - `dataset_loader`: Loads image data from a list of tuples and returns two arrays, one for images and one for labels.
> - `convert_RGB' : Converts a grayscale image to an RGB image by duplicating the number of channels and pixel values.

 ## Usage
1. Set the `dir_images` variable to the directory containing your dataset of X-ray images.

2. Create a list of tuples with class, path, and size for each image in the dataset:

```python
images_list = image_exploration(dir_images)
```

3. Generate new images from the dataset:

```python
images_creator(images_list)
```
4. Combine the original images and the newly generated images into a single list, and create a dataset:
```python
ds_list = images_list + new_images_list
ds_NORMAL, ds_PNEUMO = dataset_creator(ds_list)
ds_total = ds_NORMAL + ds_PNEUMO
```
5. Load the dataset and split it into training, validation, and test sets:
```python
X, y = dataset_loader(ds_total)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_size, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=42)
```
6. Train and evaluate the custom model:
```python
#Train the model
history = model_PROP.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

#Evaluate the model
accuracy_train = model_PROP.history.history['accuracy'][-1]
loss0, accuracy_test = model_PROP.evaluate(X_test, y_test)

print(f'ACC. TEST: {accuracy_test} -- ACC. TRAIN: {accuracy_train} ')
```
7. Train and evaluate the InceptionV3 model:
```python
#Train the model
history = model_V3.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

#Evaluate the model
accuracy_train = model_V3.history.history['accuracy'][-1]
loss0, accuracy_test = model_V3.evaluate(X_test, y_test)

print(f'ACC. TEST: {accuracy_test} -- ACC. TRAIN: {accuracy_train} ')
```
## Dependencies
- TensorFlow
- NumPy
- OpenCV
- Pillow
- Matplotlib
- tqdm
- scikit-learn
