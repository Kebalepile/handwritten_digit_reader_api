# Project Description: Handwritten Digits Recognition using Convolutional Neural Network

This project demonstrates the application of deep learning techniques, specifically Convolutional Neural Networks (CNNs), to recognize handwritten digits from the MNIST dataset. MNIST is a benchmark dataset in the field of machine learning, consisting of a large set of labeled images of handwritten digits ranging from 0 to 9.

## Key Features and Workflow

- **Data Preparation**: The MNIST dataset is extracted from Google Drive, where it is stored in a zip file. The dataset includes separate files for training and test images, as well as their corresponding labels.
  
- **Model Development**: Using TensorFlow and Keras in a Google Colab environment, a CNN model is constructed. This model architecture includes convolutional layers for feature extraction from the 2D image data, followed by pooling layers to reduce dimensionality, and fully connected layers for classification.
  
- **Training and Evaluation**: The model is trained on the training dataset and evaluated on the test dataset to measure its accuracy in recognizing unseen handwritten digits. The training process involves optimizing the model parameters using the Adam optimizer and minimizing the categorical cross-entropy loss.

- **Model Saving and Deployment**: After training, the model is saved as an HDF5 file (`mnist_model.h5`) and can be downloaded for future use or deployment in applications requiring digit recognition.

## Handwritten Digits Recognition API

### Features

- **Rate Limiting**: The API limits requests to 50 per hour and 200 per day to manage server load and prevent abuse.
  
### Usage

- Ideal for integrating digit recognition capabilities into web applications or services.
- Provides a RESTful endpoint (`/predict`) to receive images of handwritten digits in JSON format and returns predictions.

### Benefits

- Demonstrates practical implementation of CNNs for real-world applications.
- Enables seamless integration with frontend applications for real-time digit recognition.

## Usage

- This project is ideal for understanding the basics of image classification using deep learning, specifically for recognizing handwritten digits.
- It serves as a starting point for experimenting with different CNN architectures, hyperparameters, and optimization techniques to improve model performance.

## Benefits

- Provides a hands-on example of using TensorFlow and Keras in a Jupyter notebook environment (Google Colab).
- Demonstrates best practices in data preprocessing, model building, training, evaluation, and model deployment.

## License

This project is licensed under the MIT License. You are free to modify and distribute the code under the terms of this license.

## Author

- Kebalepile Mothshonana
- Contact: kmotshoana@gmail.com
