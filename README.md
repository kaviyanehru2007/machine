 This project implements a Support Vector Machine (SVM) model to classify images of cats and dogs using the Kaggle Dogs vs Cats dataset.
The goal is to understand image preprocessing, feature extraction, and machine learning classification using SVM.
 Dataset
Source: Kaggle
Link: https://www.kaggle.com/c/dogs-vs-cats/data
Classes:
Cat 
Dog 
After extracting the dataset, the directory structure should look like:
Copy code

train/
├── cat.0.jpg
├── cat.1.jpg
├── dog.0.jpg
├── dog.1.jpg
└── ...
 Technologies Used
Python
OpenCV
NumPy
Scikit-learn
Matplotlib
tqdm
 Algorithm Used
Support Vector Machine (SVM)
Kernel: Linear
Used for binary classification
Works on flattened image feature vectors
Workflow
Load images from dataset
Resize images to 64×64 pixels
Convert images to feature vectors (flattening)
Normalize pixel values
Split data into training and testing sets
Train SVM classifier
Evaluate model performance
 Installation
Install the required libraries using:
Copy code
Bash
pip install numpy opencv-python scikit-learn matplotlib tqdm
 How to Run
Download the dataset from Kaggle
Extract the train folder
Place the Python file in the same directory
Run the script:
Copy code
Bash
python svm_cats_dogs.py
 Future Improvements
Use HOG (Histogram of Oriented Gradients) features
Apply PCA for dimensionality reduction
Replace SVM with CNN for higher accuracy
 Conclusion
This project demonstrates how traditional machine learning algorithms like SVM can be applied to image classification problems through proper preprocessing and feature extraction.
