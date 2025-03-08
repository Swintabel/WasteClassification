# EVALUATING CNN-RNN HYBRID MODEL FOR ACCURATE WASTE OBJECT CLASSIFICATION

## Project Summary
Developed and evaluated a standalone CNN model and a hybrid CNN-RNN model integrating EfficientNetV2B0 with LSTM for waste classification. Leveraged data augmentation and transfer learning on a dataset of 4,752 images across nine categories. Achieved 86% test accuracy and a 96% training accuracy, improving classification for Glass and Metal while addressing challenges in diverse waste categories.

## Key Technologies
**EfficientNetV2B0 (CNN) & LSTM (RNN):** Hybrid model architecture.

**NumPy & Pandas:** Data manipulation and preprocessing.

**OpenCV:** Image processing and manipulation.

**Matplotlib:** Data and model performance visualization.

**Augment:** Custom image augmentation.

**Keras:** Deep learning model building and training.

**Scikit-learn:** Model evaluation (accuracy, confusion matrix).

**Callbacks (EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint):** Training optimization and monitoring.

**L2 Regularization:** Prevents overfitting.

**Pickle:** Model saving and loading.

**os/sys:** File handling and system operations.



## Methodologies

### Step 1: Dataset Collection and Preparation
Involves reading image file paths and labels from a directory, creating a dataframe with labels and paths.  
**Key code:** `os.walk(rootdir)`

### Step 2: Data Preprocessing
Splitting data into training, validation, and test sets with stratification for balanced class distribution.  
**Key code:** `train_test_split(data, test_size=0.3, stratify=data['Label'], random_state=42)`

### Step 3: Data Augmentation
Used `tf.keras.preprocessing.image.ImageDataGenerator` to create image generators for training, validation, and testing. Augmentation was applied only to the training data to enhance model generalization and prevent overfitting.

### Step 4: Model Architecture Design
Built two models using `tf.keras.applications.EfficientNetV2B0` as the base for feature extraction. The first model utilized EfficientNetV2B0 with dense layers for classification. The second model added LSTM layers on top of the pretrained base for temporal processing, reshaping the output to introduce a time dimension. Both models were compiled with the Adam optimizer and categorical crossentropy loss. Callbacks like TensorBoard, ModelCheckpoint, and ReduceLROnPlateau were used for logging, saving the best model, and adjusting the learning rate during training.  
**Key code:** `tf.keras.applications.EfficientNetV2B0`, `Reshape`, `LSTM`, `ModelCheckpoint`, `ReduceLROnPlateau`.

### Step 5: Model Training and Hyperparameter Tuning
In this step, both models were trained using the `fit()` method, with training and validation data provided through `train_images` and `val_images`. Key hyperparameters such as the number of epochs (set to 20) and batch size (32) were specified. The models were trained with callbacks including TensorBoard for logging, ModelCheckpoint to save the best-performing model, and ReduceLROnPlateau to adjust the learning rate when validation accuracy plateaus. Additionally, the architecture of the models, including the number of dense and LSTM layers, was manually tuned through observation of training performance to optimize accuracy.  
**Key code:** `model.fit()`, `model_l.fit()`, `TensorBoard`, `ModelCheckpoint`, `ReduceLROnPlateau`.

### Step 8: Model Evaluation and Prediction
In this step, the trained model is loaded from the saved file (e.g., "cnnModel.keras") using `load_model()`. The model's performance is evaluated on the validation set (`val_images`) using `evaluate()`, and the loss and accuracy are printed. Predictions on the test set (`test_images`) are made using the `predict()` method, and the predicted labels are mapped back to their original class names. The first 5 predictions are displayed for verification. A classification report is generated using `classification_report()` and a confusion matrix is plotted to visualize the performance across classes. The accuracy of the model on the test set is calculated and printed.  
**Key code:** `load_model()`, `evaluate()`, `predict()`, `classification_report()`, `confusion_matrix()`, `accuracy_score()`.

### Step 10: Model Deployment and Testing
In this step, the trained model is deployed using Streamlit for real-time predictions. The model is loaded from the saved file using `pickle.load()`. The Streamlit app allows users to upload images, preprocess them, and display the prediction results. The `st.write()` function is used to show the predicted class label in the app, enabling users to interact with the model seamlessly.

## Key Results

Training Results
- The Hybrid CNN-LSTM model slightly outperformed the standalone CNN model with higher training and validation accuracies.
- The training time for the hybrid model was notably longer than that of the standalone CNN model, primarily due to the added complexity introduced by the LSTM layers.
  
TEST RESULTS.
-	Both Models Achieved an over all Test Accuracy score of 0.86.
-	Overall, both models perform very well in all categories with F1-scores above 0.8 except in the category of Miscellaneous Trash where both models struggle due to the diverse nature of the category.
-	Both Models achieve equal F1-scores in the remaining categories. Food Organics, Paper, Vegetation
-	The Hybrid model slightly outperformed the standalone CNN model with higher F1-score in the Metal(0.87 vs 0.84), Miscellaneous Trash(0.70 vs 0.68) and Textile(0.94 vs 0.91) Categories. 
-	However, The standalone CNN model also slightly outperformed The Hybrid model with higher F1-score in the Cardboard(0.89 vs 0.87), Glass (0.87 vs 0.81), and Plastic(0.85 vs 0.83) categories.



