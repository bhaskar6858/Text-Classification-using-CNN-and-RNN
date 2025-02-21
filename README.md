# Text-Classification-using-CNN-and-RNN
This project focuses on classifying news articles into four categories using deep learning techniques. It explores two powerful architectures:  Convolutional Neural Networks (CNN) for capturing local patterns in text. A hybrid CNN + Bidirectional LSTM (RNN) model for understanding both local and sequential dependencies in news articles.


We use the AG News Subset dataset and compare the performance of two powerful models:

1)  Convolutional Neural Networks (CNN) for extracting local patterns.
2)  A hybrid CNN + Bidirectional LSTM (RNN) model for understanding both local and sequential dependencies in text.

Flow :

◦ Used TensorFlow and TensorFlow Datasets (TFDS) for seamless data handling.
◦ TextVectorization for efficient preprocessing and tokenization.
◦ Implementation of both CNN and CNN + RNN (Bidirectional LSTM) architectures.
◦ Visualized training metrics (accuracy and loss) across epochs.
◦ Real-time predictions on new, unseen news samples.
◦ Handled of text sequences with masking and dropout for regularization.


The dataset is automatically loaded via TensorFlow Datasets and contains thousands of news articles categorized into four classes:

◦ World
◦ Sports
◦ Business
◦ Sci/Tech

Model Architectures: 

1) CNN Model
◦ Text Vectorization
◦ Embedding Layer
◦ 2 × Conv1D + MaxPooling1D layers
◦ Dense Layer with Dropout
◦ Output Layer with Softmax Activation

2) CNN + Bidirectional LSTM Model
◦ Text Vectorization
◦ Embedding Layer
◦ Conv1D + MaxPooling1D layers
◦ Bidirectional LSTM for sequence understanding
◦ Dense Layer with Dropout
◦ Output Layer with Softmax Activation


Model Evaluation:
The models are evaluated based on:
◦ Accuracy
◦ Loss
◦ Generalization Performance on validation sets

The CNN + Bidirectional LSTM model demonstrated superior performance, accurately classifying articles across all four categories.

The model's performance (training and validation accuracy/loss) is visualized using Matplotlib for better analysis.

Sample Prediction: 
"Tesla unveils humanoid robot during AI Day event."	           Sci/Tech 
"Virat Kohli scores a century at Lords Stadium."	             Sports 
"NVIDIA acquires ARM in a billion-dollar deal."	               Business 
"World leaders meet for climate change summit."	               World 

Performance:
◦ Achieved high validation accuracy with the hybrid CNN + RNN model.
◦ Descent performance across diverse, real-world news samples.
◦ Efficient training with minimal overfitting, thanks to masking and dropout techniques.

Future Improvements: 
◦ Implement attention mechanisms for enhanced contextual understanding.
◦ Experiment with transformer-based architectures like BERT.
◦ Integrate more diverse datasets for multilingual text classification.
◦ Add hyperparameter tuning for further optimization.

Acknowledgments:

◦ TensorFlow and Keras documentation
◦ AG News Subset dataset from TensorFlow Datasets
◦ Inspiration from deep learning techniques in NLP
