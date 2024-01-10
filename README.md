# Image-Classification-using-Transfer-Learning

This repository presents the implementation of transfer learning aimed at large image classification, employing a MobileNet model. The objective involves refining the model on a personalized dataset consisting of a minimum of 3 classes, each comprising at least 100 images.

## Dataset
Captured images using a phone/camera, ensuring diversity across classes.
Partitioned the dataset into training, validation, and test sets.

## Data Preprocessing
Established an input pipeline for preprocessing and augmenting the training data.
Incorporated data augmentation techniques to enhance the model's resilience.

## Fine-tuning MobileNet
Deployed the pre-trained MobileNet model from ImageNet.
Fine-tuned the model on the customized dataset, adapting the last layers for the new classification task.

## Result
<img width="320" alt="Screenshot 2024-01-09 at 11 29 03 PM" src="https://github.com/YuktaMuthreja/Image-Classification-using-Transfer-Learning/assets/145282953/90b70157-6f88-4a7c-bc2b-08e150fa6ddd">


# Fine-Tuning DistilBERT for Text Classification

This repository offers an extensive tutorial on the process of fine-tuning the DistilBERT model for text classification utilizing TensorFlow and the Hugging Face Transformers library. Text classification stands as a prevalent natural language processing (NLP) task, and DistilBERT, a condensed iteration of BERT (Bidirectional Encoder Representations from Transformers), provides a streamlined yet robust solution.

## Dataset
This project assumes that you have a prepared dataset stored in the train_texts and test_texts variables. It is essential to ensure that your dataset is appropriately preprocessed and split into training and testing sets before proceeding with the model training.

## Tokenization
DistilBERT tokenization is a crucial step in preparing the data for training. The process is performed using the DistilBertTokenizer from the Hugging Face Transformers library. The tokenizer is initialized with a pre-trained DistilBERT model, specifically 'distilbert-base-uncased'. During tokenization, the training and testing texts are transformed into sequences of tokens, and the sequences are encoded with padding and truncation to align with the model's input requirements.

## Model Initialization
The base DistilBERT model for sequence classification is loaded using TFDistilBertForSequenceClassification from the Hugging Face Transformers library. Following the model initialization, it is compiled with an Adam optimizer and categorical cross-entropy loss, setting the stage for training.

# Custom Model Architecture
To enhance the base DistilBERT model, a custom neural network architecture named CustomDistilBERTModel is implemented. This architecture extends the base model by incorporating an additional dense layer with ReLU activation and an output layer with softmax activation. The extension allows for more flexibility and adaptability to specific text classification tasks.

# Result
The model achieved an accuracy score of 93%.
