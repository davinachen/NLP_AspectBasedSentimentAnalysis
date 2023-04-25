# NLP_SentimentAnalysis

This project contains an implementation of a RoBERTa-based sentiment analysis classifier using PyTorch and the Hugging Face transformers library. The classifier is trained on a dataset of customer reviews and their corresponding sentiment labels, with aim of predicting the sentiment of new reviews.
The dataset used in this project is a collection of customer reviews in the restaurant domain, originally obtained from SemEval-2014 Task 4. The dataset consists of 1879 reviews, each labeled with one of three sentiment categories: positive, negative, or neutral.

1. Preprocessing
The preprocessing was limited to one function that allowed us to encode each review polarity through preprocess(df).

2. Tokenizing
We tokenized our inputs (sentences and aspect terms) as well as our target (review polarity) through the class Dataset(Dataset). We tokenized all those elements through the pretrained RoBERTa tokenizer. We end up with token tensors of inputs, target as well as attention masks.

3. Classifier Model
The model we chose was based on the model RobertaForMaskedLM.from_pretrained ('roberta-base')  to learn embeddings. Then, we added 9 extra layers (ReLu, Dropouts and Linear) to obtain our predicted polarities layers within our class RobertaClassifier(nn.Module). The final 5 layers are used for fine-tuning on downstream tasks. These layers capture both the syntactic and semantic information of the input text and are capable of extracting context-specific features that are useful for downstream classification tasks.

4. Training
We took care of all the training with our class Classifier. This class calls on the previously defined classes through its train function. It also defines the forward and backward passes. The model is trained using the AdamW optimizer with a learning rate of 2e-5, a cross-entropy loss function, and a batch size of 12. The training process runs for 5 epochs, and the best model is saved based on validation set performance. The data is split into 80% training and 20% validation.

5. Results
The final model achieves a mean accuracy of 85.74, indicating good generalization to new data. 
Dev accs: [85.37, 84.84, 86.17, 87.23, 85.11]
Mean Dev Acc.: 85.74 (0.87)
Exec time: 1245.56 s. ( 249 per run )

6. Usage
To use the trained classifier to predict the sentiment of a new review, simply instantiate a Classifier object and call its predict method with the review text as input. The predict method returns a string representing the predicted sentiment category ("positive", "negative", or "neutral"). The classifier can also be retrained on new data by calling its train method with the paths to the new training and validation data files.