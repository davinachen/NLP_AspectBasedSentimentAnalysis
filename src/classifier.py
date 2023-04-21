import numpy as np
import pandas as pd
from typing import List
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer


COLUMN_NAMES = ["sentiment", "aspect_category", "aspect_term", "position", "sentence"]
MAX_LENGTH = 64
BATCH_SIZE = 12
LEARNING_RATE = 2e-5
EPOCHS = 5
EMBEDDING_SIZE = 768
TRAIN_SAMPLES = 1503
VALID_SAMPLES = 376
PATH = "model.pt"


def preprocess(df):
    df["target"] = np.nan
    df.loc[df.sentiment == "positive", "target"] = 0
    df.loc[df.sentiment == "negative", "target"] = 1
    df.loc[df.sentiment == "neutral", "target"] = 2
    df["aspect_category"] = df["aspect_category"].str.lower().str.replace("#", "-")
    # Concatenating ensures that aspect terms and categories are treated as a single unit
    df["sentence"] = df["aspect_category"] + "-" + df["aspect_term"] + ": " + df["sentence"]
    return df


class RobertaClassifier(nn.Module):
    def __init__(self, n_classes, device):
        super(RobertaClassifier, self).__init__()
        self.device = device
        self.configuration = RobertaConfig()
        self.configuration.output_hidden_states = True
        self.encoder = RobertaForMaskedLM.from_pretrained('roberta-base', output_hidden_states=True)
        self.classifier = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(EMBEDDING_SIZE,384),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(384, 192),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(192,n_classes))

    def forward(self, input_ids, attention_mask):
        hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).hidden_states
        sentence_embedding = torch.zeros(len(hidden_states[0]), EMBEDDING_SIZE).to(self.device)
        
        # Avg of last five layers (most informative in BERT) to create sentence vector for classification
        for layer in hidden_states[-5:]:
            layer_embedding = torch.mean(layer, dim=1)
            sentence_embedding += layer_embedding
        sentence_embedding /= 5
        
        output = self.classifier(sentence_embedding)
        return output


class Dataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self._sentences = df["sentence"]
        self._targets = torch.tensor(df["target"], dtype=torch.long)
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self):
        return len(self._targets)

    def __getitem__(self, item):
        target = self._targets[item]
        text = self._sentences[item]
        encoded_text = self._tokenizer.encode_plus(
            text,
            return_tensors='pt',
            max_length=self._max_length,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True)
    
        roberta_dict = dict()
        roberta_dict["targets"] = target
        roberta_dict["input_ids"] = encoded_text["input_ids"][0]
        roberta_dict["attention_mask"] = encoded_text["attention_mask"][0]
        return roberta_dict


class Classifier:
    """The Classifier"""
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile.
        """
        model = RobertaClassifier(3, device)
        
        # split columns into 'sentiment'' aspect_category' 'aspect_term' 'position'	'sentence'
        train_file = pd.read_csv(train_filename, delimiter="\t", names=COLUMN_NAMES, header=None)
        valid_file = pd.read_csv(dev_filename, delimiter="\t", names=COLUMN_NAMES, header=None)
        
        # turn 'sentiment' into numerical variable 'target' and turn category into dummy variable
        train_file = preprocess(train_file)
        valid_file = preprocess(valid_file)
        
        # tokenize both train & validation set
        train_dataset = Dataset(df=train_file, tokenizer=self.tokenizer, max_length=MAX_LENGTH)
        valid_dataset = Dataset(df=valid_file, tokenizer=self.tokenizer, max_length=MAX_LENGTH)
        
        # data loading
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=2)
        valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=2)
        
        # hyperparameters for the classifier
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss().to(device)
        epochs = EPOCHS
        
        valid_accuracy_max = 0.0
        for epoch in range(1, epochs + 1):
            model.train()

            train_loss = 0.0
            train_correct_predictions = 0
            for batch_dict in train_dataloader:
                input_ids = batch_dict["input_ids"].to(device)
                attention_mask = batch_dict["attention_mask"].to(device)
                targets = batch_dict["targets"].to(device)
                optimizer.zero_grad()
                output = model(input_ids, attention_mask)
                _, preds = torch.max(output, dim=1)

                train_correct_predictions += torch.sum(preds == targets)
                loss = criterion(output, targets)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

            model.eval()
            valid_loss = 0.0
            valid_correct_predictions = 0
            for batch_dict in valid_dataloader:
                input_ids = batch_dict["input_ids"].to(device)
                attention_mask = batch_dict["attention_mask"].to(device)
                targets = batch_dict["targets"].to(device)
                output = model(input_ids, attention_mask)
                _, preds = torch.max(output, dim=1)

                valid_correct_predictions += torch.sum(preds == targets)
                loss = criterion(output, targets)
                valid_loss += loss.item()

            # evaluation
            train_loss = train_loss / len(train_dataloader)
            valid_loss = valid_loss / len(valid_dataloader)
            train_accuracy = train_correct_predictions / TRAIN_SAMPLES
            valid_accuracy = valid_correct_predictions / VALID_SAMPLES

            print(f"Epoch: {epoch}. " \
                  f"Training Loss: {train_loss:.6f}.  " \
                  f"Validation_loss: {valid_loss:.6f}. " \
                  f"Train accuracy: {train_accuracy:.2f}. " \
                  f"Valid accuracy: {valid_accuracy:.2f}.")
        
            if valid_accuracy > valid_accuracy_max:
                print(f"Validation accuracy increased ({valid_accuracy_max:.6f} --> " \
                      f"{valid_accuracy:.6f}). Saving model..")
                torch.save(model.state_dict(), PATH)
                print("Model Saved")
                valid_accuracy_max = valid_accuracy


    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        test_file = pd.read_csv(data_filename, delimiter="\t", names=COLUMN_NAMES, header=None)
        test_file = preprocess(test_file)
        test_dataset = Dataset(df=test_file, tokenizer=self.tokenizer, max_length=MAX_LENGTH)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=2)
        predictions = []
        predictions_dict = {0: "positive", 1: "negative", 2: "neutral"}

        model = RobertaClassifier(3, device)
        model.load_state_dict(torch.load(PATH))
        model = model.to(device)
        model.eval()
        
        for batch_dict in test_dataloader:
            input_ids = batch_dict["input_ids"].to(device)
            attention_mask = batch_dict["attention_mask"].to(device)
            output = model(input_ids, attention_mask)
            _, preds = torch.max(output, dim=1)

            for prediction in preds.detach().cpu().numpy():
                predictions.append(predictions_dict[prediction])

        return predictions