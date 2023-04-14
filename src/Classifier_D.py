import numpy as np
import pandas as pd
from typing import List

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer


COLUMN_NAMES = ["sentiment", "aspect_category", "aspect_term", "position", "sentence"]
CATEGORY_NAMES = ['AMBIENCE#GENERAL', 'DRINKS#PRICES', 'DRINKS#QUALITY',
                  'DRINKS#STYLE_OPTIONS', 'FOOD#PRICES', 'FOOD#QUALITY',
                  'FOOD#STYLE_OPTIONS', 'LOCATION#GENERAL', 'RESTAURANT#GENERAL',
                  'RESTAURANT#MISCELLANEOUS', 'RESTAURANT#PRICES', 'SERVICE#GENERAL']
NUM_CATEGORIES = 12
MAX_LENGTH = 60
BATCH_SIZE = 8
LEARNING_RATE = 0.1
EPOCHS = 1
EMBEDDING_SIZE = 768
TRAIN_SAMPLES = 1503
VALID_SAMPLES = 376
PATH = "bert_model.pt"


def generate_target_and_category(df):
    df["target"] = np.nan
    df.loc[df.sentiment == "positive", "target"] = 0
    df.loc[df.sentiment == "negative", "target"] = 1
    df.loc[df.sentiment == "neutral", "target"] = 2
    df_new = pd.get_dummies(df.aspect_category)
    existing_columns = set(df_new.columns)
    for column in CATEGORY_NAMES:
        if column in existing_columns:
            df[column] = df_new[column]
        else:
            df[column] = 0

    df["aspect_category"] = df["aspect_category"].str.lower().str.replace("#", "-")
    df["sentence"] = df["aspect_category"] + "-" + df["aspect_term"] + ": " + df["sentence"]
    return df


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes, device):
        super(SentimentClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(
            "activebus/BERT_Review", output_hidden_states=True)
        self.device = device
        self.elu1 = nn.ELU()
        self.drop1 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(EMBEDDING_SIZE+NUM_CATEGORIES, 300)
        self.elu2 = nn.ELU()
        self.drop2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(300+NUM_CATEGORIES, n_classes)

    def forward(self, input_ids, attention_mask, category_dummies):
        hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
            ).hidden_states

        num_batches = len(hidden_states[0])
        sentence_embedding = torch.zeros(num_batches, EMBEDDING_SIZE)
        sentence_embedding = sentence_embedding.to(self.device)
        for layer in hidden_states[-4:]:
            layer_embedding = torch.mean(layer, dim=1)  # sentence vector of the layer
            sentence_embedding += layer_embedding

        sentence_embedding /= 4  # average sentence vector
        next_input = torch.cat((sentence_embedding, category_dummies), dim=1)
        next_input = self.elu1(next_input)
        next_input = self.drop1(next_input)
        next_input = self.fc1(next_input)
        next_input = self.elu2(next_input)
        next_input = self.drop2(next_input)
        next_input = torch.cat((next_input, category_dummies), dim=1)
        output = self.fc2(next_input)
        return output


class ABSA_Dataset(Dataset):
    def __init__(self, data_frame, tokenizer, max_length):
        self._sentences = data_frame["sentence"]
        self._targets = torch.tensor(data_frame["target"], dtype=torch.long)
        self._category_dummies = data_frame[CATEGORY_NAMES].to_numpy()
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self):
        return len(self._targets)

    def __getitem__(self, item):
        text = self._sentences[item]
        target = self._targets[item]
        category_dummies = self._category_dummies[item, :]
        encoded_text = self._tokenizer.encode_plus(
            text,
            return_tensors='pt',
            max_length=self._max_length,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            )
    
        bert_dict = dict()
        bert_dict["category_dummies"] = category_dummies
        bert_dict["targets"] = target
        bert_dict["input_ids"] = encoded_text["input_ids"][0]
        bert_dict["attention_mask"] = encoded_text["attention_mask"][0]
        return bert_dict

class Classifier:
    """The Classifier"""
    def __init__(self):
        # https://huggingface.co/activebus/BERT_Review
        self.tokenizer = AutoTokenizer.from_pretrained("activebus/BERT_Review")
    
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile.
        """
        model = SentimentClassifier(3, device)
        
        # split columns into 'sentiment'' aspect_category' 'aspect_term' 'position'	'sentence'
        train_file = pd.read_csv(train_filename, delimiter="\t", names=COLUMN_NAMES, header=None)
        valid_file = pd.read_csv(dev_filename, delimiter="\t", names=COLUMN_NAMES, header=None)
        
        # turn 'sentiment' into numerical variable 'target' and turn category into dummy variable
        train_file = generate_target_and_category(train_file)
        valid_file = generate_target_and_category(valid_file)
        
        # tokenize both train & validation set
        train_dataset = ABSA_Dataset(data_frame=train_file, tokenizer=self.tokenizer, max_length=MAX_LENGTH)
        valid_dataset = ABSA_Dataset(data_frame=valid_file, tokenizer=self.tokenizer, max_length=MAX_LENGTH)
        
        # data loading
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=2)
        
        # hyperparameters for the classifier
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss().to(device)
        epochs = EPOCHS

        valid_loss_min = np.Inf
        for epoch in range(1, epochs + 1):
            model.train()
            if epoch == 1:
                for param in model.encoder.parameters():
                    param.requires_grad = False

            train_loss = 0.0
            train_correct_predictions = 0
            for batch_dict in train_dataloader:
                input_ids = batch_dict["input_ids"].to(device)
                attention_mask = batch_dict["attention_mask"].to(device)
                targets = batch_dict["targets"].to(device)
                category_dummies = batch_dict["category_dummies"].to(device)
                optimizer.zero_grad()
                output = model(input_ids, attention_mask, category_dummies)
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
                category_dummies = batch_dict["category_dummies"].to(device)
                output = model(input_ids, attention_mask, category_dummies)
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

            # saving the model if validation loss has decreased
            if valid_loss < valid_loss_min:
                print(f"Validation loss decreased ({valid_loss_min:.6f} --> " \
                      f"{valid_loss:.6f}). Saving model..")
                torch.save(model.state_dict(), PATH)
                print("Model Saved")
                valid_loss_min = valid_loss


    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        test_file = pd.read_csv(data_filename, delimiter="\t", names=COLUMN_NAMES, header=None)
        test_file = generate_target_and_category(test_file)
        test_dataset = ABSA_Dataset(data_frame=test_file, tokenizer=self.tokenizer, max_length=MAX_LENGTH)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=2)
        predictions = []
        predictions_dict = {0: "positive", 1: "negative", 2: "neutral"}

        model = SentimentClassifier(3, device)
        model.load_state_dict(torch.load(PATH))
        model = model.to(device)
        model.eval()
        
        for batch_dict in test_dataloader:
            input_ids = batch_dict["input_ids"].to(device)
            attention_mask = batch_dict["attention_mask"].to(device)
            category_dummies = batch_dict["category_dummies"].to(device)
            output = model(input_ids, attention_mask, category_dummies)
            _, preds = torch.max(output, dim=1)

            for prediction in preds.detach().cpu().numpy():
                predictions.append(predictions_dict[prediction])

        return predictions