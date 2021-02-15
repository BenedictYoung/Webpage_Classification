import os
import torch
import numpy as np
import time

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split

from get_data import *
from functions import *

# parameters
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
batch_size = 32
max_length = 256
learning_rate = 1e-5
epochs = 5

# get data
dataset_path = './webkb'
texts, labels = read_data(dataset_path)
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=.1, stratify=labels)
train_texts, valid_texts, train_labels, valid_labels = train_test_split(train_texts, train_labels, test_size=2/9, stratify=train_labels)
print("data loaded")

# load pretrained model
pretrained_path = './pretrained'
tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_path)
model.classifier = torch.nn.Linear(768, 7)
model.num_labels = 7
model.to(device)
print("model loaded")

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

train_dataset = web_dataset(train_encodings, train_labels)
valid_dataset = web_dataset(valid_encodings, valid_labels)
test_dataset = web_dataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
print("data encoded")

optimizer = AdamW(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        if step % 50 == 0:
            prediction = outputs['logits']
            train_accuracy = cal_accuracy(prediction, labels)
            print('epoch: [{}/{}][{}/{}]\t train_loss {:.4f}\t train_acc {:.4f}'.format(epoch, epochs, step, len(train_loader), loss.item(), train_accuracy),
              time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
        right = 0
        total = 0
        total_loss = 0

        for step, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            prediction = outputs['logits']
            right += cal_number(prediction, labels)
            total += labels.shape[0]
            total_loss += loss.item()

        print('test_loss {:.4f}\t test_acc {:.4f}'.format(total_loss / len(test_loader), right / total),
              time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

