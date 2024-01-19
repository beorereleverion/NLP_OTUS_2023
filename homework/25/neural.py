# я перед этим гонял ноутбук, но у меня то памяти не хватало, то еще чего. 
# максимальную длину я вычислил в нем и многий другой выхлоп тоже оттуда. 
# пришлось перетащить скрипт обучения на машину где проблемы с интернетом. из-за этого ноутбук был переработан в скрипт

import datetime
import json
import numpy as np
import os
import pandas as pd
import random
import time
import torch

from torch.utils.data import DataLoader, TensorDataset,RandomSampler,SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

model_name="/home/vlamykin/25/ruBert-large/snapshots/efdc76b4678bc5c9a51642a4a5364371a89cea96/"
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,                       
    output_attentions=False,
    output_hidden_states=False,
)

dataset_train_fn = 'train.jsonl'
dataset_validation_fn = "val.jsonl"
dataset_test_fn = 'test.jsonl'
train_df = pd.read_json(dataset_train_fn,lines=True)
validation_df = pd.read_json(dataset_validation_fn,lines=True)
test_df = pd.read_json(dataset_test_fn,lines=True)
train_df.set_index('idx', inplace=True)
validation_df.set_index('idx', inplace=True)
test_df.set_index('idx', inplace=True)

def encode(sentences):
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                        sent,
                        add_special_tokens = True,
                        max_length = 168,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                )                
        input_ids.append(encoded_dict['input_ids'])    
        attention_masks.append(encoded_dict['attention_mask'])
    return input_ids, attention_masks

train_sentences=train_df["sentence1"]+". " + train_df["sentence2"]+". " +train_df["word"]
train_sentences=train_sentences.values
validation_sentences=validation_df["sentence1"]+". " + validation_df["sentence2"]+". " +validation_df["word"]
validation_sentences=validation_sentences.values
test_sentences=test_df["sentence1"]+". " + test_df["sentence2"]+". " +test_df["word"]
test_sentences=test_sentences.values

train_labels=train_df.label.values
train_labels=[1 if x else 0 for x in train_labels]
validation_labels=validation_df.label.values
validation_labels=[1 if x else 0 for x in validation_labels]

print('Токенизируем фразы')
train_input_ids, train_attention_masks = encode(train_sentences)
validation_input_ids, validation_attention_masks = encode(validation_sentences)
test_input_ids, test_attention_masks = encode(test_sentences)
print('Токенизация завершена')

train_input_ids = torch.cat(train_input_ids, dim=0)
train_attention_masks = torch.cat(train_attention_masks, dim=0)
train_labels = torch.tensor(train_labels)
validation_input_ids = torch.cat(validation_input_ids, dim=0)
validation_attention_masks = torch.cat(validation_attention_masks, dim=0)
validation_labels = torch.tensor(validation_labels)
test_input_ids = torch.cat(test_input_ids, dim=0)
test_attention_masks = torch.cat(test_attention_masks, dim=0)

train_dataset=TensorDataset(train_input_ids, train_attention_masks, train_labels)
validation_dataset=TensorDataset(validation_input_ids, validation_attention_masks, validation_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_masks)

batch_size = 32

train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size
        )

validation_dataloader = DataLoader(
            validation_dataset,
            sampler = SequentialSampler(validation_dataset),
            batch_size = batch_size
        )

test_dataloader = DataLoader(
            test_dataset,
            sampler = SequentialSampler(test_dataset),
            shuffle=False
        )

optimizer = torch.optim.AdamW(model.parameters(),
                  lr=2e-5,
                  eps=1e-8
                )

epochs = 4
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

total_t0 = time.time()

output_dir = './model_save'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def model_save(epoch_i):
    print(f'saving model to {output_dir}/model_epoch_{epoch_i}')
    model.save_pretrained(f'{output_dir}/model_epoch_{epoch_i}')
    print(f'saving tokemizer to {output_dir}/tokemizer_epoch_{epoch_i}')
    tokenizer.save_pretrained(f'{output_dir}/tokenizer_epoch_{epoch_i}')

for epoch_i in range(0, epochs):



    print("")
    print('======== Эпоха {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Обучение...')

    t0 = time.time()

    total_train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        if step % 50 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Батч {:>5,}  из  {:>5,}.    Затраченное время: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]
        model.zero_grad()
        res = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)
        loss = res['loss']
        logits = res['logits']
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    training_time = format_time(time.time() - t0)

    print("")
    print("  Средняя обучающая потеря: {0:.2f}".format(avg_train_loss))
    print("  Эпоха обучения заняла: {:}".format(training_time))

    print("")
    print(" Запуск валидации...")

    t0 = time.time()

    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]

        with torch.no_grad():
            res = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
        loss = res['loss']
        logits = res['logits']

        total_eval_loss += loss.item()

        logits = logits.detach().numpy()
        label_ids = b_labels.numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Точность: {0:.2f}".format(avg_val_accuracy))
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    validation_time = format_time(time.time() - t0)

    print("  Потери валидации: {0:.2f}".format(avg_val_loss))
    print("  Валидация заняла: {:}".format(validation_time))
    model_save(epoch_i)

print("")
print(" Обучение завершено!")

print("     Всего обучение заняло {:} (ч:м:с)".format(format_time(time.time()-total_t0)))


print('Прогноз меток для {:,} тестовых предложений...'.format(len(test_input_ids)))

model.eval()

predictions = []

for idx, batch in enumerate(test_dataloader):
    b_input_ids, b_input_mask = batch

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask)

    logits = outputs[0].cpu().numpy()
    
    predictions.extend([{"idx": int(idx), "label": logits[i].tolist()} for i in range(len(logits))])
print(' ГОТОВО.')

output_file = 'RUSSE_large_bert.jsonl'

print(f'Запись предсказанных значений в файл {output_file}')
with open(output_file, 'w') as json_file:
    for prediction in predictions:
        label=np.argmax(prediction["label"])
        if label==1:
            label="true"
        else:
            label="false"
        prediction["label"]=label
        json_file.write(json.dumps(prediction) + '\n')
print(' ГОТОВО.')