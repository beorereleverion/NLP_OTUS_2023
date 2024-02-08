import datetime
import numpy as np
import os 
import pandas as pd
import random
import re
import time
import torch

from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset,RandomSampler,SequentialSampler,random_split
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup


model_name = os.environ.get('MODEL_NAME')
threshold = int(os.environ.get('TRESHOLD'))
lr_value = float(os.environ.get('LR_VALUE'))
eps_value = float(os.environ.get('EPS_VALUE'))
sentence_length = os.environ.get('SENTENCE_LENGTH')
batch_size = os.environ.get('BATCH_SIZE')

folder_name=f'./{model_name}_{threshold}_{batch_size}_{lr_value}_{eps_value}_{sentence_length}'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

print(folder_name)
log_name=f'{folder_name}/log.txt'

def write_log(message):
    tz_Moscow = timezone(timedelta(hours=3))
    current_time = datetime.now(tz_Moscow)
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    log_message = f"[{formatted_time}] {message}"
    
    with open(log_name, "a") as log_file:
        log_file.write(log_message + "\n")


write_log("start reading file")
df=pd.read_json("data.jsonl",lines=True)
labels=df['customFields'].apply(lambda x: x.get('customfield_13901'))
sentence1=df['summary']
sentence2=df['description']
write_log("file has been read")


write_log("start clearing text")
http_reg = re.compile(r'[h]?ttp[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
path_reg = re.compile(r'/\w+(?:/\w+)*')

def clear_text(sentences):
    result = []
    for sentence in sentences:
        soup = BeautifulSoup(sentence)
        text = soup.get_text()
        text = text.replace("\r","")
        text = text.replace("\n","")    
        text = text.replace("\xa0"," ")
        text = text.strip()
        text = http_reg.sub('',text)    
        text = path_reg.sub('',text)    
        result.append(text)
    return pd.Series(result)

sentence1=clear_text(sentence1)
sentence2=clear_text(sentence2)
write_log("text has been cleared")


write_log(f"start converting labels with threshold {threshold}")
unique_values_counts = labels.value_counts()
labels = labels.apply(lambda x: "Неизвестная система" if unique_values_counts[x] <= threshold else x)
write_log(f"fifished converting labels, unique counts {unique_values_counts}")


device="cuda"


write_log(f"load model {model_name}")
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=93,                       
    output_attentions=False,
    output_hidden_states=False,
)
model.to(device)
write_log(f"model {model_name} has been loaded")


write_log(f"start encoding sentences with length {sentence_length} and prepare dataloaders")
sentences=sentence1+[". "]+sentence2
def encode(sentences):
    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,
                            add_special_tokens = True,
                            max_length = 256,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                    )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    return input_ids, attention_masks
input_ids, attention_masks = encode(sentences)
input_ids=torch.cat(input_ids,dim=0)
attention_masks=torch.cat(attention_masks,dim=0)

label_encoder = LabelEncoder()
labels=torch.tensor(label_encoder.fit_transform(labels))
dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

write_log('{:>5,} образцов для обучения'.format(train_size))
write_log('{:>5,} образцов для валидации'.format(val_size))

batch_size = 32
train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size
        )
validation_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset),
            batch_size = batch_size
        )
write_log(f"sentences has been encoded, datalaloaders preparation finished")


write_log(f"---")
write_log(f"---")
write_log(f"---")
write_log(f"start learning")
optimizer = torch.optim.AdamW(model.parameters(),
                  lr=lr_value,
                  eps=eps_value
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
    '''
    Принимает время в секундах и возвращает строку в формате чч:мм:сс.
    '''
    # Округляем до ближайшей секунды.
    elapsed_rounded = int(round(elapsed))

    # Форматируем как чч:мм:сс
    return str(timedelta(seconds=elapsed_rounded))

output_dir = f'{folder_name}/model_save'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def model_save(epoch_i):
    write_log(f'saving model to {output_dir}/epoch_{epoch_i}')
    model.save_pretrained(f'{output_dir}/epoch_{epoch_i}')
    write_log(f'saving tokenizer to {output_dir}/epoch_{epoch_i}')
    tokenizer.save_pretrained(f'{output_dir}/epoch_{epoch_i}')

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
training_stats = []
total_t0 = time.time()
for epoch_i in range(0, epochs):
    write_log("")
    write_log('======== Эпоха {:} / {:} ========'.format(epoch_i + 1, epochs))
    write_log('Обучение...')
    t0 = time.time()
    total_train_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            write_log('  Батч {:>5,}  из  {:>5,}.    Затраченное время: {:}.'.format(step, len(train_dataloader), elapsed))
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
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
    write_log("")
    write_log("  Средняя обучающая потеря: {0:.2f}".format(avg_train_loss))
    write_log("  Эпоха обучения заняла: {:}".format(training_time))
    write_log("")
    write_log("Запуск валидации...")
    t0 = time.time()
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():
            res = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
        loss = res['loss']
        logits = res['logits']
        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
    model_save(epoch_i)
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    write_log("  Точность: {0:.2f}".format(avg_val_accuracy))
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    validation_time = format_time(time.time() - t0)
    write_log("  Потери валидации: {0:.2f}".format(avg_val_loss))
    write_log("  Валидация заняла: {:}".format(validation_time))
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Обучающая потеря': avg_train_loss,
            'Потери на валидации': avg_val_loss,
            'Точность на валидации': avg_val_accuracy,
            'Время обучения': training_time,
            'Время валидации': validation_time
        }
    )
write_log("")
write_log("Обучение завершено!")
write_log("Всего обучение заняло {:} (ч:м:с)".format(format_time(time.time()-total_t0)))