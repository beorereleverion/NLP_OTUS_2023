import json
import numpy as np
import telebot
from telebot import types
from transformers import BertTokenizer, BertForSequenceClassification

with open('/home/vlamykin/nlp-config.json', 'r') as f:
    json_config = json.load(f)

TOKEN = json_config['tg_bot_token']

bot = telebot.TeleBot(TOKEN, num_threads=4)

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id,"""Приветствую вас! 🤖 Я - ваш надежный помощник в решении задач WiC: The Word-in-Context Dataset. Этот набор данных представляет собой надежный бенчмарк для оценки контекст-чувствительных векторных представлений слов.

Чем могу помочь? Если у вас есть слово, которое вам кажется многозначным, предоставьте его мне вместе с двумя предложениями(в три строки с указанными префиксами, имейте ввиду что предложения должны быть не длиннее 150 символов и будут обрезаться), и я вам скажу, верно ли угадали контекст. Например:

```yaml
Слово: дорожка
Предложение 1: Бурые ковровые дорожки заглушали шаги
Предложение 2: Приятели решили выпить на дорожку в местном баре
```

После вашего запроса я предоставлю вам ответ в виде бинарной классификации (true/false) в зависимости от правильности вашего выбора контекста. Давайте начнем! 🚀""",
                     parse_mode="Markdown")

tokenizer = BertTokenizer.from_pretrained("./model_save/tokenizer_epoch_3", do_lower_case=True)
model = BertForSequenceClassification.from_pretrained(
    "./model_save/model_epoch_3",
    num_labels=2,                       
    output_attentions=False,
    output_hidden_states=False,
)
device="cpu"
model.to(device)

def encode(sentence):  
    encoded_dict = tokenizer.encode_plus(
                    sentence,
                    add_special_tokens = True,
                    max_length = 168,
                    pad_to_max_length = True,
                    return_attention_mask = True,
                    return_tensors = 'pt',
            )                

    return encoded_dict['input_ids'], encoded_dict['attention_mask']

def inference(input,attention_mask):
    b_input_ids, b_input_mask = input.to(device),attention_mask.to(device)
    outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask)
    logit = outputs[0].cpu().detach().numpy()[0]
    result = np.argmax(logit)
    return result==1

def get_sentence(text):
    lines = text.split('\n')
    sentence_1 = ""
    sentence_2 = ""
    word = ""
    for line in lines:
        if line.startswith("Предложение 1: "):
            sentence_1 = line[len("Предложение 1:"):].strip()[:150]
        elif line.startswith("Предложение 2:"):
            sentence_2 = line[len("Предложение 2:"):].strip()[:150]
        elif line.startswith("Слово: "):
            word = line[len("Слово: "):].strip()

    return f"{sentence_1}. {sentence_2}. {word}"


@bot.message_handler()
def log_message(message):
    print(message, end='\n')
    sentence=get_sentence(message.text)
    input,attention_mask=encode(sentence)
    result=inference(input,attention_mask)
    bot.send_message(message.chat.id, f'Я думаю {result}\n---\nДавай еще =)\n---')

bot.infinity_polling()