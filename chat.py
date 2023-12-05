import random
import json
import torch
import unicodedata
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from fuzzywuzzy import fuzz

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding='utf8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Bot IFRS"

# Remover acentos e converter texto para minúsculas
def normalize_text(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn').lower()

# Função para encontrar a intenção com melhor correspondência
def find_intent(input_text, intents):
    max_ratio = 0
    matched_intent = None

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            ratio = fuzz.ratio(input_text, pattern)
            if ratio > max_ratio:
                max_ratio = ratio
                matched_intent = intent

    return matched_intent

def get_response(msg):
    # Normalize o texto de entrada
    msg = normalize_text(msg)

    sentence = tokenize(msg)
    print(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.90:
        # Encontre a intenção correspondente usando a correspondência aproximada
        matched_intent = find_intent(msg, intents)

        if matched_intent:
            response = random.choice(matched_intent['responses'])
            if 'link' in matched_intent and matched_intent['link']:
                response += " " + matched_intent['link']
            print(prob.item())
            return response

    return "Desculpe, não entendi sua pergunta ou ela não está contemplada nesta interação. Você pode reformular de outra maneira ou utilizar a barra de pesquisa."

if __name__ == "__main__":
    print("Vamos lá! (digite 'sair' para sair)")
    while True:
        sentence = input("Você: ")
        if sentence == "sair":
            break

        resp = get_response(sentence)
        print(resp)
