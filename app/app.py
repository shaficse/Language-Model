from flask import Flask, render_template, request, jsonify
import torch
from torchtext.data.utils import get_tokenizer
from model import LSTMLanguageModel, generate  # Import model class and generation function
import json

app = Flask(__name__)


# Tokenize the input prompt
tokenizer = get_tokenizer('basic_english')

# Load the vocabulary dictionary from JSON
with open('models/vocab.json', 'r') as f:
    stoi = json.load(f)
    
# Load the configuration
with open('models/config.json', 'r') as f:
    config = json.load(f)

# Extract hyperparameters or configuration from the config dictionary
seq_len = config['seq_len']
batch_size = config['batch_size']
vocab_size = len(stoi)
emb_dim = config['emb_dim']              
hid_dim = config['hid_dim']              
num_layers =config['num_layers']            
dropout_rate = config['dropout_rate']
seed = 0

# Determine if a CUDA (NVIDIA GPU) is available, set the device to 'cuda'. Otherwise, use 'cpu'.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load your model (ensure this is adapted to your specific model)
ModelClass  = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)
ModelClass.load_state_dict(torch.load('models/best-val-lstm_lm.pt',  map_location=device))
ModelClass.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data['prompt']
    max_seq_len = int(data['max_seq_len'])
    temperature = float(data['temperature'])
    
    # Assuming you have a generate function similar to the one you provided
    generated_text = generate(prompt, max_seq_len, temperature, ModelClass, tokenizer, 
                          stoi, device, seed)
    
    # Convert the list of tokens to a string
    generated_text = ' '.join(generated_text)
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)
