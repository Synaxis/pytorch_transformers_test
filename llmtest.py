import json
import torch
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader

# Define the dataset class
class TextDataset(Dataset):
    def __init__(self, tokenizer, conversations, block_size=128):
        # Tokenize the conversations
        self.examples = []
        for conversation in conversations:
            # Flatten the conversation pairs and tokenize
            flattened_conversation = [line for pair in conversation for line in pair]
            tokens = tokenizer.encode(' '.join(flattened_conversation), truncation=True, max_length=block_size)
            self.examples.append(torch.tensor(tokens))
        print(f"Total examples processed: {len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

# Function to load and preprocess the data
def load_and_preprocess_data(tokenizer, file_path):
    conversations = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            conversations.append(data['conversations'])
    return conversations

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

# Load and preprocess the data
conversations = load_and_preprocess_data(tokenizer, 'train.jsonl')

# Create the dataset
dataset = TextDataset(tokenizer, conversations)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load the model configuration and initialize the model
configuration = GPT2Config.from_pretrained('distilgpt2', output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained('distilgpt2', config=configuration)

# Training settings
device = torch.device("cpu")
model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    for batch in data_loader:
        inputs = batch.to(device)
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Save the trained model
model.save_pretrained('./my_model_directory/')
tokenizer.save_pretrained('./my_model_directory/')
