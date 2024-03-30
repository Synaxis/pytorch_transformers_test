# Import necessary libraries
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=128):
        with open(file_path, encoding="utf-8") as f:
            lines = [line.strip() for line in f if len(line) > 0 and not line.isspace()]
        
        print(f"Total lines read: {len(lines)}")

        # Adjust this line to keep shorter sequences
        self.examples = [torch.tensor(tokenizer.encode(line, add_special_tokens=True)[:block_size]) for line in lines]

        print(f"Total examples processed: {len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
configuration = GPT2Config.from_pretrained('distilgpt2', output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained('distilgpt2', config=configuration)

# Load the dataset
dataset = TextDataset(tokenizer, "dataset.txt")
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)  # set to 1 due to slow pc

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
