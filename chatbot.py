from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Path to your trained model
model_directory = './my_model_directory/'

# Load the trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_directory)
tokenizer = GPT2Tokenizer.from_pretrained(model_directory)

# Ensure the model is in evaluation mode
model.eval()

def generate_text(prompt, max_length=50):
    # Encode the input text
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate a response from the model
    output = model.generate(
        input_ids,
        max_length=max_length + len(input_ids[0]),
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the input part from the generated output
    return generated_text[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):]

# Interactive loop for chatting with the chatbot
print("Chatbot activated. Type 'exit' to quit.")
while True:
    # Get user input
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    # Generate and print the model's response
    response = generate_text(user_input)
    print(f"Chatbot: {response}")
