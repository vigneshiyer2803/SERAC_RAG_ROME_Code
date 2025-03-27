import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)  # Load the GPT-2 language model with a head for causal language modeling
tokenizer = GPT2Tokenizer.from_pretrained(model_name)  # Load the tokenizer for GPT-2 to convert text to tokens

# Function to perform a rank-one model edit (ROME)
def rome_edit(model, tokenizer, target_word, new_embedding):
    """
    Perform a rank-one model edit (ROME) by updating the embedding of a target word in the model.
    Args:
    - model: The pre-trained GPT-2 model.
    - tokenizer: The tokenizer used for converting text to tokens.
    - target_word: The word whose embedding we want to modify.
    - new_embedding: A new embedding vector that will replace the current embedding of the target word.
    """
    # Get the token ID for the target word
    target_token_id = tokenizer.encode(target_word, add_special_tokens=False)[0]  # Encoding the target word into token IDs
    
    # Get the model's embedding layer (the layer that transforms tokens into embeddings)
    embeddings = model.get_input_embeddings()  # This retrieves the input embeddings of the model
    
    # Perform the rank-one edit: update the embedding of the target word
    with torch.no_grad():  # Disable gradient calculation since we are not training
        embeddings.weight[target_token_id] = new_embedding  # Replace the embedding of the target word with the new one

# Function to generate text from the model
def generate_text(model, tokenizer, input_text):
    """
    Generate text using the GPT-2 model given an input text.
    Args:
    - model: The GPT-2 model.
    - tokenizer: The tokenizer for converting text into tokens.
    - input_text: The input text prompt.
    """
    input_ids = tokenizer.encode(input_text, return_tensors="pt")  # Convert input text into token IDs for the model
    
    # Generate text with adjusted parameters
    with torch.no_grad():  # Disable gradients for inference
        output = model.generate(
            input_ids,
            max_length=50,  # Generate text with a maximum length of 50 tokens
            num_return_sequences=1,  # Only return one generated sequence
            no_repeat_ngram_size=2,  # Avoid repeating n-grams of size 2 or greater
            do_sample=True,  # Enable sampling to generate more diverse outputs
            top_k=50,  # Use top-k sampling (limit to the top 50 tokens for each step)
            top_p=0.95,  # Use nucleus sampling (top-p sampling with a cumulative probability of 0.95)
            temperature=0.7  # Control the randomness of the generation (lower value = more deterministic)
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)  # Decode the output token IDs into readable text

# Example usage
if __name__ == "__main__":
    target_word = "apple"  # The word to edit in the model
    input_text = f"I like to eat {target_word}"  # Input text where the target word is used

    # Generate text before editing
    print("**Before Editing:**")
    before_edit_output = generate_text(model, tokenizer, input_text)  # Generate text before applying the edit
    print(before_edit_output)

    # Define a new embedding for the target word
    new_embedding = torch.randn(model.config.hidden_size)  # Create a random new embedding (of the same size as the original)

    # Perform a ROME edit
    rome_edit(model, tokenizer, target_word, new_embedding)  # Apply the rank-one edit to modify the embedding of the target word

    # Generate text after editing
    print("\n**After Editing:**")
    after_edit_output = generate_text(model, tokenizer, input_text)  #
