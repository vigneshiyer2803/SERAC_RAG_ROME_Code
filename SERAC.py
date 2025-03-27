import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Pre-trained Language Model (GPT-2 as base model)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
lm_model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize Nearest Neighbors for storing edits (768D vectors for GPT embeddings)
dimension = 768
edit_memory = NearestNeighbors(n_neighbors=1, metric="euclidean")
edit_store = {}  # Stores (index -> corrected output)
embeddings_list = []  # Stores embeddings for Nearest Neighbors

# Function to convert text to embeddings
def get_embedding(text):
    """Generate embedding vector for a given text query."""
    tokens = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = lm_model.transformer(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Function to store an edit example in Nearest Neighbors
def add_edit_record(query, corrected_output):
    """Store corrected output in memory."""
    embedding = get_embedding(query)
    embeddings_list.append(embedding)  # Add embedding to the list
    edit_store[len(embeddings_list) - 1] = corrected_output  # Store correction with index
    edit_memory.fit(np.array(embeddings_list))  # Re-fit Nearest Neighbors with updated embeddings

# Function to retrieve the most relevant edit
def retrieve_edit(query):
    """Retrieve the most relevant correction from memory."""
    embedding = get_embedding(query)
    distances, indices = edit_memory.kneighbors(np.array([embedding]))

    if distances[0][0] < 1.0:  # If similarity score is high (lower distance = closer match)
        return edit_store[indices[0][0]]
    return None

# Function implementing the SERAC decision logic
def serac_response(query):
    """Decide whether to use the original model or an edited response."""
    edit_correction = retrieve_edit(query)  # Search for an edit

    if edit_correction:  # If an edit is found, apply it
        print("\n[Using Edited Knowledge]\n")
        return edit_correction  # Return stored correction

    # Otherwise, use the default model
    print("\n[Using Original Model]\n")
    lm_output = lm_model.generate(tokenizer.encode(query, return_tensors="pt"), max_length=50)
    return tokenizer.decode(lm_output[0], skip_special_tokens=True)

# Test Cases
if __name__ == "__main__":
    # Store a knowledge edit
    add_edit_record("What is the company's refund policy?", "Our company now offers a 60-day refund policy.")

    # Queries
    query_1 = "What is the company's refund policy?"
    query_2 = "What is the capital of France?"

    # Get SERAC Responses
    print("Q:", query_1)
    print("A:", serac_response(query_1))  # Should return the corrected response

    print("\nQ:", query_2)
    print("A:", serac_response(query_2))  # Should return the default model response
    
    """Below is the code for ROME"""
    
    
    
    
    