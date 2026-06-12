# Note: Requires: pip install transformers
from transformers import AutoTokenizer

def demonstrate_tokenization():
    print("="*60)
    print("🧠 DEEP LEARNING: HOW AI READS TEXT")
    print("="*60)
    
    # Load a standard BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # A tricky sentence with punctuation and a made-up word
    text = "Whoa! HuggingFace is suuuper incredibly awesome, isn't it?"
    
    print(f"\n1. ORIGINAL HUMAN TEXT:\n{text}")
    print("-" * 60)
    
    # Let's see the sub-word tokens
    tokens = tokenizer.tokenize(text)
    print(f"\n2. SUB-WORD TOKENS (Notice the ## symbols for split words):\n{tokens}")
    print("-" * 60)
    
    # Let's see what actually goes into the neural network
    model_inputs = tokenizer(text)
    
    print("\n3. NEURAL NETWORK INPUTS:")
    print(f"Input IDs (The vocabulary numbers): \n{model_inputs['input_ids']}")
    
    # Explain special tokens
    print("\nDid you notice extra numbers at the start and end?")
    print(f"Token {model_inputs['input_ids'][0]} = [CLS] (Classification Start) -> Tells model a new sequence is beginning.")
    print(f"Token {model_inputs['input_ids'][-1]} = [SEP] (Sentence End) -> Tells model the sequence has finished.")
    print("-" * 60)
    
    print(f"\nAttention Mask:")
    print("1 means 'real word' (pay attention), 0 means 'padding' (ignore).")
    print(f"{model_inputs['attention_mask']}")
    print("="*60)

if __name__ == "__main__":
    demonstrate_tokenization()