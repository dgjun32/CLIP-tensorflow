import tokenizers
import pathlib
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path

if __name__ == 'main':
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files = None, 
                    vocab_size = None,
                    min_frequency = None,
                    special_tokens = [])
                    