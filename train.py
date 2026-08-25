import os
import sys
import argparse
import urllib.request
from pathlib import Path

# Add the current directory to python path to ensure aplx_llm can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from aplx_llm import (
    ModelConfig,
    AplexLLM,
    Trainer,
    TrainingConfig,
    BPETokenizer,
    TextDataset,
    TextGenerator,
    GenerationConfig,
    set_seed
)

def download_data(url: str, filepath: Path) -> str:
    """Download a text dataset, or return a large synthetic corpus if offline."""
    if filepath.exists():
        print(f"  [OK] Dataset already exists at {filepath}")
        return filepath.read_text(encoding="utf-8")
        
    print(f"  Downloading dataset from {url}...")
    try:
        # Standard urllib user-agent to prevent HTTP 403 Forbidden
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')
        filepath.write_text(html, encoding="utf-8")
        print(f"  [OK] Download completed. Saved to {filepath}")
        return html
    except Exception as e:
        print(f"  [WARN] Download failed ({e}). Using built-in synthetic training corpus instead.")
        synthetic_corpus = [
            "The transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data.",
            "It is used primarily in the fields of natural language processing and computer vision.",
            "Unlike recurrent neural networks, transformers do not require that the sequential data be processed in order.",
            "For example, if the input data is a natural language sentence, the transformer does not need to process the beginning of the sentence before the end.",
            "Due to this feature, the transformer allows for much more parallelization than RNNs and therefore reduces training times.",
            "Transformers were introduced in 2017 by a team at Google Brain and are increasingly the model of choice for NLP problems, replacing RNN models.",
            "The training process involves feeding the model text, predicting the next word, and adjusting weights using backpropagation.",
            "Python is an interpreted, high-level, general-purpose programming language. Its design philosophy emphasizes code readability.",
            "PyTorch is an open source machine learning library based on the Torch library, developed by Meta's AI Research lab.",
            "Artificial general intelligence (AGI) is the hypothetical ability of an intelligent agent to understand or learn any intellectual task that a human being can.",
            "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
            "Rotary Position Embeddings (RoPE) encode position information by rotating query and key representations in the complex plane.",
            "Grouped Query Attention (GQA) is an optimization that groups query heads together to share key-value projections, reducing memory usage.",
            "SwiGLU is a feed-forward network activation function that outperforms standard GELU in language modeling tasks."
        ]
        # Multiply the synthetic texts to make a decent size corpus
        large_corpus = "\n\n".join(synthetic_corpus * 50)
        filepath.write_text(large_corpus, encoding="utf-8")
        return large_corpus

def main():
    parser = argparse.ArgumentParser(description="APLX LLM Training Script")
    parser.add_argument("--steps", type=int, default=150, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence training chunk length")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Tokenizer vocabulary size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("  APLX_LLM Training Pipeline")
    print(f"  Device: {device.upper()}")
    print("=" * 70)

    # 1. Prepare Data
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    dataset_path = data_dir / "shakespeare.txt"
    dataset_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    raw_text = download_data(dataset_url, dataset_path)
    
    # Split text into documents/paragraphs for the tokenizer
    # Shakespeare is split by double newlines
    train_texts = [p.strip() for p in raw_text.split("\n\n") if len(p.strip()) > 20]
    
    # 2. Train Tokenizer
    print("\nTraining BPE Tokenizer (on a fast subset)...")
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    # Train BPE on a subset to avoid long training times on CPU
    tokenizer.train(train_texts[:100], verbose=False)
    print(f"  [OK] Tokenizer trained. Vocabulary size: {len(tokenizer.vocab)} tokens")
    
    # Test tokenizer
    sample = "To be, or not to be, that is the question."
    encoded = tokenizer.encode(sample)
    print(f"  Test encoding: '{sample}' -> {encoded[:8]}... ({len(encoded)} tokens)")
    
    # 3. Model Configuration
    # We select a model size optimized for learning on CPU
    config = ModelConfig(
        vocab_size=args.vocab_size,
        dim=256,
        n_layers=6,
        n_heads=8,
        n_kv_heads=4,               # Using GQA (n_kv_heads < n_heads)
        max_seq_len=500_000,         # Supports 500k token context scaling
        training_seq_len=args.seq_len,
        sliding_window_size=args.seq_len,
        use_sliding_window=True,
        rope_theta=500_000.0,
        dropout=0.05,
    )
    
    param_info = config.estimate_parameters()
    print("\nModel Parameter Summary:")
    print(f"  Embedding Params: {param_info['embedding']:,}")
    print(f"  Layer Params:     {param_info['all_layers']:,}")
    print(f"  LM Head Params:   {param_info['lm_head']:,}")
    print(f"  Total Params:     {param_info['total']:,} (~{param_info['total']/1e6:.2f}M)")

    # 4. Instantiate Model
    print("\nInstantiating AplexLLM...")
    model = AplexLLM(config).to(device)
    
    # 5. Create Dataset
    split_idx = int(len(train_texts) * 0.9)
    train_dataset = TextDataset(train_texts[:split_idx], tokenizer, seq_len=args.seq_len)
    eval_dataset = TextDataset(train_texts[split_idx:], tokenizer, seq_len=args.seq_len)
    
    print(f"  [OK] Train Dataset: {len(train_dataset)} samples")
    print(f"  [OK] Eval Dataset:  {len(eval_dataset)} samples")

    # 6. Configure Training Loop
    train_config = TrainingConfig(
        learning_rate=args.lr,
        total_steps=args.steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        warmup_steps=min(20, int(args.steps * 0.1)),
        log_interval=10,
        eval_interval=min(50, args.steps),
        save_interval=min(50, args.steps),
        output_dir="./checkpoints",
        use_amp=(device == "cuda"),
    )
    
    trainer = Trainer(
        model=model,
        train_config=train_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    # 7. Start Training
    print("\nStarting training loop...")
    history = trainer.train()
    
    # 8. Generation Test
    print("\n" + "=" * 70)
    print("  Post-Training Generation Evaluation")
    print("=" * 70)
    
    model.eval()
    generator = TextGenerator(model, tokenizer)
    gen_config = GenerationConfig(
        max_new_tokens=60,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
    )
    
    prompts = [
        "First, you",
        "To be, or not",
        "The transformer"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        try:
            generated = generator.generate_text(prompt, gen_config)
            print(f"Generated: '{generated}'")
        except Exception as e:
            print(f"  Error generating text: {e}")

if __name__ == "__main__":
    main()
