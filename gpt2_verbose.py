"""Commented version of GPT-2-pico, with verbose output. (by Daniele)
Search for "print" and "PRINT" to see the changes.
"""
import numpy as np


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)  # normalize x to have mean=0 and var=1 over last axis
    return g * x + b  # scale and offset with gamma/beta params


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b


def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # project up
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

    return x


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projection
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    qkv = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

    # split into heads
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

    # perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]  # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]

    # merge heads
    x = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # multi-head causal self attention
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(
        inputs, # [n_seq]
        wte,    # [n_vocab, n_embd] word token embeddings loaded from the GPT-2 model
        wpe,    # [n_ctx, n_embd] word positional embeddings loaded from the GPT-2 model
        blocks, # list of transformer blocks with parameters for each layer
        ln_f,   # layer norm parameters for the final layer
        n_head, # number of heads in the multi-head attention
        first = False # PRINT the first token's embeddings
        ):
    """Givent the input tokens and the model parameters, compute the logits for the next token.
    [n_seq] -> [n_seq, n_vocab]
    """
    
    # token + positional embeddings
    word_token_embedding = wte[inputs]
    word_positional_embedding = wpe[range(len(inputs))]
    x = word_token_embedding + word_positional_embedding  # [n_seq] -> [n_seq, n_embd]

    if first: # PRINT
        print()
        print(f">> Embedding for the first generation (input prompt): {x.shape[0]} tokens, {x.shape[1]} dimensions")
        np.set_printoptions(threshold=10, edgeitems=3, linewidth=1000) # PRINT arrays like [x1 x2 x3 ... x-3 x-2 x-1]
        print(f"Token Embedding for the first token ({inputs[0]}): {word_token_embedding[0]}")
        print(f"Positional Embedding for the first token (0): {word_positional_embedding[0]}")
        print(f"Sum of the two: {x[0]}")
        print()

    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate, encoder=None):

    for igen in range(n_tokens_to_generate):  # auto-regressive decode loop
        logits = gpt2(inputs, **params, n_head=n_head, first=(igen==0))  # model forward pass
        
        ## PRINT the probabilities for the TOP most likely next tokens, for the first IGEN_PRINT_MAX generations
        TOP = 3
        IGEN_PRINT_MAX = 5
        if igen < IGEN_PRINT_MAX:
            top_indices = np.argsort(logits[-1])[-TOP:][::-1]
            max_logit = np.max(logits[-1])                  # Numberically stable softmax
            exp_logits = np.exp(logits[-1] - max_logit)     # Numberically stable softmax
            probabilities = exp_logits / np.sum(exp_logits) # Numberically stable softmax
            print()
            print(f"Top {TOP} next tokens:")
            for i, idx in enumerate(top_indices):
                encoded_token = encoder.decode([idx])
                print(f"  #{i+1}. {idx} ({encoded_token!r}):\t{logits[-1][idx]:11.5f} -> {probabilities[idx]*100:5.2f}%")
            print()

        next_id = np.argmax(logits[-1])  # greedy sampling, equivalente to Temperature = 0
        inputs.append(int(next_id))  # append prediction to input
        print(f"{(igen+1):04d}. {inputs[:-1]} -> {next_id} ({encoder.decode([next_id])!r})")

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids


def main(
        prompt: str, 
        n_tokens_to_generate: int = 20,
        model_size: str = "124M",
        models_dir: str = "models"
        ):
    
    """
    Generate text using GPT-2-pico model.

    Args:
        prompt (str): Input text to generate from.
        n_tokens_to_generate (int): Number of tokens to generate. Since <EOS> is not implemented, the model will simply stop after generating this many tokens. 
        model_size (str): Model size to use. Options are "124M", "355M", "774M", "1558M". See https://openai.com/index/gpt-2-1-5b-release/.
        models_dir (str): Directory where the model files will be downloaded and/or loaded from.
    """
    from utils import load_encoder_hparams_and_params

    print()

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    print(">>>> Loading encoder and model's parameters")
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    print()

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    print(">>>> Encoding (or Tokenization)")
    input_decoded = [ encoder.decode([input_id]) for input_id in input_ids ]
    print(' '.join([f'{tok}:{idt}' for idt, tok in zip(input_ids, input_decoded)]))
    print()

    # make sure we are not surpassing the max sequence length of our model
    print(">>>> Checking Sequence Length")
    print(f"Input Length: {len(input_ids)}")
    print(f"Tokens to Generate: {n_tokens_to_generate}")
    print(f"Max Length: {hparams['n_ctx']}")
    print()
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    print(">>>> Generating Tokens")
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate, encoder=encoder)
    print()

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)

    print(">>>> Final Output")
    print("Initial Prompt:", prompt, "...")
    print("Generated Text: ...", output_text) 
    print("<<<< End of Final Output")

if __name__ == "__main__":

    import fire
    fire.Fire(main)
