# PicoGPT
Accompanying blog post: [GPT in 60 Lines of Numpy](https://jaykmody.com/blog/gpt-from-scratch/)

---

You've seen [openai/gpt-2](https://github.com/openai/gpt-2).

You've seen [karpathy/minGPT](https://github.com/karpathy/mingpt).

You've even seen [karpathy/nanoGPT](https://github.com/karpathy/nanogpt)!

But have you seen [picoGPT](https://github.com/jaymody/picoGPT)??!?

`picoGPT` is an unnecessarily tiny and minimal implementation of [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) in plain [NumPy](https://numpy.org). The entire forward pass code is [40 lines of code](https://github.com/jaymody/picoGPT/blob/main/gpt2_pico.py#L3-L41).

picoGPT features:
* Fast? ❌ Nah, picoGPT is megaSLOW 🐌
* Training code? ❌ Error, 4️⃣0️⃣4️⃣ not found
* Batch inference? ❌ picoGPT is civilized, single file line, one at a time only
* top-p sampling? ❌ top-k? ❌ temperature? ❌ categorical sampling?! ❌ greedy? ✅
* Readable? `gpt2.py` ✅ `gpt2_pico.py` ❌
* Smol??? ✅✅✅✅✅✅ YESS!!! TEENIE TINY in fact 🤏

A quick breakdown of each of the files:

* `encoder.py` contains the code for OpenAI's BPE Tokenizer, taken straight from their [gpt-2 repo](https://github.com/openai/gpt-2/blob/master/src/encoder.py).
* `utils.py` contains the code to download and load the GPT-2 model weights, tokenizer, and hyper-parameters.
* `gpt2.py` contains the actual GPT model and generation code which we can run as a python script.
* `gpt2_pico.py` is the same as `gpt2.py`, but in even fewer lines of code. Why? Because why not 😎👍.

#### Dependencies
```bash
pip install -r requirements.txt
```
Tested on `Python 3.9.10`.

#### Usage
```bash
python gpt2.py "Alan Turing theorized that computers would one day become"
```

Which generates

```
 the most powerful machines on the planet.

The computer is a machine that can perform complex calculations, and it can perform these calculations in a way that is very similar to the human brain.
```

You can also control the number of tokens to generate, the model size (one of `["124M", "355M", "774M", "1558M"]`), and the directory to save the models:

```bash
python gpt2.py \
    "Alan Turing theorized that computers would one day become" \
    --n_tokens_to_generate 40 \
    --model_size "124M" \
    --models_dir "models"
```

## Daniele's Fork

My version adds extreme verbosity to the code, to make it easier to understand its numerical operations.

```bash
python gpt2_verbose.py "Alan Turing theorized that computers would one day become"
```

Search for `print` (case insensitive) in the code to find all the added verbosity.

### Notes on the "smartness" of the model

By default the model uses the 124M parameters [GPT-2 model](https://github.com/openai/gpt-2),
with an output length of 20 tokens.
PicoGPT is designed to get the most probable token at each step (equivalent to temperature=0),
therefore these examples are reproducible.

Some examples (with the default settings):

```text
Input: Where is the Colosseum? 
Output: \n The Colosseum is a small, rectangular building in the center of the city.

Input: The Colosseum is in
Output: the middle of a great deal of activity, and the city is a great place to live. The

Input: The Colosseum is in the city of 
    Top 3 next tokens:
        #1. 350 (' P'):         -90.20802 ->  1.87%
        #2. 309 (' T'):         -90.25656 ->  1.78%
        #3. 327 (' C'):         -90.38332 ->  1.57%
    Top 3 next tokens:
        #1. 10961 ('arma'):     -92.18983 -> 11.35%
        #2. 9160 ('isa'):       -92.36401 ->  9.53%
        #3. 1142 ('ern'):       -93.27112 ->  3.85%    
Output: Parma, Italy, where it is located. \n The Colosseum is a small

Input: The famous Colosseum is in the city of
    Top 3 next tokens:
        #1. 10598 (' Rome'):    -90.74231 ->  2.40%
        #2. 350 (' P'):         -90.75540 ->  2.37%
        #3. 309 (' T'):         -90.90068 ->  2.05%
Output: Rome, where the city is known as the "Colosseum of the Dead." The Col

Input: The famous Colosseum is in the Italian city of
    Top 3 next tokens:
        #1. 5215 (' Gen'):      -84.74657 ->  5.91%
        #2. 21574 (' Milan'):   -85.02224 ->  4.49%
        #3. 10598 (' Rome'):    -85.15120 ->  3.94%
Output: Genoa, where it is said to have been built by the Romans.


Input: In which Italian city is the famous Colosseum located?
Output: \n The Colosseum is a city of the famous Colosseum, which is
```
