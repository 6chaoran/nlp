# Build A Simple Machine Translator (part-1)
*encoder decoder framework with lstm*

# Introduction

[seq2seq model](https://google.github.io/seq2seq/) is a general purpose sequence learning and generation model.
It uses encoder decoder arthitecture, which is widely wised in different tasks in NLP, such as Machines Translation, Question Answering, Image Captioning.

![](https://raw.githubusercontent.com/6chaoran/nlp/master/nmt/image/encoder-decoder-architecture.png)

The model consists of two major components:


* __Encoder__: a RNN network, used understand the input sequence and learning the pattern. 

* __Decoder__: a RNN netowrk, used to generate the sequence based on learned pattern from encoder.

major steps to train a seq2seq model:

1. __Word/Sentence representation__: this includes tokenize the input and output sentences, matrix representation of sentences, such as TF-IDF, bag-of-words.
2. __Word Embedding___: lower dimensional representation of words. With a sizable corpus, embedding layers are highly recommended.
3. __Feed Encoder__: input source tokens/embedded array into encoder RNN (I used LSTM in this post) and learn the hidden states
4. __Connect Encoder & Decoder___: pass the hidden states to decoder RNN as the initial states
5. __Decoder Teacher Forcing___: input the sentence to be translated to decoder RNN, and target is the sentences which is one word right-shifted. In the structure, the objective of each word in the decoder sentence is to predict the next word, with the condition of encoded sentence and prior decoded words.  This kind of network training is called **teacher forcing**.

However, we can't directly use the model for predicting, because we won't know the decoded sentences when we use the model to translate. Therefore, we need another inference model to performance translation (sequence generation).

major steps to infer a seq2seq model:

1. __Encoding__: feed the processed source sentences into encoder to generate the hidden states
2. __Deocoding__: the initial token to start is '<s>', with the hidden states pass from encoder, we can predict the next token.
3. __Token Search__: 
    + for each token prediction, we can choose the token with the most probability, this is called greedy search. We just get the best at current moment. 
    + alternatively, if we keep the n best candidate tokens, and search for a wider options, this is called beam search, n is the beam size.
    + the stop criteria can be the '<e>' token or the length of sentence is reached the maximal.

## Dataset

The data used in this post is from [ManyThings.org](http://www.manythings.org/anki/). It provides toy datasets for many bilingual sentence pairs. I used [english-chinese dataset](http://www.manythings.org/anki/cmn-eng.zip).

## Prepare Data

### clean punucations
* for english, I simply removed `,.!?` and convert to lower case
* for chinese, I only removed `,.!?。，！？\n`

```
# raw data
0   Hi. 嗨。
1   Hi. 你好。
2   Run.    你用跑的。
3   Wait!   等等！
4   Hello!  你好
```

### tokenize
* for english, I just split the sentence by space
* for chinese, I used [jieba](https://github.com/fxsjy/jieba) parser to cut the sentence.

```python
def clean_eng(x):
    x = x.lower()
    x = re.sub('[,.!?]','',x)
    return x

def clean_chn(x):
    x = re.sub('[,.!?。，！？\n]','',x)
    # use jieba parser to cut chinese
    x = jieba.cut(x)
    return ' '.join(x)
```
```
# processed data
0   hi  嗨
1   hi  你好
2   run 你 用 跑 的
3   wait    等等
4   hello   你好
```
### sequence reprenstation
I used integer to represent the word in the sentence, so that we can use word embedding easily. Two separate corpus will be kept for source and target sentences. To cater for sentence with different length, we capped the sentence at `maxlen` for long sentence and pad `0` for short sentence.   

I used below code snippet to generate vocabulary size, max_len, and padded sequence for both english and chinese sentences.

```python
def tokenize(texts, maxlen = 20, num_words = 9000):
    """ 
    tokenize array of texts to padded sequence
    Parameters
    ----------
    texts: list
        list of strings
    maxlen: int
        max length of sentence 
    num_words: int
        max vocab size
    Returns
    ----------
    tuple (tokenizer, vocab_size, max_len, padded_seqs)
    """
    tokenizer = Tokenizer(filters='',num_words = num_words, oov_token = '<oov>')
    tokenizer.fit_on_texts(texts)
    vocab_size = len(tokenizer.index_word) + 1
    max_len = max(list(map(lambda i: len(i.split()), texts)))
    max_len =  min(max_len, maxlen)
    vocab_size = min(vocab_size, num_words)

    seqs = tokenizer.texts_to_sequences(texts)
    padded_seqs = pad_sequences(seqs, max_len, padding='post')
    return tokenizer, vocab_size, max_len, padded_seqs
```
The resulting prepared data should look like something below:

```
# sequence representation
0   [928]   [1012]
1   [928]   [527]
2   [293]   [7, 141, 200, 5]
3   [160]   [1671]
4   [1211]  [527]

# padded sequences
0 [ 928    0    0    0    0    0    0    0    0]    [1012    0    0    0    0    0    0    0    0    0    0    0    0    0]
1 [ 928    0    0    0    0    0    0    0    0]    [ 527    0    0    0    0    0    0    0    0    0    0    0    0    0]
2 [ 293    0    0    0    0    0    0    0    0]    [   7  141  200    5    0    0    0    0    0    0    0    0    0    0]
3 [ 160    0    0    0    0    0    0    0    0]    [1671    0    0    0    0    0    0    0    0    0    0    0    0    0]
4 [1211    0    0    0    0    0    0    0    0]    [ 527    0    0    0    0    0    0    0    0    0    0    0    0    0]
```
## Encoder

## Decoder

# Reference

* [Keras Blog](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)