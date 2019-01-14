# Build A Simple Machine Translator (part-1)
*encoder decoder framework with lstm*

# Introduction

[seq2seq model](https://google.github.io/seq2seq/) is a general purpose sequence learning and generation model.
It uses Encoder Decoder arthitect, which is widely wised in different tasks in NLP, such as Machines Translation, Question Answering, Image Captioning.

![](https://raw.githubusercontent.com/6chaoran/nlp/master/nmt/image/encoder-decoder-architecture.png)

The model consists of two major components:


* __Encoder__: a RNN network, used understand the input sequence and learning the pattern. 

* __Decoder__: a RNN netowrk, used to generate the sequence based on learned pattern from encoder.

major steps to train a seq2seq model:

1. tokenize the input (encoder) and output (decoder) sentences
2. (optional) input into embedding layers to learn the low dimensional representation of words. With a sizable corpus, embedding layers are highly recommended.
3. input source tokens/embedded array into encoder RNN (I used LSTM in this post) and learn the hidden states
4. pass the hidden states to decoder RNN as the initial states
5. input the sentence to be translated to decoder RNN, and target is the sentences which is one word shifted. In the structure, one word of decoder sentence is aimed to predict the next word, and this is called **teacher forcing** training.


What connects encoder and decoder is the hidden states, which is learned from encoder. 

## Dataset

The data used in this post is from [ManyThings.org](http://www.manythings.org/anki/). It provides toy datasets for many bilingual sentence pairs. I used [english-chinese dataset](http://www.manythings.org/anki/cmn-eng.zip).

## Prepare Data

### clean punucations
for english, I simply removed `,.!?` and convert to lower case
for chinese, I only removed `,.!?。，！？\n`

### tokenize
for english, I just split the sentence by space
for chinese, I used [jieba](https://github.com/fxsjy/jieba) parser to cut the sentence.

## Encoder

## Decoder

# Reference

* [Keras Blog](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)