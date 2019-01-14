# Build A Simple Machine Translator (part-1)
* encoder decoder framework with lstm *

# Introduction

[seq2seq model](https://google.github.io/seq2seq/) is a general purpose sequence learning and generation model.
It uses Encoder Decoder arthitect, which is widely wised in different tasks in NLP, such as Machines Translation, Question Answering, Image Captioning.

![](https://raw.githubusercontent.com/6chaoran/nlp/master/nmt/image/encoder-decoder-architecture.png)

The model consists of two major components:


* __Encoder__: a RNN network, used understand the input sequence and learning the pattern. 

* __Decoder__:  another RNN netowrk, used to generate the sequence based on learned pattern from encoder.

What connects encoder and decoder is the hidden states, which is learned from encoder. 
With a sizable corpus, embedding layers are also need to added before RNN layers to compress the dimensions. 

# Reference

* [Keras Blog](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)