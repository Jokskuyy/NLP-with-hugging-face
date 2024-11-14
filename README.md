# NLP with hugging face
 
# Sentiment Analysis & Text Classification Project

This project showcases various NLP tasks such as sentiment analysis, topic classification, text generation, named entity recognition (NER), question answering, text summarization, and translation. The tasks are implemented using pre-trained models from the Hugging Face Transformers library.

## Features

1. **Sentiment Analysis**  
   Analyzes the sentiment of a given text and categorizes it as negative, neutral, or positive. This task uses a model fine-tuned on Twitter data, suitable for analyzing tweets.

2. **Topic Classification**  
   Classifies a given sentence into one or more topics using zero-shot classification. It leverages a model trained on natural language inference (NLI) tasks.

3. **Text Generation**  
   Generates text based on an initial prompt using GPT-2, a powerful language model that can create human-like text.

4. **Named Entity Recognition (NER)**  
   Extracts entities such as person names, locations, and organizations from a given sentence. This task uses a model trained specifically for NER tasks.

5. **Question Answering**  
   Answers a given question based on the context provided. It uses a model fine-tuned on the SQuAD dataset for question answering tasks.

6. **Text Summarization**  
   Summarizes long paragraphs into concise summaries. This task is powered by a model trained for abstractive summarization.

7. **Translation**  
   Translates text from English to Indonesian, allowing for multilingual text processing.

## Requirements

- Python 3.6+
- Hugging Face `transformers` library
- `torch` library (PyTorch)