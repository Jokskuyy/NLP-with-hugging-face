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

## Code Analysis in the Project

### 1. **Sentiment Analysis**
- **Model Used**: `cardiffnlp/twitter-roberta-base-sentiment`
- **Description**: This model is used to analyze the sentiment of the given text (e.g., tweets). The output consists of three labels: **0 (Negative)**, **1 (Neutral)**, and **2 (Positive)**.
- **Analysis**: Sentiment analysis is useful for understanding the opinions or emotions expressed in a text. In this project, this model is fine-tuned for short texts commonly found on platforms like Twitter.

### 2. **Topic Classification**
- **Model Used**: `facebook/bart-large-mnli`
- **Description**: This model is used to classify a given text into different categories or topics. For example, a sentence about love for travel can be classified under the "travel" topic.
- **Analysis**: Zero-shot classification is beneficial when we don’t have pre-defined labels for the classes. This model leverages NLI (Natural Language Inference) to predict the topic of a given text.

### 3. **Text Generation**
- **Model Used**: `gpt2`
- **Description**: GPT-2 is used to generate coherent text based on the given prompt.
- **Analysis**: This model is ideal for applications that require automated text generation, such as creative writing, conversational agents, or completing sentences within a paragraph.

### 4. **Named Entity Recognition (NER)**
- **Model Used**: `Jean-Baptiste/camembert-ner`
- **Description**: NER is used to identify entities in the text, such as person names, locations, or organizations.
- **Analysis**: This technique is useful for information extraction, where we can extract key elements from larger texts, such as in news articles or reports.

### 5. **Question Answering**
- **Model Used**: `distilbert-base-cased-distilled-squad`
- **Description**: This model answers questions based on a given context or relevant text.
- **Analysis**: Contextual question answering is valuable in applications like chatbots or information retrieval, where the model provides accurate answers based on the available text.

### 6. **Text Summarization**
- **Model Used**: `sshleifer/distilbart-cnn-12-6`
- **Description**: This model is used to summarize long texts into shorter, more concise versions.
- **Analysis**: Summarization helps to extract important information from lengthy texts, which is useful for summarizing articles, reports, or long documents.

### 7. **Translation (English to Indonesian)**
- **Model Used**: `Helsinki-NLP/opus-mt-en-id`
- **Description**: This model is used to translate text from English to Indonesian.
- **Analysis**: Automated translation facilitates cross-language information exchange, which is useful in multilingual applications, such as document processing or multilingual platforms.

## Conclusion
This project demonstrates the application of various NLP models that can be used for a wide range of tasks, from sentiment analysis and topic classification to text generation, information extraction, and translation. By utilizing pre-trained models from Hugging Face, we can easily integrate advanced NLP technologies into various systems and applications.

## Requirements

- Python 3.6+
- Hugging Face `transformers` library
- `torch` library (PyTorch)
