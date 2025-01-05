# Evaluating Summary Faithfulness using LLMs

The project can be splitted into two different subprojects:

## Generate an Artificial Dataset for Summary faithfulness
- llama.py - Base Class to predict with Llama 3.2 3B.
- xsum.py - Loads the Xsum Dataset, the base dataset for creating an artificial one.
- generate_artifical_dataset.py - Generates an artificial dataset using Xsum and Llama for summary faithfulness.

The dataset generated has the following structure:
- document: original document in XSum
- summary: original summary in XSum
- summary_word_tokenization: tokenized summary wil nltk
- response: response from Llama 3.2 3B in a valid json format

The model response has the following structure:
- unfaithful: bool
  - if the summary is unfaithful to the document
- word_unfaithful_labels: list[ list[ str, bool ] ]
  - word-labels with the first element being the word and the second element being the boolean unfaithful label.
 
## Finetune BERT
- artificial_dataset.py - load the artificial dataset as a pytorch Dataset ready to be used to finetune the BERT.
- bert.py - creates a model with BERT as backbone and two classification heads: one for text classification and another for per token classification.
- finetune_bert.py - finetune a BERT model using the custom BERT architecture and the artificial dataset. 

## Requirements
To run this project you need to install the packages stated in the requirements.txt file.
In addition, you will also need to have an environment variable called HF_TOKEN set with your Hugging Face API token.
Remember you need to have access to the Llama 3.2 3B model to run the project.