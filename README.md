# TODO: Title

The project can be splitted into two different subprojects:

## Generate an Artificial Dataset for Summary faithfulness
- llama.py - Base Class to predict with Llama 3.2 3B.
- xsum.py - Loads the Xsum Dataset, the base dataset for creating an artificial one.
- generate_artifical_dataset.py - Generates an artificial dataset using Xsum and Llama for summary faithfulness.
 
## Finetune BERT
- artificial_dataset.py - load the artificial dataset as a pytorch Dataset ready to be used to finetune the BERT.
- bert.py - creates a model with BERT as backbone and two classification heads: one for text classification and another for per token classification.
- finetune_bert.py - finetune a BERT model using the custom BERT architecture and the artificial dataset. 