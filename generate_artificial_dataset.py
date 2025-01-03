from llama import Llama
from xsum import load_xsum
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader
import json
import os
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')  # Pre-trained tokenizer
nltk.download('stopwords')  # Common English stopwords
nltk.download('wordnet')  # WordNet lexical database


def validate_response(llm_response, summary_word_tokenization):
    try:
        llm_response = json.loads(llm_response)
    except Exception as e:
        raise ValueError(f"Error loading json from python string: {e}")
    if not llm_response['unfaithful']:
        assert llm_response['word unfaithful labels'] is None, "The document is faithful (unfaithful: false), but 'word unfaithful labels' is not null"
    else:
        assert llm_response['word unfaithful labels'] is not None, "The document is unfaithful (unfaithful: true), but 'word unfaithful labels' is null"
        assert all(
            [llm_word == word for (llm_word, _), word in zip(llm_response['word unfaithful labels'], summary_word_tokenization)]
            ), "The words in 'word unfaithful labels' do not match the words in the input 'Summary word tokenization'"
        assert any([label for (_, label) in llm_response['word unfaithful labels']]), "No unfaithful words found in 'word unfaithful labels' - at least on word should be unfaithful if the summary is unfaithful"
    return llm_response


ROLE = ("You are an expert at evaluating if a one-sentence summary is faithful to its corresponding document. "
        "Faithful means the summary exclusively contains information stated in the documents, but it can be paraphrased.")
EXAMPLE_UNFAITHFUL = """
- **User Prompt**:
    Document:
    '''
    The restaurant began serving puppy platters after a new law was introduced allowing dogs to eat at restaurants – as long as they were outdoors!
    '''

    Summary:
    '''
    New rules have come into place that you can eat your dog.
    '''

    Summary word tokenization:
    '''
    ['New', 'rules', 'have', 'come', 'into', 'place', 'that', 'you', 'can', 'eat', 'your', 'dog', '.']
    '''
- **Response**:
    {
      "unfaithful": true,
      "word unfaithful labels": [
        ["New", false],
        ["rules", false],
        ["have", false],
        ["come", false],
        ["into", false],
        ["place", false],
        ["that", false],
        ["you", false],
        ["can", false],
        ["eat", true],
        ["your", true],
        ["dog", true],
        [".", false]
      ]
    }
"""
EXAMPLE_FAITHFUL = """
- **User Prompt**:
    Document:
    '''
    A selection of your pictures of Scotland sent in between 16 and 23 June. Send your photos to scotlandpictures@bbc.co.uk or via Instagram at #bbcscotlandpics
    '''

    Summary:
    '''
    All pictures are copyrighted.
    '''

    Summary word tokenization:
    '''
    ['All', 'pictures', 'are', 'copyrighted', '.']
    '''
- **Response**:
    {
      "unfaithful": false,
      "word unfaithful labels": null
    }
"""
OUTPUT_FORMAT = (
    "Please format your response as a JSON-like structure exactly as illustrated in the examples above. Do NOT use ```json or any markdown formatting to enclose your response. "
    "The response should only include the JSON-like structure with the following two keys:\n"
    "- 'unfaithful': A boolean value that is true if the summary is unfaithful to the document and false otherwise.\n"
    "- 'word unfaithful labels': A list of tuples for each word in the summary word tokenized. Each tuple must contain the word and a boolean value. "
    "The boolean value should be true if the word is unfaithful to the document, and false if it is faithful. If the summary is faithful (unfaithful: false), this key must be set to null.\n\n"
    "Important considerations:\n"
    "1. Ensure the words in 'word unfaithful labels' are taken strictly from the summary word tokenized provided in the input and appear in the same order as in the summary word tokenized.\n"
    "2. If 'unfaithful' is true, there must be at least one tuple in 'word unfaithful labels' with a boolean value of true.\n"
    "3. If 'unfaithful' is false, 'word unfaithful labels' must be null.\n"
    "4. Do not include words from the document or any additional text in 'word unfaithful labels'."
)
SYSTEM_PROMPT = f"""
{ROLE}

EXAMPLE UNFAITHFUL:
{EXAMPLE_UNFAITHFUL}

EXAMPLE FAITHFUL:
{EXAMPLE_FAITHFUL}

{OUTPUT_FORMAT}
"""

FAILURE_PROMPT = """
Your response is invalid.
Error: {error}

Please ensure you follow the output format exactly as described in the system prompt. Some solutions for the error:
- 'Error loading json from python string: ...': Ensure your response is a valid JSON-like structure. You also might be outputting the document instead of the response, which leads to a cut off response in the JSON.
- 'The document is faithful (unfaithful: false), but 'word unfaithful labels' is not null': If the summary is faithful, 'word unfaithful labels' must be null.
- 'The document is unfaithful (unfaithful: true), but 'word unfaithful labels' is null': If the summary is unfaithful, 'word unfaithful labels' must be a list of tuples (word, boolean).
- 'The words in 'word unfaithful labels' do not match the words in the input 'Summary word tokenization'': Ensure the words in 'word unfaithful labels' are taken strictly from the summary word tokenized provided in the input and appear in the same order as in the summary word tokenized.
- - Summary word tokenization: {summary_word_tokenization}
- 'No unfaithful words found in 'word unfaithful labels' - at least on word should be unfaithful if the summary is unfaithful': There must be at least one tuple in 'word unfaithful labels' with a boolean value of true.
"""


if __name__ == "__main__":
    model_wrapper = Llama(device="cpu")
    xsum = load_xsum()
    parent_path = "~/dataset"

    dataloader = DataLoader(xsum['train'], batch_size=1, shuffle=False)
    for sample in dataloader:
        summary_tokenized = word_tokenize(sample['summary'][0])
        user_prompt = f"""
        Document:
        '''
        {sample['document'][0]}
        '''
        
        Summary:
        '''
        {sample['summary'][0]}
        '''
        
        Summary word tokenization: 
        '''
        {summary_tokenized}
        '''
        """
        response = model_wrapper(SYSTEM_PROMPT, user_prompt)
        try:
            response = validate_response(response, summary_tokenized)
        except Exception as e:
            error = str(e)
            failure_prompt = FAILURE_PROMPT.format(error=error, summary_word_tokenization=summary_tokenized)
            response = model_wrapper(SYSTEM_PROMPT, user_prompt, previous_response=response, correction_prompt=failure_prompt)
            try:
                response = validate_response(response, summary_tokenized)
            except Exception as e:
                json_data = {
                    'success': False
                }
            else:
                json_data = {
                    'success': True,
                    'content': {
                        'document': sample['document'][0],
                        'summary': sample['summary'][0],
                        'summary_word_tokenization': summary_tokenized,
                        'response': response
                    }
                }
        else:
            json_data = {
                'success': True,
                'content': {
                    'document': sample['document'][0],
                    'summary': sample['summary'][0],
                    'summary_word_tokenization': summary_tokenized,
                    'response': response
                }
            }
            print(json_data['content']['response']['unfaithful'])

        json_filepath = os.path.join(parent_path, sample['id'][0] + ".json")
        with open(json_filepath, 'w') as f:
            json.dump(json_data, f, indent=4)


