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
    The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed.
    Repair work is ongoing in Hawick and many roads in Peeblesshire remain badly affected by standing water.
    Trains on the west coast mainline face disruption due to damage at the Lamington Viaduct.
    Many businesses and householders were affected by flooding in Newton Stewart after the River Cree overflowed into the town.
    First Minister Nicola Sturgeon visited the area to inspect the damage.
    The waters breached a retaining wall, flooding many commercial properties on Victoria Street - the main shopping thoroughfare.
    Jeanette Tate, who owns the Cinnamon Cafe which was badly affected, said she could not fault the multi-agency response once the flood hit.
    However, she said more preventative work could have been carried out to ensure the retaining wall did not fail.
    "It is difficult but I do think there is so much publicity for Dumfries and the Nith - and I totally appreciate that - but it is almost like we're neglected or forgotten," she said.
    "That may not be true but it is perhaps my perspective over the last few days.
    "Why were you not ready to help us a bit more when the warning and the alarm alerts had gone out?"
    Meanwhile, a flood alert remains in place across the Borders because of the constant rain.
    Peebles was badly hit by problems, sparking calls to introduce more defences in the area.
    Scottish Borders Council has put a list on its website of the roads worst affected and drivers have been urged not to ignore closure signs.
    The Labour Party's deputy Scottish leader Alex Rowley was in Hawick on Monday to see the situation first hand.
    He said it was important to get the flood protection plan right but backed calls to speed up the process.
    "I was quite taken aback by the amount of damage that has been done," he said.
    "Obviously it is heart-breaking for people who have been forced out of their homes and the impact on businesses."
    He said it was important that "immediate steps" were taken to protect the areas most vulnerable and a clear timetable put in place for flood prevention plans.
    Have you been affected by flooding in Dumfries and Galloway or the Borders? Tell us about your experience of the situation and how it was handled. Email us on selkirk.news@bbc.co.uk or dumfries@bbc.co.uk.
    '''

    Summary:
    '''
    Clean-up operations are continuing across the Scottish Borders and Dumfries and Galloway after flooding caused by Storm Frank.
    '''

    Summary word tokenization:
    '''
    ['Clean-up', 'operations', 'are', 'continuing', 'across', 'the', 'Scottish', 'Borders', 'and', 'Dumfries', 'and', 'Galloway', 'after', 'flooding', 'caused', 'by', 'Storm', 'Frank', '.']
    '''
- **Response**:
    {
      "unfaithful": true,
      "word unfaithful labels": [
        ["Clean-up", false],
        ["operations", false],
        ["are", false],
        ["continuing", false],
        ["across", false],
        ["the", false],
        ["Scottish", false],
        ["Borders", false],
        ["and", false],
        ["Dumfries", false],
        ["and", false],
        ["Galloway", false],
        ["after", false],
        ["flooding", false],
        ["caused", false],
        ["by", false],
        ["Storm", true],
        ["Frank", true],
        [".", false]
      ]
    }
"""
EXAMPLE_FAITHFUL = """
- **User Prompt**:
    Document:
    '''
    Ferrari appeared in a position to challenge until the final laps, when the Mercedes stretched their legs to go half a second clear of the red cars.
    Sebastian Vettel will start third ahead of team-mate Kimi Raikkonen.
    The world champion subsequently escaped punishment for reversing in the pit lane, which could have seen him stripped of pole.
    But stewards only handed Hamilton a reprimand, after governing body the FIA said "no clear instruction was given on where he should park".
    Belgian Stoffel Vandoorne out-qualified McLaren team-mate Jenson Button on his Formula 1 debut.
    Vandoorne was 12th and Button 14th, complaining of a handling imbalance on his final lap but admitting the newcomer "did a good job and I didn't".
    Mercedes were wary of Ferrari's pace before qualifying after Vettel and Raikkonen finished one-two in final practice, and their concerns appeared to be well founded as the red cars mixed it with the silver through most of qualifying.
    After the first runs, Rosberg was ahead, with Vettel and Raikkonen splitting him from Hamilton, who made a mistake at the final corner on his first lap.
    But Hamilton saved his best for last, fastest in every sector of his final attempt, to beat Rosberg by just 0.077secs after the German had out-paced him throughout practice and in the first qualifying session.
    Vettel rued a mistake at the final corner on his last lap, but the truth is that with the gap at 0.517secs to Hamilton there was nothing he could have done.
    The gap suggests Mercedes are favourites for the race, even if Ferrari can be expected to push them.
    Vettel said: "Last year we were very strong in the race and I think we are in good shape for tomorrow. We will try to give them a hard time."
    Vandoorne's preparations for his grand prix debut were far from ideal - he only found out he was racing on Thursday when FIA doctors declared Fernando Alonso unfit because of a broken rib sustained in his huge crash at the first race of the season in Australia two weeks ago.
    The Belgian rookie had to fly overnight from Japan, where he had been testing in the Super Formula car he races there, and arrived in Bahrain only hours before first practice on Friday.
    He also had a difficult final practice, missing all but the final quarter of the session because of a water leak.
    Button was quicker in the first qualifying session, but Vandoorne pipped him by 0.064secs when it mattered.
    The 24-year-old said: "I knew after yesterday I had quite similar pace to Jenson and I knew if I improved a little bit I could maybe challenge him and even out-qualify him and that is what has happened.
    "Jenson is a very good benchmark for me because he is a world champion and he is well known to the team so I am very satisfied with the qualifying."
    Button, who was 0.5secs quicker than Vandoorne in the first session, complained of oversteer on his final run in the second: "Q1 was what I was expecting. Q2 he did a good job and I didn't. Very, very good job. We knew how quick he was."
    The controversial new elimination qualifying system was retained for this race despite teams voting at the first race in Australia to go back to the 2015 system.
    FIA president Jean Todt said earlier on Saturday that he "felt it necessary to give new qualifying one more chance", adding: "We live in a world where there is too much over reaction."
    The system worked on the basis of mixing up the grid a little - Force India's Sergio Perez ended up out of position in 18th place after the team miscalculated the timing of his final run, leaving him not enough time to complete it before the elimination clock timed him out.
    But it will come in for more criticism as a result of lack of track action at the end of each session. There were three minutes at the end of the first session with no cars on the circuit, and the end of the second session was a similar damp squib.
    Only one car - Nico Hulkenberg's Force India - was out on the track with six minutes to go. The two Williams cars did go out in the final three minutes but were already through to Q3 and so nothing was at stake.
    The teams are meeting with Todt and F1 commercial boss Bernie Ecclestone on Sunday at noon local time to decide on what to do with qualifying for the rest of the season.
    Todt said he was "optimistic" they would be able to reach unanimous agreement on a change.
    "We should listen to the people watching on TV," Rosberg said. "If they are still unhappy, which I am sure they will be, we should change it."
    Red Bull's Daniel Ricciardo was fifth on the grid, ahead of the Williams cars of Valtteri Bottas and Felipe Massa and Force India's Nico Hulkenberg.
    Ricciardo's team-mate Daniil Kvyat was eliminated during the second session - way below the team's expectation - and the Renault of Brit Jolyon Palmer only managed 19th fastest.
    German Mercedes protege Pascal Wehrlein managed an excellent 16th in the Manor car.
    Bahrain GP qualifying results
    Bahrain GP coverage details
    '''

    Summary:
    '''
    Lewis Hamilton stormed to pole position at the Bahrain Grand Prix ahead of Mercedes team-mate Nico Rosberg.
    '''

    Summary word tokenization:
    '''
    ['Lewis', 'Hamilton', 'stormed', 'to', 'pole', 'position', 'at', 'the', 'Bahrain', 'Grand', 'Prix', 'ahead', 'of', 'Mercedes', 'team-mate', 'Nico', 'Rosberg', '.']
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
            failure_prompt = FAILURE_PROMPT.format(error=error)
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

        json_filepath = os.path.join(parent_path, sample['id'] + ".json")
        with open(json_filepath, 'w') as f:
            json.dump(json_data, f, indent=4)


