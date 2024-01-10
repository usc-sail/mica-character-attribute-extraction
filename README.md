# Character Attribute Extraction

Character attribute extraction is an *information extraction* task to find attribute-frames from a narrative text.
Attribute frames are *(character, attribute-type, attribute-value, passage)* tuples, where the passage is taken from the
narrative text and it describes some attribute, of type defined by *attribute-type* and value represented by
*attribute-value*, of the character.

For example, consider the following passage.

```
The Sports Commentator is at the airport and about to interview the world heavyweight boxing champion, Apollo Creed. 
Creed is twenty-eight years old. He is a tall, smooth-muscled black man 
with barely a scar on his light coffee-colored face.
```

Assume **C** = "Apollo Creed" and **P** is the whole passage.
We can extract the following attribute frames from the above passage.

1. (**C**, profession, boxer, **P**)
2. (**C**, accomplishment, heavyweight champion, **P**)
3. (**C**, age, 28 years, **P**)
4. (**C**, physical appearance, "tall, smooth-muscled black man, light coffee-colored face", **P**)
5. (**C**, race, black, **P**)

The character attribute extraction task can be simplified to a question-answering task by providing the character,
passage, and attribute-type as inputs and asking the model to find the attribute-value from the passage.

### Directory Structure

The directory names are prefixed with numbers in their order of execution.

1. **01-attribute-types** - 
    Create a list of attribute-types by zero-shot prompting character descriptions.

2. **02-implicitness** - 
    Find the approximate attribute-type distribution of character descriptions by asking yes/no questions from Flan-T5.
    The answer probability of the response of Flan-T5 provides the implicitness score.

3. **03-prompt-creation** - 
    Create the zero-shot, few-shot, and chain-of-thought prompts for finding the attribute-values.

4. **04-prompting** -
    Perform the zero-shot, few-shot, and chain-of-thought prompting to find the attribute-values.

5. **05-error-analysis** -
    Analyze the errors made by different prompting methods.

### Data

The prompts used can be found at `data/prompts` folder.

The annotation data for this project can be found at `data/annotations.xlsx` file.

The first page contains annotation instructions for the raters.
It informs them that for each sample (character, passage, attribute-type, attribute-value), the rater can pick one of
four options based on whether they deem the model-provided attribute-value to be correct or not.
The options are "correct", "wrong", "unsure", and "invalid".
The page explains each of these four options.

Next, if the rater chose the "wrong" option (the rater judged the model's answer to be wrong), they should pick the
reason why attribute-value is wrong.
There are four options again: "answer exists", "different character", "different attribute", and "wrong/missing".
The page explains each of these four options.

Next, the page shows definitions for the attribute-types.
The rater is required to understand those before proceeding with their annotation task.
Lastly, the page shows a live counter of how many samples are annotated.

The remaining pages contain the samples for annotation.
Each annotation page contains columns for attribute-type, passage, character, model-provided attribute-value, two
columns for the human annotations, and an optional remarks column.
There is also a strategies column which mentions the prompting strategy used to find the attribute-value.
This column is hidden from the annotator.

### Citation

```
Baruah, Sabyasachee and Narayanan, Shrikanth, 
"Character Attribute Extraction from Movie Scripts using LLMs" 
ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 
COEX, Seoul, Korea, pp. xx, doi: xx/xx
```