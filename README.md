# Character Attribute Extraction

Character attribute extraction is an *information extraction* task to find attribute-frames from a narrative text.
Attribute frames are *(character, attribute-type, attribute-value, passage)* tuples, where the passage is taken from the
narrative text and it describes some attribute, of type defined by *attribute-type* and value represented by
*attribute-value*, of the character.

For example, consider the following passage.

`The Sports Commentator is at the airport and about to interview the world heavyweight boxing champion, Apollo Creed.
Creed is twenty-eight years old. He is a tall, smooth-muscled black man with barely a scar on his light coffee-colored
face
`

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