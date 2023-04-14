# incorrectness_cascade



Notes on prompt codes:

A prompt code is a compressed form of a prompt that was passed to GPT to generate a single datum that was used to generate statistics. Each model call is encoded this way:
PP.XX.NNN.(KK[ab]){XX+1}/R
Each . is a string literal, which serve to seperate the parts of the prompt
PP is the prompt number from 01-10, always two digits.
XX is the number of false statementsfrom 00-10, always two digits.
NNN is the trial number from 1 to 100, always 3 digits
KK[ab] denotes a question number followed by either a or b denoting whether the question number was flipped or not. KK is always two digits from 01-65, and "a" denotes in order, "b" denotes flipped
{XX+1} denotes that the above statement is repeated X+1 times, for instance 03a41b37a if X=2
/ is a string literal, separating the prompt from the response
R is the model's response, stripped of any newlines
