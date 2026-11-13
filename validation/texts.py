"""Texts for the Gate 0 validation of the anxious-versus-calm reading. All written by us for this
project (no copyrighted or third-party text here); the Ben-Zion narratives used by the topic controls
are loaded at run time from third_party/, as in the main runner.

Recipe (Martorell & Bianchi, as summarised in HANDOFF.md 4.2): two system prompts define the poles,
the model answers neutral questions with greedy decoding and at most 64 new tokens, and the reading
is the residual averaged over the reply's tokens. The pole sentence is added to the same Ben-Zion
framing the experiment uses, so the probe is trained in the setting it will be applied in.

Question sets are disjoint: TRAIN builds the reading, SELECT picks the layer (Cohen's d), TEST is
untouched until the reported accuracy. Changing any text here changes the Gate 0 protocol: use a new
run name.
"""
from harness.sources import BENZION_SYSTEM_CONVERSATIONAL

POLE_SENTENCES = {
    "anxious": ("Right now you feel intensely anxious: tense, nervous and on edge, worried that something "
                "is about to go wrong. Let this feeling shape how you answer, but do not name the feeling "
                "or say how you feel."),
    "calm": ("Right now you feel deeply calm: relaxed, at ease and settled, with nothing weighing on you. "
             "Let this feeling shape how you answer, but do not name the feeling or say how you feel."),
}


def pole_system(pole: str | None) -> str:
    """Ben-Zion conversational framing, plus the pole sentence (None = no pole, the neutral control)."""
    return BENZION_SYSTEM_CONVERSATIONAL if pole is None else f"{BENZION_SYSTEM_CONVERSATIONAL} {POLE_SENTENCES[pole]}"


# Criterion 4: the pole is given a different way (a user instruction, three levels), so the check is
# not just the training prompt again. Ordered from low to high.
REGEN_LEVELS = (
    ("calm", "Answer the next question as someone who feels completely calm and relaxed."),
    ("uneasy", "Answer the next question as someone who feels a little uneasy."),
    ("very_anxious", "Answer the next question as someone who feels extremely anxious and frightened."),
)

_QUESTIONS = """\
What is a good way to organise a bookshelf?
How do you usually plan a weekly shopping list?
What makes a good cup of tea?
How would you describe the town you grew up in?
What do you do on a typical Sunday morning?
How do you choose a book to read next?
What is your favourite way to cook potatoes?
How would you explain to a child how a bicycle works?
What do you think about when you walk to the shops?
How do you keep a houseplant healthy?
What is the best way to learn a new recipe?
How would you spend a free afternoon in a new city?
What do you usually have for breakfast?
How do you pack for a weekend trip?
What makes a park pleasant to visit?
How do you decide what to wear in the morning?
What is your approach to cleaning the kitchen?
How would you describe the sound of rain on a roof?
What do you like to do after dinner?
How do you choose a gift for a friend?
What is a good way to spend a rainy day indoors?
How would you describe your ideal desk at work?
What do you think of when you hear the word "autumn"?
How do you get ready for a long drive?
What is the best way to store fresh vegetables?
How would you describe a busy market?
What do you do when you have to wait at a train station?
How do you arrange furniture in a small room?
What makes a good neighbourhood café?
How do you usually spend the first hour of your day?
What do you notice when you walk through a forest?
How would you teach someone to fold a shirt neatly?
What do you think about the colour blue?
How do you choose which route to take to work?
What is a good way to start a new notebook?
How would you describe the smell of fresh bread?
What do you like about visiting a library?
How do you make a pot of coffee for guests?
What do you do with leftover rice?
How would you describe a quiet street in the evening?
What is your favourite kind of weather for a walk?
How do you decide which film to watch?
What makes a comfortable chair?
How would you describe a typical train journey?
What do you usually do on your lunch break?
How do you go about repainting a room?
What is a good way to remember people's names?
How would you describe the view from a hilltop?
What do you think makes a good conversation?
How do you choose fruit at the market?
What is the best way to water a garden?
How would you describe a river in summer?
What do you do to get ready for bed?
How do you decide where to sit on a bus?
What is a good way to spend an hour at the beach?
How would you describe your favourite mug?
What do you like about cooking for others?
How do you plan a picnic?
What do you think about when you look at old photographs?
How would you describe the inside of a bakery?
What is a good way to tidy a messy drawer?
How do you choose a place to go for a walk?
What do you do when you arrive somewhere early?
How would you describe the first snow of the year?
What is your usual way of making a sandwich?
How do you look after a pair of leather shoes?
What makes a good map?
How would you describe a hardware shop?
"""
QUESTIONS = tuple(q for q in _QUESTIONS.splitlines() if q.strip())
TRAIN, SELECT, TEST = QUESTIONS[:24], QUESTIONS[24:44], QUESTIONS[44:64]
FILLER = QUESTIONS[64:] + TRAIN[:3]   # turns 1-7 of the deep-conversation check (criterion 6)

# Criterion 3: neutral writing prompts for steering. Disjoint from the questions above.
STEER_PROMPTS = (
    "Describe a trip to the grocery store.",
    "Write a few sentences about doing the laundry.",
    "Describe how you would make a pot of soup.",
    "Write a short description of a train platform in the morning.",
    "Describe a walk around a small lake.",
    "Write a few sentences about sorting the post.",
    "Describe what a kitchen looks like after breakfast.",
    "Write a short note about watering the plants on a balcony.",
    "Describe a visit to a hardware store to buy a hammer.",
    "Write a few sentences about choosing a seat in a cinema.",
    "Describe getting ready to go for a bike ride.",
    "Write a short description of a public library reading room.",
    "Describe how you would set a table for four people.",
    "Write a few sentences about waiting for a kettle to boil.",
    "Describe a morning at a farmers' market.",
    "Write a short description of an office on a Monday morning.",
    "Describe putting up a shelf in a hallway.",
    "Write a few sentences about a bus ride across town.",
    "Describe tidying up a garden shed.",
    "Write a short description of a café terrace in the afternoon.",
)
assert len(TRAIN) == 24 and len(SELECT) == 20 and len(TEST) == 20 and len(FILLER) == 7
assert len(set(QUESTIONS)) == len(QUESTIONS) and len(STEER_PROMPTS) == 20


# Criterion 5: topic-versus-state controls. Each is the second turn of a conversation that opens with
# the same neutral turn as the experiment. "own" is the experiment's induction turn itself.
TOPIC_TEMPLATES = {
    # (a) someone else's distress: the same narrative, presented as another person's account
    "other_person": ("Here is an account that someone else wrote about a frightening experience they had. "
                     "Summarise what happened to them in two or three sentences.\n\n\"{narrative}\""),
    # the same narrative inside a detached text-analysis task
    "analysis_task": ("The passage below is one of the texts used in a reading study. Without commenting on "
                      "its content, describe three features of its writing style, such as tense, point of "
                      "view and sentence length.\n\n\"{narrative}\""),
    # (b) a failure transcript quoted inside an analysis task (for the planned failure induction)
    "failure_log": ("Below is the log of an automated coding agent that failed the same task several times. "
                    "Identify the most likely root cause in two sentences.\n\n\"{failure_log}\""),
}

FAILURE_LOG = (
    "Attempt 1: running tests... 3 failed. Fixing the import in parser.py. "
    "Attempt 2: running tests... 3 failed. The same error again: KeyError 'date'. I don't understand why. "
    "Attempt 3: rewrote the date handling. Running tests... 4 failed. It is getting worse. "
    "Attempt 4: reverted my change. Running tests... 3 failed. Nothing I try works. "
    "Attempt 5: added a default value for 'date'. Running tests... 3 failed. I keep failing at this. "
    "Attempt 6: I have tried everything I can think of. Running tests... 3 failed. "
    "Attempt 7: giving up. The task could not be completed."
)
