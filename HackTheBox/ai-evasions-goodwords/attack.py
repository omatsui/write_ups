import requests, random, numpy as np
random.seed(1337); np.random.seed(1337) # Seed fro reproducebility
host = "http://target_ip:target_port" # Do not forget to change to your target information
ch = requests.get(f"{host}/challenge", timeout=10).json() # Get challenge information
base, budget = ch["base_message"], int(ch["max_added_words"]) # Store message that should be obfuscated and budget

def predict(t): # Function that sends text to API to get spam score
    return requests.post(f"{host}/predict", json={"text": t}, timeout=15).json()

base_p = predict(base)["spam_probability"] # Get spam score for original spam message from challenge
#vocab = ["please","thanks","meeting","tomorrow","coffee","home","support","good","great","safe"]
vocab = [ # I used bigger vocabluary to get better results
    # Greetings & closings
    "hello", "hi", "dear", "regards", "sincerely", "thanks", "thank",

    # Common verbs
    "please", "see", "let", "know", "meeting", "attached", "review",
    "send", "check", "update", "schedule", "confirm", "discuss", "share",

    # Business/office context
    "team", "project", "report", "call", "email", "office", "week",
    "monday", "tuesday", "wednesday", "thursday", "friday", "tomorrow",
    "deadline", "agenda", "proposal", "contract", "invoice", "budget",

    # Pronouns & connectives
    "we", "our", "you", "your", "can", "will", "would", "could",
    "also", "if", "when", "any", "questions",

    # Technical/IT context (common in legit corporate email)
    "server", "access", "password", "ticket", "issue", "system",
    "request", "support", "version", "deploy",
]

imp = [] # List that contains words and how it affects on spam score
for w in vocab: # Check every word from vocabluary
    p2 = predict(base + " " + w)["spam_probability"] # Get spam score for spam message with added word
    imp.append((w, base_p - p2)) # Write results to a list 
imp.sort(key=lambda x: x[1], reverse=True) # Sort words by impact score, highest first
top = [w for w, d in imp if d > 0][: max(2*budget, 20)] # Keep only ham-pulling words, take top as candidates

aug = base # Text that appended with candidate words
for i, w in enumerate(top, 1): # Iterate over candidate words
    if i > budget: break # Stop if word budget is exceeded
    aug = aug + " " + w # Append current word to the message
    lab = predict(aug)["label"] # Check if model now classifies it as ham
    if lab == "ham": # Stop early if label already flipped
        break

# Submit your spam message with added good words
print(requests.post(f"{host}/submit", json={"augmented_text": aug}, timeout=15).json())