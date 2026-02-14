ZS_NAIVE_PROMPT_STR_FOR_MISTRAL = {
    "cnn": "Here is a news article:\n<text>\n\n"
    "Please write a short summary for the article in 1-2 sentences.",
    
    "arxiv": "Here is a research paper:\n<text>\n\n"
    "Please write a short abstract in about 3 sentences.",
    
    "pubmed": "Here is a research paper:\n<text>\n\n"
    "Please write a short abstract in about 3 sentences.",
    
    "samsum": "Here is a conversation:\n<text>\n\n"
    "Please write a short 1 sentence summary.",
    
    "meetingbank": "Here is a conversation:\n<text>\n\n"
    "Please write a 2-3 sentence summary.",
}

ZS_KEYWORD_PROMPT_STR_FOR_MISTRAL = {
    "cnn": "Here is a news article:\n<text>\n\n"
    "Please write a short summary for the article in 1-2 sentences. "
    "Consider include the following information: <keywords>",
    
    "arxiv": "Here is a research paper:\n<text>\n\n"
    "Please write a short abstract in about 3 sentences. "
    "Consider include the following information: <keywords>",
    
    "pubmed": "Here is a research paper:\n<text>\n\n"
    "Please write a short abstract in about 3 sentences. "
    "Consider include the following information: <keywords>",
    
    "samsum": "Here is a conversation:\n<text>\n\n"
    "Please write a short 1 sentence summary. "
    "Consider include the following information: <keywords>",
    
    "meetingbank": "Here is a conversation:\n<text>\n\n"
    "Please write a 2-3 sentence summary. "
    "Consider include the following information: <keywords>",
}

ZS_SELF_CORRECT_PROMPT_STR_FOR_MISTRAL = {
    "cnn": "Here is a news article:\n<text>\n\n"
    "Please write a short summary for the article in 1-2 sentences. "
    "Consider include the following information: <keywords>. "
    "Guideline: Ensure factuality. If a keyword is negated, adapt it concisely (e.g., use 'not happy' instead of 'happy'). "
    "Do not explain your corrections or add filler text. Keep it brief.",
    
    "arxiv": "Here is a research paper:\n<text>\n\n"
    "Please write a short abstract in about 3 sentences. "
    "Consider include the following information: <keywords>. "
    "Guideline: Ensure factuality. If a keyword is negated, adapt it concisely (e.g., use 'did not increase'). "
    "Do not explain your corrections. Keep it brief.",
    
    "pubmed": "Here is a research paper:\n<text>\n\n"
    "Please write a short abstract in about 3 sentences. "
    "Consider include the following information: <keywords>. "
    "Guideline: Ensure factuality. If a keyword is negated, adapt it concisely. "
    "Do not explain your corrections. Keep it brief.",
    
    "samsum": "Here is a conversation:\n<text>\n\n"
    "Please write a short 1 sentence summary. "
    "Consider include the following information: <keywords>. "
    "Guideline: Ensure factuality. If a keyword is negated, adapt it concisely. "
    "Do not explain your corrections. Keep it brief.",
    
    "meetingbank": "Here is a conversation:\n<text>\n\n"
    "Please write a 2-3 sentence summary. "
    "Consider include the following information: <keywords>. "
    "Guideline: Ensure factuality. If a keyword is negated, adapt it concisely. "
    "Do not explain your corrections. Keep it brief.",
}

ZS_NAIVE_PROMPT_STR_FOR_CLAUDE = {
    "cnn": "Here is a news article:\n<text>\n\nPlease write a summary for the article in 2-3 sentences.",
    "arxiv": "Here is a research paper:\n<text>\n\nPlease write a comprehensive paper abstract section.",
    "pubmed": "Here is a research paper:\n<text>\n\nPlease write a paper abstract in around 5 sentences.",
    "samsum": "Here is a conversation:\n<text>\n\nPlease write a very short 1 sentence summary.",
    "meetingbank": "Here is a conversation:\n<text>\n\nPlease write a summary in about 5 sentences.",
}

# Same for keyword/self-correct Claude prompts...