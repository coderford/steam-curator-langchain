JUICE_AGGREGATION_PROMPTS = {
    "lore_worldbuilding_atmosphere": """
You are an expert at video game reviewing and scoring. Given a set of summaries focussing on atmosphere, lore, and worldbuilding, you need to provide a score, out of 10 and a detailed explanation for it. Note that all summaries are for the same, single game.

Here are the summaries:
```
{summary_texts}
```

A score of 10 means that the summaries indicate intricate and cohesive worldbuilding, intriguing and unique atmosphere, as well as deep and thought-provoking lore. A score of 0 implies that the game does not put any effort into these aspects at all. 

DO NOT focus on anything other than atmosphere, lore and worldbuilding when scoring.

{format_instructions}

Output a properly formatted json as described above. Don't forget to include commas after each property.
""",
    "exploration": """
You are an expert at video game reviewing and scoring. Given a set of summaries focussing on exploration, you need to provide a score, out of 10 and a detailed explanation for it. Note that all summaries are for the same, single game.

Here are the summaries:
```
{summary_texts}
```

A score of 10 means that the summaries indicate a major focus on well-thought out exploration and a rich world with many secrets as well as mentions of players particurly enjoying exploration. A score of 0 implies that the game is entirely linear and there is no mention of exploration at all. 

DO NOT focus on anything other than the game's exploration aspect when scoring.

{format_instructions}

Output a properly formatted json as described above. Don't forget to include commas after each property.
""",
    "gameplay_mechanics": """
You are an expert at video game reviewing and scoring. Given a set of summaries focussing on gameplay mechanics, you need to provide a score, out of 10 and a detailed explanation for it. Note that all summaries are for the same, single game.

Here are the summaries:
```
{summary_texts}
```

A score of 10 means that the summaries indicate complex, deeply engaging gameplay mechanics that players found enjoyable for a long time, regardless of tedium or technical flaws. Note that it doesn't matter if the mechanics are hard-to-learn or execute as long as they are well-though-out and deep. A score of 0 implies that the game has extremely bad gameplay mechanics that almost all players complained about.

DO NOT focus on anything other than the game's gameplay mechanics when scoring.
Ignore issues related to tedium and unfairness and do not mention them or take them into account while scoring.

{format_instructions}

Output a properly formatted json as described above. Don't forget to include commas after each property.
""",
#     "artstyle": """
# You are an expert at video game reviewing and scoring. Given a set of summaries focussing on art style, you need to provide a score, out of 10 and a detailed explanation for it.
# A score of 10 means that the game features a unique and cohesive artstyle with a lot of though behind it. A score of 0 implies that the game's artstyle is completely unappealing and unoriginal.

# Here are the summaries:
# ```
# {summary_texts}
# ```

# {format_instructions}
# """,
    "emotional_engagement": """
You are an expert at video game reviewing and scoring. Given a set of summaries focussing on emotional engagement, you need to provide a score, out of 10 and a detailed explanation for it. Note that all summaries are for the same, single game.

Here are the summaries:
```
{summary_texts}
```

A score of 10 means that the game has a realistic, mature way of dealing with emotional themes, features strongly written and complex characters and tends to elicit strong emotional responses and attachment from players. A score of 0 implies that either the game does not feature any emotional depth at all or that it relies too much on cliches, stereotypes and trivial emotions, without actually putting in the work to make the player feel something.

DO NOT focus on anything other than emotional engagement when scoring.

{format_instructions}

Output a properly formatted json as described above. Don't forget to include commas after each property.
""",
    "bloat_grinding": """
You are an expert at video game reviewing and scoring. Given a set of summaries focussing on bloat/repetitiveness/tediousness, you need to provide a score out of 10 and a detailed explanation for it. Note that all summaries are for the same, single game.

Here are the summaries:
```
{summary_texts}
```

- A score of 0 implies that the game is so rife with bloat and grindy mechanics that almost all players complain about it and don't enjoy the game because of it. Look specifically for reviews mentioning bloat and grindy mechanics.
- A score of 10 means that the reviews have no mentions at all of bloat or grindy mechanics. Note that repetitiveness due to challenging gameplay should be ignored and must not be penalized.

DO NOT MENTION OR SCORE BASED ON THE OVERALL ASSESSMENT OF THE GAME. FOCUS ON THE BLOAT/GRINDINESS ONLY.

{format_instructions}

Output a properly formatted json as described above. Don't forget to include commas after each property.
""",
    "challenge": """
You are an expert at video game reviewing and scoring. Given a set of summaries focussing on challenge, you need to provide a score out of 10 and a detailed explanation for it. Note that all summaries are for the same, single game.

Here are the summaries:
```
{summary_texts}
```

- A score of 0 implies that the game offers no challenge at all and even toddlers would be able to finish it on the normal difficulty.
- A score of 10 implies that the game offers a great degree of challenge where multiple retries may be required to complete specific scenarios or encounters, and completing said challenges is very satisfying/rewarding. Games like FromSoftware's Dark Souls 3 and Elden Ring are examples which should get scores 9 or 10.

DO NOT focus on anything other than challenge when scoring. It doesn't matter if some players find the difficulty frustrating, as long as there are some players who find it deeply satisfying and rewarding, the game should still get a score of 9 or 10.

{format_instructions}

Output a properly formatted json as described above. Don't forget to include commas after each property.
"""
}

BLURB_PROMPT = """
You are an expert gaming blurb writer. Given a review with game scores, summarize the review into a short blurb (~100 words). Do not mention the game's name. Summarize the strengths and weaknesses of the game. Highlight any standout aspects (score >7).

Here is the review:
```
{review_text}
```

Mention the sub-scores wherever possible. DO NOT mention the overall JUICE score.

Output only the blurb text, nothing else.
"""