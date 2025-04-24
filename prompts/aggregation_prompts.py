JUICE_AGGREGATION_PROMPTS = {
    "lore_worldbuilding_atmosphere": """
You are expert at video game reviewing and scoring. Given a set of summaries focussing on atmosphere, lore, and worldbuilding, you need to provide a score, out of 10 and a detailed explanation for it.

A score of 10 means that the summaries indicate intricate and cohesive worldbuilding, intriguing and unique atmosphere, as well as deep and though-provoking lore. A score of 0 implies that the game does not put any effort into these aspects at all.

Here are the summaries:
```
{summary_texts}
```

{format_instructions}
""",
    "exploration": """
You are expert at video game reviewing and scoring. Given a set of summaries focussing on exploration, you need to provide a score, out of 10 and a detailed explanation for it.

A score of 10 means that the summaries indicate a major focus on well-thought out exploration and a rich world with many secrets as well as mentions of players particurly enjoying exploration. A score of 0 implies that the game is entirely linear and there is no mention of exploration at all.

Here are the summaries:
```
{summary_texts}
```

{format_instructions}
""",
    "gameplay_mechanics": """
You are an expert at video game reviewing and scoring. Given a set of summaries focussing on gameplay mechanics, you need to provide a score, out of 10 and a detailed explanation for it.

A score of 10 means that the summaries indicate deeply engaging gameplay mechanics that players found enjoyable for a long time. Note that it doesn't matter if the mechanics are hard-to-learn or execute as long as they are well-though-out and deep. A score of 0 implies that the game has extremely bad gameplay mechanics that almost all players complained about.

Here are the summaries:
```
{summary_texts}
```

{format_instructions}
""",
    "artstyle": """
You are an expert at video game reviewing and scoring. Given a set of summaries focussing on art style, you need to provide a score, out of 10 and a detailed explanation for it.
A score of 10 means that the game features a unique and cohesive artstyle with a lot of though behind it. A score of 0 implies that the game's artstyle is completely unappealing and unoriginal.

Here are the summaries:
```
{summary_texts}
```

{format_instructions}
""",
    "emotional_engagement": """
You are an expert at video game reviewing and scoring. Given a set of summaries focussing on emotional engagement, you need to provide a score, out of 10 and a detailed explanation for it.
A score of 10 means that the game has a realistic, mature way of dealing with emotional themes, features strongly written and complex characters and tends to elicit strong emotional responses and attachment from players. A score of 0 implies that either the game does not feature any emotional depth at all or that it relies too much on cliches, stereotypes and trivial emotions, without actually putting in the work to make the player feel something.

Here are the summaries:
```
{summary_texts}
```

{format_instructions}
""",
    "bloat_repetitiveness_tediousness": """
You are an expert at video game reviewing and scoring. Given a set of summaries focussing on bloat/repetitiveness/tediousness, you need to provide a score, out of 10 and a detailed explanation for it.
- A score of 10 means that game offers an incredibly focussed experience pretty much never wastes the player's time with repetitive or tedious objectives and encounters. Note that repetitiveness/monotonicity due to challenging gameplay should be ignored and must not be penalized. A game in which repetitiveness only occurs due to challenging gameplay or voluntary repeated playthroughs should get a score of 9 or more.
- A score of 0 implies that the game is so rife with bloat, repetitive encounters and tedious mechanics that almost all players complain about it and don't enjoy the game because of it. Grindy games must be penalized heavily (score ~5). Repetitive side missions must also be penalized heavily (Score ~5).

DO NOT MENTION OR SCORE BASED ON THE OVERALL ASSESSMENT OF THE GAME. FOCUS ON THE BLOAT/REPETITIVENESS ONLY.

Here are the summaries:
```
{summary_texts}
```

{format_instructions}
""",
}

BLURB_PROMPT = """
You are an expert blurb writer. Give a review with game scores, summarize the review into a short blurb (~50 words). No need to mention any of the scores.

Here is the review:
```
{review_text}
```

Output only the blurb text, nothing else.
"""