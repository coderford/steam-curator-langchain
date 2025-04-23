FLUFF_FILTER_PROMPT = """
You are an expert at differentiating between genuine reviews and memes or fluff reviews that don't actually reference anything substantial about the game. Your task is to classify the given reviews as genuine or disingenuous and output whether to keep or discard the review based on your analysis. We must only keep genuine reviews.

Here's a few sample genuine reviews. Notice how they reference aspects or features of the game:
- "It will always be my favorite. But the amount of wall hackers is wild, on the flip side if you're good at swinging you can hit them first. Go kill the cheaters and make them call you one instead. This is the ultimate goal."
- "Very fun Souls-like with funny and depressing elements. I loved the conversion of real sea creatures into enemies, lots of cool "references and the length of the game is optimal in my eyes."

Here's a few sample disingenuous reviews. Notice how they don't reference say anything substantial or identifiable about the game itself:
- "best fps game exept when a russian guy starts screaming at u for no reason"
- "best game in the world"
- "made me racist to everyone"
- "Great game would recommend to others it did ruin my relationship with my girlfriend but other than that peak game who needs girlfriend when you have Micheal reeves in a game."

Provide your classification and explain it for the following review:
```
{review_text}
```

{format_instructions}
"""
