# LangChain Steam Game Reviewer and QA

An LLM-based steam game reviewer that parses through hundreds of steam reviews to form an opinion. Generates a "Juice Score" based on various aspects like atmosphere, exploration, gameplay mechanics, challenge etc. FromSoftware Games will usually receive high scores.

Supports local models via Ollama and Gemini/OpenAI models as well (you'll need to set `GOOGLE_API_KEY` or `OPENAI_API_KEY` in your env).
Also includes a command-line QA interface powered by Retrieval-Augmented Generation (RAG), allowing you to ask specific questions and receive answers based on the review data.

## Usage

Run `run_chains.py` to get Juice Score and detailed explanations for a given Steam app ID:
```sh
export GOOGLE_API_KEY=<your google gemini API key>
python3 run_chains.py --app_id 1245620 # Get Juice score for Elden Ring
```

Example output for [Elden Ring](https://store.steampowered.com/app/1245620/ELDEN_RING/):
```
JUICE SCORE: 8.1/10

Lore, Worldbuidling and Atmosphere (8/10): The summaries consistently praise the game's atmosphere, worldbuilding, and lore. Many reviewers describe the atmosphere as immersive, intense, and oppressive, contributing to a captivating experience. The world is described as vast, mysterious, beautiful, and brutal, with unique areas and breathtaking visuals. The lore is frequently cited as a high point, with some calling it 'insane' and a 'triumph.' However, a common criticism is that the lore is vague, poorly explained, and requires external resources to fully understand. Some reviewers also suggest that the focus on lore comes at the expense of a compelling narrative with relatable characters. While the positive aspects are strong, the issues with the lore's accessibility and clarity prevent a perfect score. Therefore, an 8 out of 10 reflects the excellent but flawed state of these aspects.

Exploration (8/10): The summaries consistently praise the game's exploration aspects, highlighting a vast world filled with secrets and rewarding curiosity. The density of content and the sense of adventure are frequently mentioned. The ability to control difficulty through exploration is also a plus. However, some reviewers note issues like barren areas, repetitive mini-dungeons, and potentially useless rewards depending on the player's build which prevents the score from being a 9 or 10. The phrase 'exploring like 5 continents' and the mention of 'go anywhere' are strong positives, but the potential for emptiness and repetitive content holds it back slightly.

Gameplay Mechanics (8/10): The gameplay mechanics are highly praised for their depth, complexity, and customization options, drawing heavily from the Soulslike genre while adding new elements. The availability of diverse weapons, armor, spells, and builds allows for varied playstyles and replayability. The combat is described as smooth, fluid, and responsive. While some reviewers mention balance issues, difficult fights, and occasional jank, the overall consensus points to a well-designed and engaging combat system with a high degree of player agency. The ability to tailor the difficulty through character builds and the presence of numerous systems to master contribute to a rewarding gameplay experience. The few criticisms, such as boss design and collision issues, are not significant enough to lower the score substantially, given the overwhelmingly positive feedback on the core mechanics.

Emotional Engagement (7/10): The game consistently evokes strong emotional responses from players, ranging from frustration and despair to euphoria and accomplishment. Several reviewers mention memorable experiences and a lasting impact, with specific boss encounters and the game's world contributing to this emotional engagement. The descriptions of despair, pride, fear, and triumph indicate a relatively broad emotional palette. Some reviewers even describe the experience as transformative or deeply personal. However, there's a lack of consistent and detailed descriptions of mature emotional depth, strongly written or complex characters, or nuanced emotional themes. While the game clearly succeeds in eliciting emotional reactions, it doesn't consistently demonstrate the depth and complexity needed for a higher score. The presence of frustration and anger, while valid emotions, also suggests that some of the emotional response may stem from difficulty rather than narrative or character engagement. The mention of connecting players to a community is a positive sign, indicating a broader impact beyond individual gameplay.

Bloat and Grinding (4/10): The reviews consistently mention bloat, repetitiveness, and grinding as significant issues. Several reviewers point to reused bosses and enemies, repetitive dungeons, and an empty open world padded with filler content. The need to grind levels and weapon upgrades is also frequently cited, along with frustration regarding item progression and build optimization. Some reviews even mention the game becoming boring due to these issues. While some reviews acknowledge the game's expansiveness and unique elements, the prevalence of negative feedback regarding bloat and grindiness necessitates a low score. A score of 4 reflects the significant impact of these issues on the overall player experience, while acknowledging that some players may still find enjoyment in other aspects of the game. The two summaries that mention no grind are outweighed by the five that do.

Challenge (9/10): The summaries consistently describe the game as challenging, often comparing it to the 'Dark Souls' series in difficulty. Reviewers highlight the satisfaction of overcoming tough bosses and mastering the combat mechanics, with examples like Malenia being particularly challenging. While some find the difficulty unreasonable or unfair, the overall sentiment points towards a rewarding experience for players seeking a test of skill and perseverance. Frequent deaths and the need to learn enemy patterns are core elements. The presence of 'cheesing' strategies in later playthroughs and occasional criticism of the 'masochistic' nature doesn't detract significantly from the overall high difficulty, which is often praised as a core aspect of the game's appeal. A score of 10 would imply near-unbreakable difficulty, and while the game is very hard, it sounds like a 9 is more appropriate.


Blurb Version
-------------
JUICE Score: 8.1. Prepare for a brutal and beautiful world filled with secrets! This title excels in its **exploration (8/10)**, offering a vast, rewarding world to uncover. The complex **gameplay mechanics (8/10)**, with its deep customization, smooth combat, and Soulslike inspirations, will keep you engaged. Experience intense emotions, from frustration to triumph, as you face a **challenge (9/10)** that will test your skills. The **lore, worldbuilding, and atmosphere (8/10)** are captivating, though its depth might require external resources to fully grasp. Be warned, however: the game suffers from significant bloat and grinding, leading to repetitive dungeons and an empty world (**Bloat and Grinding: 4/10**).
```

Run `rag_qa.py` to set up an interactive QA interface based on reviews of a given game:
```sh
export GOOGLE_API_KEY=<your google gemini API key>
python3 rag_qa.py -i 1245620
```

