# atari-irl-v2

Work in progress. This is a refactor of https://github.com/HumanCompatibleAI/atari-irl with two major goals:
1. Better seperate out the training code from the policy code, largely using the buffer abstraction
2. Handle the typical workflows with automatic caching, using the cache abstraction

script.py handles the full workflow
