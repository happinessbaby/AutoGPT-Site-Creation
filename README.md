# Creating a functional site using AutoGPT
### setting up AutoGPT
- If you followed the [setup page][https://docs.agpt.co/setup/] and ran into problems, I got a few troubleshooting tips
  - When you download their latest release [https://github.com/Significant-Gravitas/Auto-GPT], note the python version required. Currently it's 3.10 or later. If your virtual environment has a python version that does not meet the requirement, you will run into multiple problems. So make sure to specify the right python version when you create your environment!
  - If you enabled Redis as the backend, make sure you have Redisearch too! You may need to start Redisearch (docker run -p 6379:6379 redislabs/redisearch:latest) before running AutoGPT. If you run into Redis related errors, I suggest you kill all the containers running on port 6379 and restart Redisearch. If you're asked to set up an authentication password for the server, follow this [how to link][https://stackink.com/how-to-set-password-for-redis-server/]
  - The very first time before activating your virtual environment, I suggest you run "pip install -r requirements.txt" first. Although AutoGPT should automatically run requirements.txt the very first time when it sees you don't have required packages installed, sometimes it doesn't, which means you'll have to manually pip install every package (not suggested!)

### knowing AutoGPT's limitations
- It is an AI assistance that needs human assistance. This could be the start of human-AI collaboration, which could be where future is going with AI. Imagine we each have a personal AI assistant that helps us run our lives. Before it can assist us, however, we need to help it understand our needs and wishes. It will get confused if you're confused on what you want to do. 


### what I want it to do for me
#### create a functional webpage that uses OpenAI's LLM models for content creation
#### give it access to this repository and have it push any code on demand 
#### extra: maintain the site and monitor the budget
