# Creating a functional site using AutoGPT
### setting up AutoGPT
- If you followed the [setup page][https://docs.agpt.co/setup/] and ran into problems, I got a few troubleshooting tips
  - When you download their latest release [https://github.com/Significant-Gravitas/Auto-GPT], note the python version required. Currently it's 3.10 or later. If your virtual environment has a python version that does not meet the requirement, you will run into multiple problems. So make sure to specify the right python version when you create your environment!
  - If you enabled Redis as the backend, make sure you have Redisearch too! You may need to start Redisearch (docker run -p 6379:6379 redislabs/redisearch:latest) before running AutoGPT
  - The very first time before activating your virtual environment, I suggest you run "pip install -r requirements.txt" first. Although AutoGPT should automatically run requirements.txt the very first time when it sees you don't have required packages installed, sometimes it doesn't, which means you'll have to manually pip install every package (not suggested!)
