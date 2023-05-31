# Code for OpenAI API call

import openai

openai.api_key = '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'

model_engine = 'text-davinci-002'

def call_openai_api(prompt):
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text
