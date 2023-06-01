import openai
import os


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

models = openai.Model.list()


def get_text_model():
	for model in models['data']:
		if model['id'].startswith('text-'):
			try:
				openai.Completion.create(
					model=model['id'],
					prompt='This is a test prompt.',
					max_tokens=5
				)
				print(f"available text model: {model}")
				return model
			except:
				pass
	return ""

get_text_model()
			
