import openai


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

models = openai.Model.list()


def get_text_model(model):
    for model in models['data']:
        if model['id'].startswith('text-'):
			try:
				openai.Completion.create(
					engine=model,
					prompt='This is a test prompt.',
					max_tokens=5
				)
				return model
			except:
				pass
	return ""
			
