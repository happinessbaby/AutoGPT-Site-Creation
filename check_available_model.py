import openai

openai.api_key = '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'

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
			
