from transformers import pipeline
classifier = pipeline("sentiment-analysis")
print(classifier(["I hate the world!"]))

generator=pipeline("text-generation", model="distilgpt2")
print(generator("Hello, my name is", max_length=300,num_return_sequences=2))