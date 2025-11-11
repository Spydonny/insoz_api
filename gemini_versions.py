import google.generativeai as genai

genai.configure(api_key="AIzaSyBsehDFlZ7zqncGlKzOdd1iehIqNFgzl-A")

for m in genai.list_models():
    print(m.name)