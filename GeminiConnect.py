# # API Key AIzaSyA1fXMSm7KNtXRr5IRyGaLYxteG2AzpDrM

# import pathlib
# import textwrap
# from IPython.display import display
# from IPython.display import Markdown
# from google.colab import userdata
# def to_markdown(text):
#   text = text.replace('•', '  *')
#   return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
# for m in genai.list_models():
#   if 'generateContent' in m.supported_generation_methods:
#     print(m.name)

import google.generativeai as genai
API_KEY = ""

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("Capital of New York?")
print(response._result.candidates[0].content.parts[0].text)