from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

import os

model = ChatGroq(
    model="Gemma2-9b-It",
    groq_api_key="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"
)

# Prompt template
system_template = "Translate the following into {language}."
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}")
])

parser = StrOutputParser()

chain = prompt_template | model | parser

input_file = "output.txt"
output_file = "translated_output.txt"
language_to_translate = "english"

# Read lines from input file
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

translated_lines = []

for line in lines:
    text = line.strip()
    if text: 
        result = chain.invoke({"text": text, "language": language_to_translate})
        translated_lines.append(result)

with open(output_file, "w", encoding="utf-8") as f:
    for translated_line in translated_lines:
        f.write(translated_line + "\n")

print(f"Translation complete! Translated text saved in {output_file}")
