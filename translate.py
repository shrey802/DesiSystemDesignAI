from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os

def translate():
    try:
        # Initialize the Groq model
        model = ChatGroq(
            model="Gemma2-9b-It",
            groq_api_key=""
        )
        print("Model initialized")

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

        # Iterate through the lines and translate them
        for line in lines:
            text = line.strip()
            if text:
                # Synchronous call to the chain
                result = chain.invoke({"text": text, "language": language_to_translate})
                translated_lines.append(result)

        # Write the translated text to the output file
        with open(output_file, "w", encoding="utf-8") as f:
            for translated_line in translated_lines:
                f.write(translated_line + "\n")

        print(f"Translation complete! Translated text saved in {output_file}")

    except Exception as e:
        print(f"⚠️ Error during translation: {e}")

translate()