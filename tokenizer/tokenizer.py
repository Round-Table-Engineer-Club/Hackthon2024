import os
import openai
from dotenv import load_dotenv
import time

load_dotenv()


def open_ai_formatting(text):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    retries = 3
    while retries > 0:
        try:
            response = openai.ChatCompletion.create(
                messages=[
                    {
                        "role": "system",
                        "content": f"For the following setence/word try to separate words from a long string, or check grammar mistake and reformat the sentence. For example, the input is 'gobuckeyes', the output should be 'go buckeyes'. \n",
                    },
                    {
                        "role": "user",
                        "content": f"{text}",
                    },
                ],
                model="gpt-3.5-turbo",
                temperature=0,
                request_timeout=15,
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            if e:
                retries -= 1
                print("Retrying...")
                time.sleep(5)
            else:
                raise e
