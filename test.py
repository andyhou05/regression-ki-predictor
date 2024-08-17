#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
from openai import OpenAI
client = OpenAI(api_key="sk-8SQs3YxEzTHjJ5VnepAyBpj_E2oewAC0FIT_K1BD0hT3BlbkFJ-dh20izYxxGXVsZ9yq6TOLmb_rPZJDDAaoRZv2dEIA")

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What are IC50 and Ki values"
        }
    ]
)

print(completion.choices[0].message)