import re
import os
import openai
import pandas as pd
from openai import OpenAI

from dotenv import load_dotenv

# Load the openai-api-key
load_dotenv()

# set the api_key
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def classify_captions_to_dataframe(input_txt_path: str) -> pd.DataFrame:
    # Read the input data from the text file
    with open(input_txt_path, 'r') as file:
        input_data = file.readlines()

    # Prepare a list to store the classified rows
    classified_rows = []

    # Define the prompt for ChatGPT
    prompt = f"""Classify the following text into three categories: sleeves type, neck type, and item.
            Provide the result in a string format separated by ',' as "sleeves_type, neck_type, item".
            If a category is not present, mark it as "N/A".

            Example:
            Text: "Short Sleeves Round Neck T-Shirt"
            Result: "Short Sleeves, Round Neck, T-Shirt"
            """

    # Get the response from OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # Process each line in the input data
    for line in input_data:
        # Split the filename and description
        filename, description = line.strip().split(' ', 1)

        # Extraer la respuesta del modelo
        result = response['choices'][0]['message']['content'].strip()

        # Split the result into the three categories
        sleeves_type, neck_type, item = result.split(', ')

        # Append the classified data as a row
        classified_rows.append([filename, sleeves_type, neck_type, item])

    # Create a DataFrame with the classified data
    df = pd.DataFrame(classified_rows, columns=["filename", "sleeves_type", "neck_type", "item"])

    return df


# Example usage (adjust input path)
input_txt_path = 'dc_caption_upper.txt'

# Get the classified DataFrame
df_classified = classify_captions_to_dataframe(input_txt_path)

# Display the DataFrame (if in a Jupyter environment, it will show automatically)
print(df_classified)

# data = []
# with open(filename, 'r') as file:
#     for line in file:
#         parts = line.strip().split(' ', 1)
#         if len(parts) == 2:
#             data.append(parts)
#         else:
#             print('Something went wrong')
#
# df = pd.DataFrame(data)
# df.columns = ['file_name', 'category']
# df.category = df.category.str.title()



# df['neck_type'] = df['category'].apply(lambda x: re.search(r'(?<!Sleeve\s)([\w-]+\s*(Neck|Collar))', x).group(1) if re.search(r'(?<!Sleeve\s)([\w-]+\s*(Neck|Collar))', x) else 'Unknown')
#
#
# print(df.neck_type.value_counts())



# df_counts = df['category'].value_counts(normalize=True, dropna=False).reset_index()
# df_counts['cumulative'] = df_counts['proportion'].cumsum()
#
# df_to_keep = df_counts[df_counts['cumulative'] <= 0.95]
# if df_to_keep.shape[0] > 7:
#     df_to_keep = df_to_keep.head(7)
#     df_to_keep = group_small_segments(df_to_keep, 'category')
#
# print(df_to_keep)

