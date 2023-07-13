import streamlit as st
import torch
import pandas as pd
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast

model_path = "D:/VoC 2/models"

# Load the saved model
model = DistilBertForQuestionAnswering.from_pretrained(model_path)

# Load the saved tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# Load the dataset
dataset = pd.read_csv("D:\VoC 2\context_data.csv")  

# Get the context based on the question
def get_context(question):
    matching_title = dataset[dataset["title"].apply(lambda x: x.lower() in question.lower())]
    if not matching_title.empty:
        context = matching_title.iloc[0]["context"]
        return context
    return None

# Question answering
def answer_question(question, context):
    # Tokenize the question and context
    inputs = tokenizer(question, context, truncation=True, padding=True, return_tensors='pt')

    # Perform the question answering inference
    outputs = model(**inputs)

    # Extract the start and end indices of the answer
    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits) + 1  

    # Decode the tokens and extract the answer text
    answer = tokenizer.decode(inputs['input_ids'][0][start_index:end_index])

    return answer

# Streamlit app
def main():
    st.title("HAPPY RANK")
    st.write("Please Enter Your Question")

    
    question = st.text_input("Question")

    
    if st.button("Answer"):
        if question:
            # Get the context based on the question
            context = get_context(question)

            if context:
                
                answer = answer_question(question, context)

                
                st.write("Answer:", answer)
            else:
                st.write("No matching context found for the question.")
        else:
            st.write("Please enter a question.")

if __name__ == '__main__':
    main()
