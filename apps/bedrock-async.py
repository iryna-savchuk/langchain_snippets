"""
This code creates multiple concurrent tasks to generate responses
for different input values using an AI model hosted on Bedrock.
The generated responses are printed once they become available.
"""

import asyncio
import boto3
from langchain_community.llms import Bedrock
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

# Load AWS API keys 
from keys import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY


# Function to generate a response for a given input value using an AI model hosted on Bedrock 
async def generate_response(input_value):

    # Create a client interface to interact with the Bedrock service 
    bedrock_client = boto3.client(
        service_name='bedrock-runtime', 
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name='us-west-2'
    )

    prompt = ChatPromptTemplate.from_template("Give definition of {input_value} in one sentence.")
    model = Bedrock(client=bedrock_client, model_id="mistral.mixtral-8x7b-instruct-v0:1")
    parser = StrOutputParser()

    # Create a pipeline for processing input through the prompt, the AI model, and then the output parser 
    chain = prompt | model | parser

    complete_response = ""  # Initialize an empty string to store the complete response
    async for chunk in chain.astream({"input_value": input_value}): # as each chunk becomes available, it is processed within the loop body
        complete_response += chunk # concatenate each chunk to the complete response
    
    print(complete_response) # print the complete response, once all chunks have been received and concatenated


# Function to create multiple concurrent tasks to generate responses for different input values
async def main():

    tasks = []

    # Make multiple concurrent calls
    for input_value in ["regression", "neural network", "API", "confusion matrix", "webhook"]:
        tasks.append(asyncio.create_task(generate_response(input_value)))
    
    # Wait for all tasks to complete
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())