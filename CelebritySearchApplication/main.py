import os
from constant import openai_key
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

st.title("Celebrity Search Application")
input_text = st.text_input("Search the topic you want")

# Initializing OPENAI Model
llm = OpenAI(temperature=0.1)

# Prompts templates
First_Prompt = PromptTemplate(
    input_variables=['name'],
    template='Tell me about {name}'
)

Second_Prompt = PromptTemplate(
    input_variables=['Person'],
    template='When was {Person} born?'
)

Third_Prompt = PromptTemplate(
    input_variables=['DOB'],
    template='Tell me about 5 major events that happened around {DOB}'
)

# Memory
Person_Memory = ConversationBufferMemory(input_key='name',memory_key='chat_history')
DOB_Memory = ConversationBufferMemory(input_key='Person',memory_key='chat_history')
Events_Memory = ConversationBufferMemory(input_key='DOB',memory_key='events_history')


chain = LLMChain(
    llm=llm,
    prompt = First_Prompt,
    verbose=True,
    output_key='Person',
    memory=Person_Memory
)
chain2 = LLMChain(
    llm=llm,
    prompt = Second_Prompt,
    verbose=True,
    output_key='DOB',
    memory=DOB_Memory
)
chain3 = LLMChain(
    llm=llm,
    prompt=Third_Prompt,
    verbose=True,
    output_key='MajorEvents',
    memory=Events_Memory
)

chaining = SimpleSequentialChain(
    chains = [chain,chain2,chain3],
    input_variables=['name'],
    output_variables=['Person','DOB','MajorEvents'],
    verbose = True
)

if input_text:
    st.write(chaining({'name':input_text}))
    
    with st.expander('Person Memory'):
        st.info(Person_Memory.buffer)
        
    with st.expander('Events Memory'):
        st.info(Events_Memory.buffer)
    