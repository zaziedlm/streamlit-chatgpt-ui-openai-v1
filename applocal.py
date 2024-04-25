import os
# import openai
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message

load_dotenv()

# Setting page title and header
st.set_page_config(page_title="myGPT", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>myGPT - a experimental chatbot 😬</h1>", unsafe_allow_html=True)

# Set org ID and API key

# old_API----------
# openai.organization = os.environ["OPENAI_ORGANIZATION_ID"]
# openai.api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    # example...Jan Local API Server. 'http://192.168.XX.XX:1337/v1/'
    # For OpenAI_access base_url=None
    # For LocalLLM_access example...Jan Local API Server. 'http://192.168.XX.XX:1337/v1/'
    base_url=os.environ.get('OPENAI_API_URL', None),

)

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Sidebar")
model_name = st.sidebar.radio("Choose a model:", (os.environ['ENGINE_FOR_GPT-3.5'], os.environ['ENGINE_FOR_GPT-4']))

counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Map model names to OpenAI model IDs
if model_name == os.environ['ENGINE_FOR_GPT-3.5']:
    model = os.environ['ENGINE_FOR_GPT-3.5']
else:
    model = os.environ['ENGINE_FOR_GPT-4']

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


# generate a response
def generate_response(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})

    # old_API----------
    # completion = openai.ChatCompletion.create(
    #     model=model,
    #     messages=st.session_state['messages']
    # )
    completion = client.chat.completions.create(
        messages=st.session_state['messages'],
        #model=model,
        # Local model(llama.cpp)
        #model="swallow-7b-instruct.Q5",
        # Local model(Ollama)
        #model="swallow-7b",
        model="llama3",
        
    )

    response = completion.choices[0].message.content
    st.session_state['messages'].append({"role": "assistant", "content": response})

    # print(st.session_state['messages'])
    total_tokens = completion.usage.total_tokens
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    return response, total_tokens, prompt_tokens, completion_tokens


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output, total_tokens, prompt_tokens, completion_tokens = generate_response(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        st.session_state['model_name'].append(model_name)
        st.session_state['total_tokens'].append(total_tokens)

        # from https://openai.com/pricing#language-models
        if model_name == "GPT-3.5":
            cost = total_tokens * 0.002 / 1000
        else:
            cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

        st.session_state['cost'].append(cost)
        st.session_state['total_cost'] += cost

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            st.write(
                f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
            counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
