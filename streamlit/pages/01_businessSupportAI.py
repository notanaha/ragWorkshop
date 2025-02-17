import os, json
from IPython.display import Image
import openai
import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from tenacity import retry, stop_after_attempt, wait_random_exponential
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizedQuery, 
)

#--------------------------------------#
# Set OpenAI variables                 #
#--------------------------------------#
aoai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
aoai_api_key = os.environ["AZURE_OPENAI_API_KEY"]
api_version = os.environ["AZURE_OPENAI_API_VERSION"]

aoai_client = openai.AzureOpenAI( 
    azure_endpoint=aoai_endpoint,
    api_key=aoai_api_key,
    api_version= api_version
)
    
#chat_model: str = "gpt-4-turbo-jp"
chat_model: str = os.environ["AZURE_OPENAI_CHAT_MODEL"]
embedding_model: str = os.environ["AZURE_OPENAI_EMBEDDING_MODEL"]

service_endpoint = os.environ["SEARCH_ENDPOINT"] 
key = os.environ["SEARCH_KEY"]
search_index_name = os.environ["SEARCH_INDEX_NAME"]
credential = AzureKeyCredential(key)

search_client = SearchClient(endpoint=service_endpoint, index_name=search_index_name, credential=credential)

#/*----------------*/
#  gpt4_query      */
#/*----------------*/
def gpt4_query(messages, tools=None, tool_choice=None, model=chat_model):

    response = aoai_client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0,
    max_tokens = 4000,
    #response_format={ "type": "json_object" },
    tools=tools,
    tool_choice=tool_choice
    )

    response_message = response.choices[0].message
    messages.append(response_message)

    return response_message, messages


def call_gpt4(messages, tools=None, tool_choice=None, user_context=None):

    response_message, messages = gpt4_query(messages, tools=tools, tool_choice=tool_choice)

    answer     = response_message.content
    tool_calls = response_message.tool_calls
    #print("answer:", answer)
    print("tool_calls:", tool_calls)

    if tool_calls:
        checkandrun_function_calling(tool_calls, messages)
        response_message, messages  = gpt4_query(messages)
    else:
        print("No tool calls were made by the model.")

    return response_message, messages


def checkandrun_function_calling(tool_calls, messages, ft_content=''):
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        print("function_name: ", function_name)
        function_args = json.loads(tool_call.function.arguments)
        print("function_args: ", function_args)

        if function_name == "searchDocuments":
            function_response = searchDocuments(
                query=function_args.get("query")
            )
        else:
            function_response = json.dumps({"error": "Unknown function"})

        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            }
        )  
        ft_content += function_response
            
    return ft_content


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def generate_embeddings(text, model, aoai_client):
    text = text.replace("\n", " ")
    return aoai_client.embeddings.create(input = [text], model=model).data[0].embedding


def searchDocuments(query, aoai_client=aoai_client, embedding_model=embedding_model, search_client=search_client):

    vector_query = VectorizedQuery(vector=generate_embeddings(query, embedding_model, aoai_client), k_nearest_neighbors=3, fields="contentVector")

    results = list(search_client.search(  
        search_text=query,  
        vector_queries=[vector_query],
        select=["category", "title", "content"],
        query_type=QueryType.SEMANTIC, 
        semantic_configuration_name="default",
        #query_caption=QueryCaptionType.EXTRACTIVE, 
        #query_answer=QueryAnswerType.EXTRACTIVE,
        top=3
    ))

    concatenated_documents = ""
    for doc in results:
        print("Document title:", {doc['title']})
        concatenated_documents += f"<DOCUMENT>\nCategory Name: {doc['category']}\nTitle: {doc['title']}\nContent: {doc['content']}\n</DOCUMENT>\n"
    
    #concatenated_documents += '\n' + user_context

    return concatenated_documents



tools = [
    {
        "type": "function",
        "function": {
            "name": "searchDocuments",
            "description": "Use this to search for documents relevant to the query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "query string to search for documents. Use the query as it is.",
                    },
                },
                "required": ["query"],
            },
        }
    }
]


def main():

    query = st.text_input("Enter your question:", value="パーキングブレーキについて教えてください。")

    if st.button("Enter", key="confirm"):
        if query.lower() == "reset":
            for key in st.session_state.keys():
                del st.session_state[key]
            st.write("Conversation history has been reset.")
            return

        if 'messages' not in st.session_state:
            #--------------------------------------#
            # Retrieving answer from GPT-4         #
            #--------------------------------------#
            with open(os.path.join('utilities','01_businessSupport_sys_msg01.txt'), "r", encoding = 'utf-8') as f:
                system_message = f.read()

            messages = []
            messages.append({"role": "system","content": system_message})
        else:
            messages = st.session_state['messages']


        messages.append({"role": "user", "content": query})

        st.write("Retrieving answer from AI Assistant...")
        print("etrieving answer from AI Assistant...")

        response_message, messages = call_gpt4(messages, tools=tools, tool_choice='auto', user_context=query)
        st.write("Response message: ", response_message.content)
        print("Response message: ", response_message.content)

        st.session_state['messages'] = messages


if __name__ == '__main__':
    main()

