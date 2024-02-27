import langchain_openai
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.agents import (
    OpenAIFunctionsAgent,
    AgentExecutor,
    initialize_agent,
    AgentType,
    create_openai_functions_agent,
)
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

from tools.sql import run_query_tool, list_tables, describe_tables_tool
from tools.report import write_report_tool
from handlers.chat_model_start_handler import ChatModelStartHandler

load_dotenv()

handler = ChatModelStartHandler()
chat = langchain_openai.ChatOpenAI(callbacks=[handler])

tables = list_tables()
prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(
            content=(
                "You are an AI that has access to a Sqlite database. \n"
                f"The database has table of: {tables}"
                "Do not make any assumptions of what tables or columns exist."
                "Instead use the 'describe_tables' function."
            )
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
tools = [run_query_tool, describe_tables_tool, write_report_tool]

# agent_executor = initialize_agent(
#     llm=chat, tools=tools, agent=AgentType.OPENAI_FUNCTIONS, verbose=True
# )

agent = OpenAIFunctionsAgent(llm=chat, prompt=prompt, tools=tools)

# agent = create_openai_functions_agent(llm=chat, prompt=prompt, tools=tools)

agent_executor = AgentExecutor(
    agent=agent,
    # verbose=True,
    tools=tools,
    memory=memory,
)


# agent_executor.invoke("How many users have provided a shipping address?")

# agent_executor.invoke(
#     "Summarize the top 5 products. Write the results to a report file"
# )

agent_executor.invoke("How many orders are there? Write the results to an html report.")

agent_executor.invoke("Repeat the exact same process for users.")
