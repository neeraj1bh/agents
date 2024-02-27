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
from dotenv import load_dotenv

from tools.sql import run_query_tool, list_tables, describe_tables_tool

load_dotenv()

chat = langchain_openai.ChatOpenAI()

tables = list_tables()
prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(
            content=f"You are an AI that has access to a Sqlite database. \n{tables}"
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

tools = [run_query_tool, describe_tables_tool]

# agent_executor = initialize_agent(
#     llm=chat, tools=tools, agent=AgentType.OPENAI_FUNCTIONS, verbose=True
# )

agent = OpenAIFunctionsAgent(llm=chat, prompt=prompt, tools=tools)

# agent = create_openai_functions_agent(llm=chat, prompt=prompt, tools=tools)

agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools)


agent_executor.invoke("How many users have provided a shipping address?")
