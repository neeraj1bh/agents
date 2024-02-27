import langchain_openai
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.agents import (
    OpenAIFunctionsAgent,
    AgentExecutor,
    initialize_agent,
    AgentType,
)
from dotenv import load_dotenv

from tools.sql import run_query_tool

load_dotenv()

chat = langchain_openai.ChatOpenAI()


prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

tools = [run_query_tool]

# agent_executor = initialize_agent(
#     llm=chat, tools=tools, agent=AgentType.OPENAI_FUNCTIONS, verbose=True
# )

agent = OpenAIFunctionsAgent(llm=chat, prompt=prompt, tools=tools)

agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools)


agent_executor("How many users are there in the database?")
