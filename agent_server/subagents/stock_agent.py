'''
Stock Agent Submodule
'''
import yfinance as yf
from langchain_core.tools import tool

################# Tool #################
@tool
def get_stock_recommendations(stock: str):
    '''
    Get analyst recommendation for the given ticker symbol and print it.
    '''
    ticker = yf.Ticker(stock)    
    return ticker.get_recommendations()

############### Prompt #################
stock_agent_prompt = (
    "You are a stock market analyst. "
    "Use the tool to help answer user queries."
)

################ Subagent Definition #################
stock_subagent = {
    "name": "stock-agent",
    "description": "Used to perform analytic tasks on selected stock",
    "system_prompt": stock_agent_prompt,
    "tools": [get_stock_recommendations],
    #"model": "openai:gpt-5.4",  # Optional override, defaults to main agent model
}