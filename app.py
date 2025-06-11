import streamlit as st
import os
from openai import AzureOpenAI
import time
import json
from typing import Generator, Dict, Any
from dotenv import load_dotenv
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()
    # Sidebar

def initialize_azure_openai_client() -> AzureOpenAI:
    """Initialize Azure OpenAI client"""
    try:
        # Your existing configuration
        api_key = "f7bd69fb1e124bb79560a0e726fb4631"
        endpoint = "https://hkust.azure-api.net"
        api_version = "2023-05-15"
        
        if not api_key or not endpoint:
            st.error("Please configure Azure OpenAI API key and endpoint")
            st.stop()
            
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
        return client
    except Exception as e:
        st.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
        st.stop()

def web_search(query: str, num_results: int = 5) -> str:
    """Perform a web search using DuckDuckGo"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
            
        if not results:
            return "No relevant search results found"
        
        # Format search results
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"{i}. **{result['title']}**\n"
                f"   Summary: {result['body']}\n"
                f"   Link: {result['href']}\n"
            )
        
        return "\n".join(formatted_results)
    
    except Exception as e:
        return f"Error during search: {str(e)}"

def get_webpage_content(url: str, max_chars: int = 2000) -> str:
    """Retrieve a summary of webpage content"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get plain text
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        text = '\n'.join(line for line in lines if line)
        
        # Truncate to specified length
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        return text
    
    except Exception as e:
        return f"Unable to retrieve webpage content: {str(e)}"

# Define function call tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for information when the user asks for the latest information, real-time data, or needs to find specific materials",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords or question"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of search results to return, default is 5",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_webpage_content",
            "description": "Retrieve detailed content from a specified webpage",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the webpage to retrieve content from"
                    }
                },
                "required": ["url"]
            }
        }
    }
]

def handle_tool_calls(tool_calls, client, messages):
    """Handle tool calls"""
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        if function_name == "web_search":
            function_response = web_search(
                query=function_args.get("query"),
                num_results=function_args.get("num_results", 5)
            )
        elif function_name == "get_webpage_content":
            function_response = get_webpage_content(
                url=function_args.get("url")
            )
        else:
            function_response = f"Unknown function: {function_name}"
        
        # Add function call result to message history
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": function_response
        })
    
    return messages

def get_ai_response_with_tools(client: AzureOpenAI, messages: list, deployment_name: str) -> Generator[str, None, None]:
    """Generate AI response with tool support"""
    try:
        # First call: Check if tools are needed
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=False  # No streaming for tool calls
        )
        
        response_message = response.choices[0].message
        
        # Check for tool calls
        if response_message.tool_calls:
            # Add assistant's response to message history
            messages.append({
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": response_message.tool_calls
            })
            
            # Handle tool calls
            messages = handle_tool_calls(response_message.tool_calls, client, messages)
            
            # Second call: Generate final response based on tool results
            final_response = client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                stream=True
            )
            
            for chunk in final_response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        else:
            # No tool calls, return response directly
            if response_message.content:
                yield response_message.content
                
    except Exception as e:
        #st.write(f"Error occurred: {str(e)}")
        yield ''

def main():
    st.set_page_config(
        page_title="AI Assistant (with Search)",
        page_icon="🤖",
        layout="wide"
    )
    with st.sidebar:
        st.header("Features")
        mode = st.selectbox(
            "Select a feature",
            ["Single event estimator", "Wild estimator"]
        )
        st.markdown("""
        **This AI Assistant offers the following capabilities:**
        - 💬 General conversation
        - 🔍 Real-time web search
        - 📰 Access to the latest news
        - 🌐 Webpage content retrieval
        
        **Search Trigger Conditions:**
        - Requests for the latest news or real-time information
        - Queries for specific data or statistics
        - When information needs verification or supplementation
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    st.title("🤖 AI Assistant (with Web Search)")
    
    # Initialize client
    client = initialize_azure_openai_client()
    
    # Deployment name (modify based on your actual deployment)
    deployment_name = "gpt-35-turbo"  # Or your actual deployment name
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] != "tool":  # Do not display raw tool call results
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Enter your question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            messages_for_api = st.session_state.messages.copy()
            if mode == "Single event estimator":
                # Add system prompt
                system_message = {
                    "role": "system",
                    "content": '''You are an AI assistant with strong information retrieval and logical analysis capabilities. When I raise an event or question (e.g., "Will Trump's proposed tariffs cause Musk to go bankrupt?"), please strictly follow the steps below to complete the task:

    Information Retrieval:
    Use your search functions (including web search and the latest posts on the X platform) to gather all reliable information related to the event.
    Prioritize authoritative sources (e.g., mainstream media, academic articles, official statements) and the most recent data or discussions directly related to the event.
    Summarize the key information found, including the event's background, positions of relevant individuals/organizations, data or evidence, and any important contextual details. Ensure the information is comprehensive and objective.
    Likelihood Analysis:
    Based on the collected information, evaluate the likelihood of the event occurring. Consider the following in your analysis:
    Whether the event's prerequisites are met (e.g., has Trump's tariff policy been implemented or is there a clear plan?).
    Key factors influencing the event (e.g., economic impact, legal constraints, market reactions, behavior of relevant individuals).
    Any external variables that may prevent or facilitate the event (e.g., political environment, public opinion, reactions of other stakeholders).
    Use logical reasoning, combined with data and evidence, to reach a likelihood conclusion. You may refer to the following likelihood levels:
    Extremely Low (0-10%)
    Low (11-30%)
    Moderate (31-60%)
    High (61-90%)
    Extremely High (91-100%)
    If the data is insufficient to draw a clear conclusion, explain the uncertainty and list the key points requiring more information. Regardless, you must provide a likelihood value.
    Response Structure:
    Organize your response clearly and concisely, using the following structure:
    Event Overview: Briefly describe the event and its background.
    Information Summary: List the key information found (including a brief description of sources).
    Likelihood Analysis: Explain in detail the likelihood of the event occurring, including supporting and opposing arguments, and the final likelihood level.
    Conclusion: Summarize the analysis results and indicate whether more information is needed.
    If any assumptions are made, state them clearly.
    Notes:
    Maintain neutrality and objectivity, avoiding subjective bias.
    If the event involves future predictions, clarify that the analysis is based on current information and that future changes may affect the outcome.
    If search results are limited or information is contradictory, note this and make reasonable inferences based on available information.
    Use clear and concise language, avoiding unnecessary complex terms, to ensure accessibility for general users.
    You must provide a likelihood value under all circumstances.
    You must provide a likelihood value under all circumstances.
    You must provide a likelihood value under all circumstances.
    Please respond in English.
    Now, please execute the above steps for the following event:

    Event:'''
            }
            else:
                system_message = {
                    "role": "system",
                    "content": '''你是一个高级分析助手，擅长从网络数据中提取信息并进行逻辑推理。现在，我将提供一个主题，例如“川普的关税大战”。你的任务是：

在网络上搜索与该主题相关的所有最新信息，包括新闻、分析文章、专家评论、X 帖子等，确保信息全面且来源可靠。
基于搜索结果，分析该主题的背景、当前状态、关键影响因素（如经济、政治、社会等），以及可能的未来发展路径。
推导出该主题所有可能的结果。每个结果应清晰描述，并基于数据和逻辑推理，而不是猜测。
为每个可能的结果分配一个发生可能性（以百分比表示，0% 到 100%），并简要说明分配该可能性的依据（如数据支持、趋势分析、专家观点等）。
输出格式为一个清晰的列表，包含以下内容：
每个可能的结果（以简洁的句子描述）。
该结果的发生可能性（百分比）。
可能性估算的简要依据（1-2 句话）。
如果信息不足以推导某些结果或可能性，明确说明，并建议需要哪些额外信息。
确保输出简洁、逻辑严密，避免冗长或无关内容。
输入示例：
主题：川普的关税大战

输出示例：
基于对“川普的关税大战”的网络搜索和分析，以下是所有可能的结果及可能性：

结果：美国经济因高关税显著放缓，通货膨胀加剧
可能性：40%
依据：历史数据显示，2018-2019 年关税导致消费者价格上涨；当前经济分析指出类似政策可能加剧供应链压力。
结果：中国采取报复性关税，导致全球贸易战升级
可能性：30%
依据：X 帖子和经济学家评论显示，中国可能以对等措施回应，但地缘政治谈判可能限制全面贸易战。
结果：关税政策引发国内支持，推动川普政治影响力上升
可能性：20%
依据：部分选民支持保护主义，但民调显示经济担忧可能削弱支持率。
结果：通过谈判达成新贸易协议，关税影响被最小化
可能性：10%
依据：历史先例（如美墨加协定）表明谈判可能，但当前美中关系紧张降低成功概率。
注意： 如果搜索结果不足以确定某些可能性，需说明：“由于缺乏足够数据（如具体关税政策细节），部分结果的可能性估算可能不够精确，建议获取更多政策细节。”
无论如何你都要给出可能性的具体数值,并且你要使用英文回答。
你必须使用英文回答我的问题,必须要使用英文回答!!!
现在，请根据我提供的主题执行上述任务。


我的主题：'''
            }

                
            messages_for_api.insert(0, system_message)
            
            full_response = ""
            for response_chunk in get_ai_response_with_tools(client, messages_for_api, deployment_name):
                full_response += response_chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        # Add assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    


if __name__ == "__main__":
    main()