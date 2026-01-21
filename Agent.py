from phi.agent import Agent
from phi.model.ollama import Ollama
from phi.tools.googlesearch import GoogleSearch
from datetime import datetime

# -------------------------------
# Search Agent: Fetches tech news
# -------------------------------
search_agent = Agent(
    name="Tech News Search Agent",
    model=Ollama(id="llama3.1"),
    tools=[GoogleSearch()],
    description="Searches the web for the latest AI & Tech news articles.",
    instructions=[
        "Search for the latest AI and Technology news (past 48 hours).",
        "For each news item, extract: Title, Source, Date, Link, and Image if available.",
        "Summarize each news into ~1000 words.",
        "Use engaging bullet points 🔹 to explain:",
        "- What happened",
        "- Why it matters",
        "- Key applications",
        "- Potential impact on industry/workforce",
        "- Example use case if relevant",
        "Add the source link at the end of each news item.",
        "Keep the tone professional, LinkedIn-friendly, and insightful.",
        "End the post with hashtags: #AI #TechNews #Innovation #FutureOfWork #TechLeaders #StartupLife #HR #Recruitment"
    ],
    show_tool_calls=True,
)

# -----------------------------------
# Summarizer Agent: Formats the news
# -----------------------------------
summarizer_agent = Agent(
    name="Tech News Summarizer Agent",
    model=Ollama(id="llama3"),
    description="Summarizes news into a detailed LinkedIn-ready post.",
    instructions=[
        "Take the top 5 news items from the Search Agent.",
        "For each news item, expand the summary to 200–350 words.",
        "Use bullet points 🔹 for each news item.",
        "Include details such as: key developments, potential applications, industry impact, and examples if relevant.",
        "Start with a catchy intro sentence for each news item.",
        "Include source, date, and link at the end of each bullet.",
        "Use professional, engaging language suitable for LinkedIn.",
        "End the post with relevant hashtags: #AI #TechNews #Innovation #HR #Recruitment #FutureOfWork #TechLeaders #StartupLife"
    ],
    show_tool_calls=True,
)

# ---------------------------------
# Main TechNewsBot: Coordinates all
# ---------------------------------
tech_newsbot = Agent(
    team=[search_agent, summarizer_agent],
    model=Ollama(id="llama3.1"),
    instructions=[
        "You are TechNewsBot. Collect the latest top 5 tech news for a given topic and summarize it into an engaging LinkedIn post.",
        "Include intro, emojis, detailed ~1000 word descriptions, and relevant hashtags (#AI #TechNews #Innovation #HR #Recruitment #FutureOfWork #TechLeaders #StartupLife).",
        "Use Search Agent first, then Summarizer Agent.",
        "For each news item, extract: Title, Source, Date, Link, and Image if available.",
        "Summarize each news into ~1000 words.",
        "Use engaging bullet points 🔹 to explain:",
        "- What happened",
        "- Why it matters",
        "- Key applications",
        "- Potential impact on industry/workforce",
        "- Example use case if relevant",
        "Add the source link at the end of each news item.",
        "Keep the tone professional, LinkedIn-friendly, and insightful.",
        "End the post with hashtags: #AI #TechNews #Innovation #FutureOfWork #TechLeaders #StartupLife #HR #Recruitment",
    ],
    show_tool_calls=True,
    debug_mode=True,
    markdown=True,
)

# -----------------------------
# Run TechNewsBot
# -----------------------------
if __name__ == "__main__":
    topic = "AI technology"
    query = f"Collect the latest top 5 {topic} news items and summarize for LinkedIn post with intro, emojis, detailed descriptions, and hashtags."

    # Run the bot
    response = tech_newsbot.run(message=query)

    # Extract only the assistant's content
    if response and hasattr(response, "content"):
        summarized_news = response.content
        print(f"\n🔹 LinkedIn-ready post preview 🔹\n")
        print(summarized_news)
    else:
        print("⚠️ No news generated. Check your agent setup or topic.")
