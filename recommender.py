import re
import spacy
import wikipedia
import warnings
import requests
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# AI Tools database
ai_tools = [
    {"name": "Akkio", "description": "No-code AI platform for predictive analytics and sales forecasting."},
    {"name": "Tableau", "description": "Business intelligence tool for interactive data visualization and reporting."},
    {"name": "MonkeyLearn", "description": "No-code text analysis tool for sentiment analysis and classification."},
    {"name": "H2O.ai", "description": "AI/ML platform for scalable enterprise machine learning."},
    {"name": "Google Analytics", "description": "Tracks website traffic and provides smart insights using machine learning."},
    {"name": "Julius AI", "description": "Upload data files and ask questions in plain English—no coding needed."},
    {"name": "Domo", "description": "Cloud BI platform with over 1,000 connectors and data visualization features."},
    {"name": "Zoho Analytics", "description": "Self-service BI tool with dashboards, ML integration, and data syncing."},
    {"name": "ThoughtSpot", "description": "Search-based analytics platform using AI for intuitive data exploration."},
    {"name": "Qlik", "description": "Explores data freely with AI-driven insights and associative data model."},
    {"name": "SAP BusinessObjects", "description": "Provides reporting, analysis, and data visualization capabilities."},
    {"name": "Dundas BI", "description": "Browser-based BI platform for interactive dashboards and performance tracking."},
    {"name": "Google Looker", "description": "Real-time business insights and semantic modeling with AI integration."},
    {"name": "Microsoft Power BI", "description": "Business analytics with AI integration and easy-to-use visual tools."},
    {"name": "SAS Viya", "description": "Advanced analytics platform for ML and large-scale data processing."},
    {"name": "Databricks", "description": "Unified data analytics on Apache Spark with MLflow and autoscaling."},
    {"name": "IBM Watson Analytics", "description": "Uses NLP for easy data exploration and predictive analytics."},
    {"name": "Salesforce Einstein", "description": "AI for predictive analytics and intelligent CRM automation."},
    {"name": "HubSpot AI", "description": "Enhances marketing and sales with AI content and lead scoring."},
    {"name": "Albert.ai", "description": "Automates and optimizes digital advertising campaigns with AI."},
    {"name": "Algolia", "description": "Delivers personalized AI-powered search and recommendations."},
    {"name": "Brand24", "description": "Media monitoring tool to track online sentiment and trends."},
    {"name": "Influencity", "description": "Manages influencer marketing campaigns with AI analytics."},
    {"name": "Improvado", "description": "Automates data collection for marketing analytics dashboards."},
    {"name": "Surfer SEO", "description": "Optimizes content based on top-ranking SEO data insights."},
    {"name": "Jasper AI", "description": "Generates high-quality marketing content using AI."},
    {"name": "ContentShake AI", "description": "Semrush tool for SEO-friendly content generation with LLMs."},
    {"name": "Reply.io AI Assistant", "description": "Crafts personalized email replies using sales AI."},
    {"name": "Optimove", "description": "CRM and marketing automation with predictive analytics."},
    {"name": "Bluecore", "description": "Retail marketing platform with predictive AI campaigns."},
    {"name": "Arya.ai", "description": "Provides AI APIs for fraud detection, finance, and risk."},
    {"name": "Zest AI", "description": "AI platform for fair, fast, and predictive credit scoring."},
    {"name": "AlphaSense", "description": "Search engine for finance pros to explore market intel."},
    {"name": "Spindle AI", "description": "Predicts financial trends using machine learning models."},
    {"name": "Quantivate", "description": "Risk and compliance platform powered by AI insights."},
    {"name": "Zapliance", "description": "Automates payment recovery and cash flow operations."},
    {"name": "Tipalti", "description": "Streamlines invoicing and reconciliation with AI."},
    {"name": "BlackLine", "description": "Automates financial operations and account reconciliations."},
    {"name": "Anaplan", "description": "AI-driven platform for financial and supply chain planning."},
    {"name": "UiPath", "description": "Top RPA tool to automate workflows with AI-powered bots."},
    {"name": "Automation Anywhere", "description": "Comprehensive platform combining AI with RPA."},
    {"name": "Blue Prism", "description": "Enterprise-grade RPA for secure, scalable automation."},
    {"name": "WorkFusion", "description": "Combines AI and RPA for automating high-volume tasks."},
    {"name": "Kofax", "description": "AI-powered document and workflow automation platform."},
    {"name": "Pega Systems", "description": "AI-driven customer engagement and workflow platform."},
    {"name": "Appian", "description": "Low-code app builder with integrated AI workflows."},
    {"name": "NICE Systems", "description": "Optimizes contact centers using AI automation."},
    {"name": "Microsoft Power Automate", "description": "AI automation flows for cloud-based business tools."},
    {"name": "Zapier", "description": "Connects apps and automates tasks without coding."},
    {"name": "Celonis", "description": "Optimizes process flows using process mining AI."},
    {"name": "Hyperscience", "description": "Transforms unstructured content into structured data using AI."},
    {"name": "ChatGPT", "description": "Generative AI for Q&A, writing, coding, and content tasks."},
    {"name": "Google Gemini", "description": "Multimodal AI integrated into Google tools and workspace."},
    {"name": "Notion AI", "description": "Smart assistant built into the Notion productivity suite."},
    {"name": "Grammarly", "description": "Writing assistant for grammar, tone, and clarity improvements."}
]

# Definitions for fallback
fallback_definitions = {
    "business": "Business refers to the organized efforts and activities of individuals to produce and sell goods and services for profit.",
    "marketing": "Marketing is the process of promoting, selling, and distributing a product or service.",
    "finance": "Finance is the management of large sums of money, especially by governments or large companies.",
    "sales": "Sales is the activity of selling products or services in return for money or other compensation.",
    "analytics": "Analytics is the computational analysis of data or statistics to discover meaningful insights.",
    "machine learning": "Machine Learning is a branch of AI that enables computers to learn from data without being explicitly programmed.",
    "deep learning": "Deep Learning is a subset of machine learning that uses neural networks with many layers to model complex patterns in data.",
    "crm": "Customer Relationship Management (CRM) refers to strategies and tools used by businesses to manage customer interactions.",
    "kpi": "Key Performance Indicator (KPI) is a measurable value that shows how effectively a company is achieving business objectives.",
    "stock market": "A stock market is a place where shares of publicly held companies are issued, bought and sold.",
    "artificial intelligence": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn.",
}

# Prepare tool vectorizer
corpus = [tool["description"] for tool in ai_tools]
vectorizer = TfidfVectorizer()
corpus_vectors = vectorizer.fit_transform(corpus)

def correct_spelling(text):
    return str(TextBlob(text).correct())

def classify_input(user_input):
    doc = nlp(user_input)
    if doc[0].pos_ == "AUX" or doc[0].tag_ in ["WP", "WRB"]:
        return "doubt"
    return "usecase"

def recommend_tools(user_input, top_k=3):
    query_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(query_vec, corpus_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    recommendations = []
    for idx in top_indices:
        score = round(similarities[idx], 3)
        if score > 0:
            tool = ai_tools[idx]
            recommendations.append({
                "name": tool["name"],
                "description": tool["description"],
                "score": score
            })
    return recommendations

def get_fallback_definition(query):
    for key in fallback_definitions:
        if key.lower() in query.lower():
            return fallback_definitions[key]
    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{query.lower()}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return f"Definition from dictionary: {data[0]['meanings'][0]['definitions'][0]['definition']}"
    except:
        pass
    return None

def get_daily_news():
    try:
        url = "https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey=e724579b75054feb91ff0f4b4f2faacb"
        response = requests.get(url)
        if response.status_code == 200:
            news = response.json()
            headlines = [article['title'] for article in news['articles'][:5]]
            return "\n".join(f"- {headline}" for headline in headlines)
    except:
        return "Failed to fetch news."
    return "No news available."

def answer_business_doubt_local(question):
    try:
        if "news" in question.lower() and ("today" in question.lower() or "latest" in question.lower()):
            return get_daily_news()

        corrected = correct_spelling(question)
        cleaned = re.sub(r"(?i)\b(what\s+is|what\s+about|tell\s+me\s+about|define|describe|explain|who\s+is|about)\b\s*", "", corrected).strip()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return wikipedia.summary(cleaned, sentences=2)

    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
        fallback = get_fallback_definition(cleaned)
        return fallback if fallback else "Sorry, I couldn't find a detailed answer for your query."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

if __name__ == "__main__":
    user_input = input("Enter your business query or use case (or type 'news' for headlines): ")
    if user_input.strip().lower() == "news":
        print("\nToday's Business News Headlines:\n")
        print(get_daily_news())
    else:
        input_type = classify_input(user_input)

        if input_type == "usecase":
            print("\nRecommended AI Tools for Your Use Case:\n")
            results = recommend_tools(user_input)
            if results:
                for res in results:
                    print(f"- {res['name']} (Score: {res['score']}): {res['description']}")
            else:
                print("Sorry, no relevant tools matched your query.")
        else:
            print("\nBusiness Analytics Insight:\n")
            print(answer_business_doubt_local(user_input))
