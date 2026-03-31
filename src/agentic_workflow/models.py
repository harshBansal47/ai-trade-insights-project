from langchain_google_genai import ChatGoogleGenerativeAI



default_model = ChatGoogleGenerativeAI(model ="gemini-3.1-pro-preview", temperature=0.7, max_tokens=2048)


