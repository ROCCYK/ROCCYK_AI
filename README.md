# ROCCYK_AI
**ROCCYK_AI** is a custom-built, open-source AI chatbot trained on the personal biography of Rhichard Koh. Designed as a personalized conversational agent, this project showcases how large language models can be used to create highly customized chatbot experiences.

## 🧠 About
This chatbot serves as an AI-powered digital twin that can answer questions, hold conversations, and share insights based on the real-life biography of its creator. The project demonstrates the capabilities of personalizing large language models for niche, individual use cases, such as portfolio enhancement, education, and digital identity applications.

## 🚀 Features
- 🔍 **Biography-trained model**: The chatbot is powered by a system prompt curated from the creator's academic and professional history.
- 💬 **Conversational agent**: Provides natural responses in Q&A format.
- ⚡ **Groq-powered**: Uses Groq's ultra-fast inference API with open-source LLMs (Llama 3.3 70B).
- 🆓 **Fully free to run**: No paid API required — Groq offers a generous free tier.
- 🛠️ **Modular architecture**: Easy to extend or adapt for other personalized use cases.
- 📖 **RAG (Retrieval-Augmented Generation)**: Optionally supports integration with vector stores for context-based responses.

## 🏗️ Tech Stack
- **Python**
- **Groq API** (LLM inference — Llama 3.3 70B)
- **LangChain** (optional, for RAG)
- **FAISS** (for embedding search)
- **Streamlit** (frontend)

## 📁 Repository Structure
```
ROCCYK_AI/
├── ROCCYK_AI.py        # Main Streamlit app
├── requirements.txt    # Python dependencies
├── .env                # API keys (local development)
└── README.md           # Project overview
```

## 🧑‍💻 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/ROCCYK/ROCCYK_AI.git
   cd ROCCYK_AI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get a free Groq API key**
   - Sign up at [console.groq.com](https://console.groq.com)
   - Navigate to **API Keys** → **Create Key**

4. **Prepare your `.env` file** (for local development)
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   BIO=Your biography text here...
   ```

5. **Run the chatbot**
   ```bash
   streamlit run ROCCYK_AI.py
   ```

## ☁️ Deploying to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. In **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   BIO = "your biography text here"
   ```
4. Deploy — no `.env` file needed in the cloud!

## 📚 Example Use Cases
- Portfolio-based AI assistant
- Personal biography showcase
- Academic or career Q&A
- Demo project for open-source LLM integration

## 🤖 Demo Questions
- "What program did Rhichard Koh graduate from?"
- "Describe Rhichard's work on ASL translation."
- "What AI tools has Rhichard used?"

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).

## 🙋‍♂️ Author
Built by [Rhichard Koh](https://github.com/ROCCYK) – feel free to connect or contribute!
