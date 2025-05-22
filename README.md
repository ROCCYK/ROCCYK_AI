# ROCCYK_AI

**ROCCYK_AI** is a custom-built, open-source AI chatbot trained on the personal biography of Rhichard Koh. Designed as a personalized conversational agent, this project showcases how retrieval-augmented generation (RAG) and fine-tuned language models can be used to create highly customized chatbot experiences.

## 🧠 About

This chatbot serves as an AI-powered digital twin that can answer questions, hold conversations, and share insights based on the real-life biography of its creator. The project demonstrates the capabilities of personalizing large language models for niche, individual use cases, such as portfolio enhancement, education, and digital identity applications.

## 🚀 Features

- 🔍 **Biography-trained model**: The chatbot is trained on a dataset curated from the creator’s academic and professional history.
- 💬 **Conversational agent**: Provides natural responses in Q&A format.
- 🛠️ **Modular architecture**: Easy to extend or adapt for other personalized use cases.
- 📖 **RAG (Retrieval-Augmented Generation)**: Optionally supports integration with vector stores for context-based responses.

## 🏗️ Tech Stack

- **Python**
- **LangChain** / **Haystack** (optional, for RAG)
- **OpenAI GPT / LLM APIs**
- **FAISS / Chroma** (for embedding search)
- **Streamlit / Flask** (optional frontend)

## 📁 Repository Structure

```
ROCCYK_AI/
├── data/               # Biographical documents
├── chatbot/            # Core chatbot logic
├── embeddings/         # Vector DB and retriever
├── ui/                 # Optional frontend interface
├── requirements.txt    # Python dependencies
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

3. **Prepare your `.env` file**
   Include any necessary API keys (e.g., OpenAI key).

4. **Run the chatbot**
   Depending on implementation:
   ```bash
   python chatbot/main.py
   ```

   Or for web:
   ```bash
   streamlit run ui/app.py
   ```

## 📚 Example Use Cases

- Portfolio-based AI assistant
- Personal biography showcase
- Academic or career Q&A
- Demo project for LLM fine-tuning

## 🤖 Demo Questions

- “What program did Rhichard Koh graduate from?”
- “Describe Rhichard’s work on ASL translation.”
- “What AI tools has Rhichard used?”

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

## 🙋‍♂️ Author

Built by [Rhichard Koh](https://github.com/ROCCYK) – feel free to connect or contribute!
