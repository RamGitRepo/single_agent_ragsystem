<img width="1071" height="842" alt="arch" src="https://github.com/user-attachments/assets/1bb721ad-57f2-4e9e-9e78-f15d38db7fc5" />


### 💼 `AI Career Assistant – Semantic Kernel + RAG + Plugin Orchestration`

This project is an intelligent **AI Career Assistant** built using **FastAPI**, **Semantic Kernel**, and **Azure OpenAI**, enhanced with **RAG (Retrieval-Augmented Generation)** and **plugin orchestration**.
It performs real-time CV skill extraction, leverages external search APIs, and returns actionable insights — all grounded in retrieved content.

---

### 🔧 `Features`

- ✅ Semantic Kernel Agent with plugin support
- 🔌 Modular plugin architecture for RAG, search, and query refinement
- 📡 Real-time skill suggestions from CV + external search
- 🔁 End-to-end orchestration with plugin chaining
- 📦 Optional Redis support for memory

---

### 🧠 `Plugin Architecture`

### 📄 `cvSkillsPlugin`

- Calls a local **RAG function** (`/api/rag-query`) with:
  - `question`
  - `conversation_id`
- Returns **extracted skills** from chunked CV content.

---

### 🔍 `SearchSkillPlugin`

- Uses **Tavily Search API** to retrieve external skill-building resources.
- Returns:
  - Summarized answers
  - Top 5 source URLs

---

### 🧹 `QueryRefinerPlugin`

- Uses **Azure OpenAI** to clean and normalize **raw user queries**.
- Removes filler language like _“can you”, “please”, “I want to”_.
- Rewrites queries into **LLM-friendly refined intent**.
- Improves downstream plugin accuracy by standardizing input.

---

### 🔁 `Orchestrated Skill Flow`

User Input
↓
Agent receives input
↓
➀ refine_query() – Normalize the question
↓
➁ get_skills() – Calls RAG API to extract CV skills
↓
➂ search_resources() – Uses Tavily API to suggest improvements

---

### 🤖 `Assistant UI (React + Tailwind)`

This is a modern React.js frontend UI for a conversational AI assistant built with Tailwind CSS. It communicates with a backend RAG system (Retrieval-Augmented Generation) 
to provide smart responses based on user input and cached memory.

---

### ✨ `Features`

- 🔹 React + Tailwind CSS for clean UI
- 🔹 Conversational input/output flow
- 🔹 API integration with RAG backend (FastAPI / Azure Function)
- 🔹 UUID-based session tracking
- 🔹 Gradient background + responsive layout
- 🔹 Typing loader (`Thinking...`) indicator
- 🔹 Supports local or Redis-based chat memory

---

### 📈 `Observability & Monitoring`

### 🔍 `OpenTelemetry Integration`
- Integrated OpenTelemetry to enable full-stack observability for the agentic RAG application. This includes:
- Token Usage Tracking: Logs total prompt/completion tokens used per request, enabling cost analysis and usage optimization.
- Latency Monitoring: Measures time taken for individual plugin invocations (e.g., get_skills, search_resources) and end-to-end agent responses.
- Cost Estimation: Emits metrics to estimate cost per call based on token usage and Azure OpenAI pricing tiers.
- Trace Correlation: Each request is tagged with a unique trace ID for debugging multi-step plugin flows.

---

### 📌 `Coming Soon`
- AI Foundry
- Token Management
- Dashboard
