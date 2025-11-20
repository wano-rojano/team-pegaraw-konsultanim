# Konsultanim

A sophisticated multi-crop agricultural satellite data and AI-powered assistant built with LangGraph and the **A2A (Agent-to-Agent) Protocol**. Konsultanim provides expert guidance on crop diseases, integrated pest management (IPM), weather-based farming advisories, and crop insurance assistance for Filipino farmers growing **rice, corn, and coconut** through both a web API and an intuitive Chainlit chat interface, and SMS capabilities.

## 🎯 Features

### Multi-Agent System
- **Crop Doctor Agent**: Disease and pest diagnosis for rice, corn, and coconut
- **Advisory Agent**: Weather forecasts and disease risk assessments
- **Insurance Agent**: Crop insurance policy guidance and claims assistance

### Multi-Channel Access
- 📱 **SMS Interface**: Twilio-powered SMS gateway for farmers
- 🌐 **Web API**: A2A protocol-compliant REST API
- 💬 **Chat Interface**: Chainlit-based conversational UI

### Evidence-Based Approach
- RAG over curated references and academic papers (diseases, pests, insurance policies)
- Real-time weather data integration (Open-Meteo API)
- Academic research access (PubMed, arXiv)
- Mandatory source citations

## 📚 Supported Crops

- **Rice** (Palay): Blast, bacterial blight, sheath blight, tungro, stem borers
- **Corn** (Mais): Borer, common cutworm, armyworm, earworm
- **Coconut** (Niyog): Cadang-Cadang, bud rot, leaf beetle, palm weevil

## 🏗️ Architecture

```
Konsultanim/
├── data/
│   ├── rice/          # Rice disease PDFs
│   ├── corn/          # Corn disease PDFs
│   ├── coconut/       # Coconut disease PDFs
│   └── insurance/     # Crop insurance policy PDFs
├── app/
│   ├── crop_doctor_agent.py   # Multi-crop diagnosis
│   ├── advisory_agent.py      # Weather & advisory
│   ├── insurance_agent.py     # Insurance assistance
│   ├── agent_executor.py      # Intelligent routing
│   ├── rag.py                 # Multi-category RAG
│   └── tools.py               # Tool integration
```

## 🚀 Quick Start

1. **Setup environment:**
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

2. **Populate data folders:**
```bash
mkdir -p data/{rice,corn,coconut,insurance}
# Add PDFs to respective folders
```

3. **Run A2A server:**
```bash
python -m app
# Runs on port 10000
```

4. **Run SMS server:**
```bash
python sms_server.py
# Runs on port 5000
```

## 📱 SMS Usage

Farmers can text queries like:
- "My corn has brown lesions on leaves" → Crop Doctor
- "Weekly weather for rice in Laguna" → Advisory Agent
- "How to file crop insurance claim?" → Insurance Agent

## 🔧 Configuration

Key environment variables:
```bash
DASHSCOPE_API_KEY=xxx
RAG_DATA_DIR=data
TWILIO_ACCOUNT_SID=xxx
TWILIO_AUTH_TOKEN=xxx
FARMER_REGISTRY="+639xxx:Laguna:rice,+639yyy:Nueva Ecija:corn"
```

## 📊 Agent Routing

Automatic routing based on query keywords:
- **Insurance keywords** → Insurance Agent
- **Weather keywords** → Advisory Agent
- **Default** → Crop Doctor Agent

---

Built with ❤️ for Filipino farmers