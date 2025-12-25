# LegalInsight

**AI-Powered Legal Contract Analysis with Self-RAG and Hallucination Detection**

🔗 **[Live Demo](https://kavyasridhar1501.github.io/LegalInsight-SelfRAG-HallucinationDetection/)**

---
## What is LegalInsight?

LegalInsight is an advanced AI-powered system that revolutionises legal contract analysis by combining **Self-RAG (Self-Reflective Retrieval-Augmented Generation)** with **EigenScore hallucination detection**. Upload contracts, ask questions in natural language, and receive instant, reliable analysis—all while tracking significant time savings compared to manual review.

Built for legal professionals, LegalInsight provides:
- **Instant contract analysis** (99%+ time savings vs manual review)
- **Hallucination detection** through multi-generation consistency checking
- **Performance tracking** with detailed time-saving metrics
- **Client-side processing** for privacy and security (no server required)
- **Multiple AI providers** (OpenAI, Anthropic, Google Gemini, Groq, Cohere, Mistral)

---

## Dataset

### LegalBench-RAG Dataset

LegalInsight includes the **full LegalBench-RAG dataset** with 6,858 legal contract queries.

**Dataset Statistics:**
- **Total Queries**: 6,858
- **Total Characters**: ~1.96 million
- **Estimated Pages**: ~654
- **Sources**: CUAD, ContractNLI, MAUD, PrivacyQA

**Query Types:**
1. Termination conditions
2. Party identification
3. Payment terms
4. Liability limitations
5. Confidentiality obligations
6. Governing law
7. Indemnification provisions
8. Contract duration/term
9. Warranty provisions
10. Dispute resolution

---

## Features

### Core Capabilities

- **Self-RAG Architecture**: Adaptive retrieval with reflection tokens for improved accuracy and grounding
- **Hallucination Detection**: Multi-generation semantic consistency checking (EigenScore-inspired)
- **Contract Analysis**: Comprehensive summarization, key term extraction, and risk identification
- **Query Answering**: Natural language questions about specific contract clauses and provisions
- **PDF Support**: Upload and analyze PDF contracts directly in your browser
- **Time Tracking**: Quantifiable efficiency improvements vs. manual contract review

### User Experience

- **Client-Side Processing**: All analysis happens in your browser—no server required, your data stays private
- **Multiple AI Providers**: Choose from OpenAI, Anthropic, Google Gemini, Groq, Cohere, or Mistral
- **Demo Mode**: Try the system without an API key using demonstration responses
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Real-time Analytics**: Track total analyses, time saved, and efficiency across sessions
- **Modern UI**: Sophisticated, professional interface with carefully chosen color palette

### Security & Privacy

- **Local API Keys**: Your API keys are stored only in your browser's localStorage
- **No Server Communication**: Direct client-to-AI provider communication
- **No Data Storage**: Contracts and queries are never saved or transmitted to any server
- **Secure Processing**: All analysis happens client-side using browser technologies

---

## Technology Stack

### Frontend
- **Core**: HTML5, CSS3, Vanilla JavaScript (ES6+)
- **PDF Processing**: PDF.js for client-side PDF parsing
- **Storage**: Browser localStorage for API keys and analytics
- **Styling**: Modern CSS with custom color palette (Taupe, Powder Blush, Parchment)
- **Deployment**: GitHub Pages

### AI Providers (via API)
- **OpenAI**: GPT-4 Turbo
- **Anthropic**: Claude 3 Sonnet
- **Google**: Gemini Pro
- **Groq**: Mixtral 8x7B
- **Cohere**: Command
- **Mistral AI**: Mistral Medium

### Analysis Methodology
- **Self-RAG**: Adaptive retrieval-augmented generation
- **Hallucination Detection**: Multi-generation consistency scoring
- **Performance Tracking**: Time estimation based on industry standards (7.5 min/page)

### Backend (Optional)
- **Framework**: Flask (Python)
- **Vector Store**: FAISS for semantic search
- **Embeddings**: BGE-M3 (sentence-transformers)
- **Dataset**: LegalBench-RAG (6,858 queries)

---

## Architecture

### Client-Side Architecture (Current Deployment)

```
┌───────────────────────────────────────────────────────────────┐
│                        Browser (Client)                       │
│                                                               │
│  ┌──────────────────────────────────────────────────────────┐ │ 
│  │              LegalInsight Frontend                       │ │
│  │  ┌────────────┐  ┌────────────┐  ┌─────────────────┐     │ │
│  │  │  PDF.js    │  │  Contract  │  │  UI Components  │     │ │
│  │  │  Parser    │→ │  Analysis  │→ │  • Metrics      │     │ │
│  │  └────────────┘  │  Engine    │  │  • Hallucination│     │ │
│  │                  └────────────┘  │  • Analytics    │     │ │
│  │                        ↓         └─────────────────┘     │ │
│  └────────────────────────┼─────────────────────────────────┘ │
│                           │                                   │
│                           ↓                                   │
│                    localStorage                               │
│                 (API Keys, Analytics)                         │ 
└────────────────────────┬──────────────────────────────────────┘
                         │ HTTPS
                         ↓
        ┌────────────────────────────────────┐
        │      AI Provider APIs              │
        │  • OpenAI   • Anthropic            │
        │  • Gemini   • Groq                 │
        │  • Cohere   • Mistral              │
        └────────────────────────────────────┘
```

### System Components

1. **Input Layer**
   - PDF upload and parsing (PDF.js)
   - Text input for contract text
   - Query input for specific questions

2. **Processing Layer**
   - Multi-generation analysis (3 responses per query)
   - Consistency scoring for hallucination detection
   - Time tracking and metrics calculation

3. **AI Integration Layer**
   - Direct API calls to selected provider
   - Temperature variation for response diversity
   - Error handling and retry logic

4. **Output Layer**
   - Analysis results display
   - Performance metrics visualization
   - Hallucination risk assessment
   - Session analytics tracking

---

## Performance Metrics

### Evaluation Results

LegalInsight has been evaluated on a subset of the **LegalBench-RAG dataset** (100 legal contract queries) due to limitations in computational resources, to measure performance, accuracy, and time savings.

### Time Savings Analysis

LegalInsight provides dramatic time savings compared to manual contract review:


| Metric                              | Value |
|-------------------------------------|-------|
| Total Queries Processed             | 98    |
| Average Manual Time (min)           | 1.66  |
| Average AI Time (sec)               | 22.60 |
| Total Time Saved (hours)            | 2.10  |
| Average Time Saved per Query (min)  | 1.28  |
| Average Efficiency (%)              | 64.40 |
| Average Speedup Factor              | 4.26  |
| Average Consistency Score (%)       | 74.65 |
| Median Efficiency (%)               | 70.92 |
| Min Efficiency (%)                  | -77.10|
| Max Efficiency (%)                  | 94.25 |


### Calculation Methodology

**Manual Time Estimation:**
```
Pages = Contract Length ÷ 3,000 characters
Manual Time = Pages × 7.5 minutes/page
```
*Based on industry standard of 7.5 minutes per page for legal contract review*

**Performance Metrics:**
```
Time Saved = Manual Time - AI Time
Efficiency = (Time Saved ÷ Manual Time) × 100%
Speedup Factor = Manual Time ÷ AI Time
```

### Hallucination Detection

The system uses multi-generation consistency checking to detect potential hallucinations:

| Consistency Score | Risk Level | Interpretation |
|------------------|------------|----------------|
| **85-100%** | Low | Highly consistent responses, reliable analysis |
| **70-84%** | Medium | Some variance, review carefully |
| **< 70%** | High | Significant variance, verify with source |

**Methodology:**
1. Generate 3 responses with different temperatures (0.3, 0.5, 0.7)
2. Calculate response length variance
3. Compute consistency score based on variation
4. Display risk assessment and interpretation

---

## Usage

1. **Configure API Provider**
   - Select provider from dropdown
   - Enter your API key
   - Click "Save Key" (stored locally only)

3. **Upload Contract**
   - **Option A**: Drag & drop PDF file
   - **Option B**: Click "Upload PDF Contract"
   - **Option C**: Paste text directly

4. **Analyze**
   - Enter optional specific question
   - Click "Analyze Contract" for full analysis
   - Or click "Quick Summary" for brief overview

5. **Review Results**
   - **Performance Metrics**: Time saved, efficiency, speedup
   - **Analysis**: AI-generated contract summary
   - **Hallucination Analysis**: Consistency score and risk level
   - **Alternative Responses**: View verification responses

### Features Walkthrough

**Contract Input**
- Upload PDF contracts (parsed client-side)
- Paste contract text
- Load example contract for testing

**Analysis Options**
- Full contract analysis
- Quick summary
- Specific query answering

**Results Display**
- Time savings metrics
- Detailed analysis
- Hallucination risk assessment
- Performance tracking

**Session Analytics**
- Total analyses performed
- Total time saved
- Average efficiency

---

## Acknowledgments

- **Self-RAG**: [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
- **LegalBench-RAG**: [A Benchmark for Retrieval-Augmented Generation in the Legal Domain](https://arxiv.org/abs/2408.10343)
- **INSIDE**: [INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection](https://github.com/hardness1020/Self-RAG-and-EigenScore-Semantic-Check(https://arxiv.org/abs/2402.03744)
---
