# 💼 Cashanova AI

> 🧠 A smart AI-powered **Finance Dashboard** built with **Streamlit** to analyze, visualize, and manage your income, expenses, and assets — all in one beautiful interface.  
> 💜 Built and created by **Pramzz**  

---

## 🌟 Overview

**Cashanova AI** is a modern personal finance assistant that helps you track and understand your finances effortlessly.  
Upload your financial PDF, enter data manually, or fill a structured statement — Cashanova AI automatically:
- Extracts and processes financial data  
- Converts across multiple currencies  
- Summarizes your income, expenses, and assets  
- Displays interactive visualizations and metrics  
- Saves your data securely in local CSV files  

> A perfect blend of **AI + Finance + Design**, made for individuals who value both data and style.

---

## ✨ Features

### 🧾 Input Options
- 📂 Upload **Finance PDFs** and extract data automatically  
- ✍️ Enter finance details manually (key-value format)  
- 📊 Fill a structured **Financial Statement Form** with income, assets, and expenses  

### 💱 Currency Settings
- Supports: INR, USD, EUR, GBP, JPY, CAD, AUD, CNY, SGD, AED  
- Auto-converts input values between currencies  
- Shows accurate symbols and rates  

### 📈 Financial Dashboard
- 💰 Total Income  
- 💸 Total Expenses  
- 📊 Net Income  
- 🏦 Total Assets  

> Visualized beautifully using **Plotly Express** interactive charts.

### 💾 Data Management
- Saves all financial records to `finance_data.csv`  
- Automatically logs interactions to `finance_agent_log.csv`  
- Allows easy download of CSVs and chart PNGs  

### 🧠 AI Integration (Optional)
- Supports **Google Gemini API key** for AI-powered Q&A  
- Uses embeddings and retriever system for context-aware financial answers  

### 🎨 Stunning Design
- Gradient background (`#667eea → #764ba2`)  
- Glassmorphism cards  
- Modern **Inter font** and animated buttons  
- Intuitive user experience  

---

## 🧰 Tech Stack

| Category | Technology |
|-----------|-------------|
| Frontend | Streamlit |
| Visualization | Plotly Express |
| AI Engine | Google Generative AI (Gemini) |
| File Processing | PyPDF2 |
| Data Handling | Pandas, CSV |
| Text Splitting | LangChain Text Splitters |
| Language | Python 3.10+ |

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/cashanova-ai.git
cd cashanova-ai
