# 💼 Cashanova AI

> 🧠 A next-generation **personal finance dashboard** built with **Streamlit** and powered by **Google Gemini AI**.  
> Analyze, visualize, and manage your finances effortlessly — in style.  
> 💜 *Built and created by **Pramzz***  

---

## 🌟 Overview

**Cashanova AI** is a smart financial analytics app that lets you:

- Enter or upload your financial data  
- Automatically calculate income, expenses, profit, and assets  
- Visualize data with rich, interactive charts  
- Save and export your reports to CSV or image formats  
- Optionally connect to **Google Gemini AI** for intelligent insights  

The app combines clean UI, multi-currency support, and practical analytics to help you understand your money beautifully.

---

## ✨ Features

### 🧾 Input Options
- ✍️ **Enter Data Manually** — simple key : value pairs (e.g., `Salary: 25000`)  
- 📊 **Structured Financial Statement** — dedicated fields for income, expenses, and assets  

### 💱 Currency Conversion
- Supports **10+ currencies** including INR, USD, EUR, GBP, JPY, AUD, CAD, CNY, SGD, AED  
- Auto-converts between selected and input currencies  
- Displays values with correct symbols and formatting  

### 📈 Financial Dashboard
- Real-time summary cards for **Income**, **Expenses**, **Net Income**, and **Assets**  
- Visual charts built with **Plotly Express**:
  - 💵 Income Breakdown (Pie Chart)  
  - 💸 Expense Breakdown (Pie Chart)  
  - 📊 Income vs Expenses Comparison (Bar Chart)  
  - 🏦 Assets Breakdown (Bar Chart)  

### 💾 Data Storage & Export
- Automatically logs and saves data to `finance_data.csv`  
- Download:
  - 📑 Finance CSV file  
  - 🖼️ Charts as PNG images (requires `kaleido`)  

### 🧠 AI-Ready (Optional)
- Configure your **Google Gemini API Key**  
- Embedding functions & retriever classes are included for future finance Q&A  

### 🎨 Elegant UI
- Gradient theme (`#667eea → #764ba2`)  
- Glassmorphism cards, soft shadows & rounded inputs  
- Inter Font Family + animated buttons  
- Fully responsive Streamlit layout  

---

## 🧰 Tech Stack

| Category | Technology |
|-----------|-------------|
| Language | Python 3.10+ |
| Framework | Streamlit |
| Visualization | Plotly Express |
| Data Handling | Pandas, CSV |
| AI Engine | Google Generative AI (Gemini) |
| File Parsing | LangChain Text Splitters |
| Misc | PyPDF2, Math, Dataclasses |

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/cashanova-ai.git
cd cashanova-ai
