import os
import csv
import math
from dataclasses import dataclass
import streamlit as st
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
from typing import Any, Dict, List
import pandas as pd
import plotly.express as px
import io

# ------------------------------------------------------------
# 🌈 PAGE SETUP
# ------------------------------------------------------------
st.set_page_config(page_title="💼 Cashanova AI", page_icon="💰", layout="wide")

# Enhanced CSS styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    .title {
        text-align: center;
        font-size: 3rem;
        background: -webkit-linear-gradient(45deg, #00b4d8, #48cae4, #90e0ef, #caf0f8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #e0e0e0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 12px;
        border: 2px solid #667eea;
        background-color: rgba(255, 255, 255, 0.05);
        color: white;
        padding: 0.75rem;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #48cae4;
        box-shadow: 0 0 10px rgba(72, 202, 228, 0.3);
    }
    
    .stButton > button {
        border-radius: 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }
    
    h1, h2, h3 {
        color: #ffffff !important;
    }
    
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 2px solid #667eea;
    }
    
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border: 2px dashed #667eea;
        padding: 2rem;
    }
    
    .stRadio > div {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">💼 Cashanova AI </div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Built and Created by<b>Pramzz</b> </div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# 🔑 GEMINI API CONFIG
# ------------------------------------------------------------
with st.container():
    st.markdown("### 🔐API Configuration")
    api_key = st.text_input("Enter your Google Gemini API key (starts with 'AIza'):", type="password", help="Get your API key from Google AI Studio")

if not api_key:
    st.warning("⚠️ Please enter your Gemini API key to continue.")
    st.stop()

try:
    genai.configure(api_key=api_key)
    os.environ["GOOGLE_API_KEY"] = api_key
    st.success("✅ Gemini API configured successfully!")
except Exception as e:
    st.error(f"❌ Failed to configure Gemini API: {e}")
    st.stop()

# ------------------------------------------------------------
# 💱 CURRENCY SELECTION & CONVERSION
# ------------------------------------------------------------
st.markdown("---")
st.markdown("### 💱 Currency Settings")

# Currency definitions with symbols and conversion rates (to USD)
CURRENCIES = {
    "USD": {"symbol": "$", "name": "US Dollar", "rate": 1.0},
    "INR": {"symbol": "₹", "name": "Indian Rupee", "rate": 83.0},
    "EUR": {"symbol": "€", "name": "Euro", "rate": 0.92},
    "GBP": {"symbol": "£", "name": "British Pound", "rate": 0.79},
    "JPY": {"symbol": "¥", "name": "Japanese Yen", "rate": 150.0},
    "CAD": {"symbol": "C$", "name": "Canadian Dollar", "rate": 1.35},
    "AUD": {"symbol": "A$", "name": "Australian Dollar", "rate": 1.52},
    "CNY": {"symbol": "¥", "name": "Chinese Yuan", "rate": 7.2},
    "SGD": {"symbol": "S$", "name": "Singapore Dollar", "rate": 1.34},
    "AED": {"symbol": "د.إ", "name": "UAE Dirham", "rate": 3.67},
}

col1, col2 = st.columns([2, 1])
with col1:
    selected_currency = st.selectbox(
        "Select Currency:",
        options=list(CURRENCIES.keys()),
        format_func=lambda x: f"{CURRENCIES[x]['symbol']} {CURRENCIES[x]['name']} ({x})",
        index=1,  # Default to INR (Rupees)
        help="Select your preferred currency for displaying amounts"
    )

with col2:
    # Option to convert from another currency
    convert_from = st.selectbox(
        "Convert from:",
        options=["None"] + list(CURRENCIES.keys()),
        format_func=lambda x: f"{CURRENCIES[x]['symbol']} {x}" if x != "None" else "None",
        help="If your input is in a different currency, select it here"
    )

# Store currency info
currency_info = CURRENCIES[selected_currency]
currency_symbol = currency_info["symbol"]
currency_rate = currency_info["rate"]

def format_currency(amount: float, show_symbol: bool = True) -> str:
    """Format amount with selected currency symbol"""
    if show_symbol:
        return f"{currency_symbol}{amount:,.2f}"
    return f"{amount:,.2f}"

def convert_amount(amount: float, from_currency: str = None) -> float:
    """Convert amount from one currency to selected currency"""
    if from_currency and from_currency != "None" and from_currency in CURRENCIES:
        # Convert from source currency to USD, then to target currency
        from_rate = CURRENCIES[from_currency]["rate"]
        to_rate = currency_rate
        # amount in source currency -> USD -> target currency
        usd_amount = amount / from_rate
        converted_amount = usd_amount * to_rate
        return converted_amount
    return amount

if convert_from != "None":
    st.info(f"💱 Amounts will be converted from {CURRENCIES[convert_from]['symbol']} {convert_from} to {currency_symbol} {selected_currency}")

# ------------------------------------------------------------
# 📄 INPUT OPTIONS
# ------------------------------------------------------------
st.markdown("---")
st.markdown("### 🧾 Choose Input Type")

# Toggle to fully disable Gemini Q&A features (keeps CSV and charts only)
ENABLE_QA = False

option = st.radio("Select input method:", ["Enter Data Manually", "Structured Financial Statement"], horizontal=True)
raw_text = ""
financial_data = {}

if option == "Enter Data Manually":
    st.info(f"💱 Enter amounts in {currency_symbol} {selected_currency}. Currency symbols will be automatically detected and converted if needed.")
    raw_text = st.text_area(
        "📝 Enter your finance details (income, expenses, etc.):",
        placeholder=f"Example:\nSalary: 25000\nRent: 5000\nInvestment: 2500\nProfit: 2000\nLoss: 1000\nRevenue: 50000\nExpenses: 15000\n\n(Amounts will be displayed in {currency_symbol} {selected_currency})",
        height=220,
        help="Enter financial data in key: value format. Currency symbols are auto-detected."
    )
    if not raw_text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
        st.stop()
    st.success("✅ Manual data captured successfully!")

else:  # Structured Financial Statement
    st.markdown("#### 📊 Enter Financial Statement Details")
    st.info(f"💱 Enter amounts in {currency_symbol} {selected_currency}. If your data is in a different currency, use the 'Convert from' option above.")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💰 Income & Revenue**")
        salary = st.number_input("Salary", min_value=0.0, value=0.0, step=1000.0)
        revenue = st.number_input("Revenue", min_value=0.0, value=0.0, step=1000.0)
        investment_income = st.number_input("Investment Income", min_value=0.0, value=0.0, step=1000.0)
        other_income = st.number_input("Other Income", min_value=0.0, value=0.0, step=1000.0)
        
        st.markdown("**📈 Assets**")
        cash = st.number_input("Cash", min_value=0.0, value=0.0, step=1000.0)
        investments = st.number_input("Investments", min_value=0.0, value=0.0, step=1000.0)
        property_value = st.number_input("Property Value", min_value=0.0, value=0.0, step=1000.0)
    
    with col2:
        st.markdown("**💸 Expenses**")
        rent = st.number_input("Rent", min_value=0.0, value=0.0, step=1000.0)
        utilities = st.number_input("Utilities", min_value=0.0, value=0.0, step=100.0)
        groceries = st.number_input("Groceries", min_value=0.0, value=0.0, step=100.0)
        transportation = st.number_input("Transportation", min_value=0.0, value=0.0, step=100.0)
        other_expenses = st.number_input("Other Expenses", min_value=0.0, value=0.0, step=1000.0)
        
        st.markdown("**📊 Financial Metrics**")
        profit = st.number_input("Profit", min_value=0.0, value=0.0, step=1000.0)
        loss = st.number_input("Loss", min_value=0.0, value=0.0, step=1000.0)
    
    # Build structured data
    financial_data = {
        "Salary": salary,
        "Revenue": revenue,
        "Investment Income": investment_income,
        "Other Income": other_income,
        "Total Income": salary + revenue + investment_income + other_income,
        "Rent": rent,
        "Utilities": utilities,
        "Groceries": groceries,
        "Transportation": transportation,
        "Other Expenses": other_expenses,
        "Total Expenses": rent + utilities + groceries + transportation + other_expenses,
        "Cash": cash,
        "Investments": investments,
        "Property Value": property_value,
        "Total Assets": cash + investments + property_value,
        "Profit": profit,
        "Loss": loss,
        "Net Income": (salary + revenue + investment_income + other_income) - (rent + utilities + groceries + transportation + other_expenses)
    }
    
    # Convert to text format for processing
    raw_text = "\n".join([f"{k}: {v}" for k, v in financial_data.items() if v > 0])
    
    if not raw_text.strip():
        st.warning("⚠️ Please enter some financial data to analyze.")
        st.stop()
    st.success("✅ Financial statement data captured successfully!")

# ------------------------------------------------------------
# 📊 VISUALIZE DATA (if structured)
# ------------------------------------------------------------
fig_income = None
fig_expense = None
fig_bar = None
fig_assets = None

if financial_data and any(v > 0 for v in financial_data.values()):
    st.markdown("---")
    st.markdown("### 📊 Financial Overview Dashboard")
    
    # Calculate metrics
    total_income = financial_data.get("Total Income", 0)
    total_expenses = financial_data.get("Total Expenses", 0)
    net_income = financial_data.get("Net Income", total_income - total_expenses)
    total_assets = financial_data.get("Total Assets", 0)
    
    col1, col2, col3, col4 = st.columns(4)
    # Convert amounts if needed
    total_income_conv = convert_amount(total_income, convert_from)
    total_expenses_conv = convert_amount(total_expenses, convert_from)
    net_income_conv = convert_amount(net_income, convert_from)
    total_assets_conv = convert_amount(total_assets, convert_from)
    
    with col1:
        st.metric("💰 Total Income", format_currency(total_income_conv), delta=f"{net_income_conv/total_income_conv*100 if total_income_conv > 0 else 0:.1f}% net" if total_income_conv > 0 else None)
    with col2:
        st.metric("💸 Total Expenses", format_currency(total_expenses_conv), delta=f"{total_expenses_conv/total_income_conv*100 if total_income_conv > 0 else 0:.1f}% of income" if total_income_conv > 0 else None)
    with col3:
        st.metric("📈 Net Income", format_currency(net_income_conv), delta=format_currency(net_income_conv) if net_income_conv >= 0 else None)
    with col4:
        st.metric("🏦 Total Assets", format_currency(total_assets_conv))
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Income vs Expenses Pie Chart
        income_data = {
            "Salary": financial_data.get("Salary", 0),
            "Revenue": financial_data.get("Revenue", 0),
            "Investment Income": financial_data.get("Investment Income", 0),
            "Other Income": financial_data.get("Other Income", 0)
        }
        expense_data = {
            "Rent": financial_data.get("Rent", 0),
            "Utilities": financial_data.get("Utilities", 0),
            "Groceries": financial_data.get("Groceries", 0),
            "Transportation": financial_data.get("Transportation", 0),
            "Other Expenses": financial_data.get("Other Expenses", 0)
        }
        
        income_filtered = {k: v for k, v in income_data.items() if v > 0}
        expense_filtered = {k: v for k, v in expense_data.items() if v > 0}
        
        if income_filtered:
            # Convert income values
            income_converted = {k: convert_amount(v, convert_from) for k, v in income_filtered.items()}
            fig_income = px.pie(
                values=list(income_converted.values()),
                names=list(income_converted.keys()),
                title=f"💵 Income Breakdown ({currency_symbol})",
                color_discrete_sequence=px.colors.sequential.Greens
            )
            fig_income.update_traces(texttemplate=f'{currency_symbol}%{{value:,.0f}}', textposition='outside')
            fig_income.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_size=16
            )
            st.plotly_chart(fig_income, use_container_width=True)
        
        if expense_filtered:
            # Convert expense values
            expense_converted = {k: convert_amount(v, convert_from) for k, v in expense_filtered.items()}
            fig_expense = px.pie(
                values=list(expense_converted.values()),
                names=list(expense_converted.keys()),
                title=f"💸 Expense Breakdown ({currency_symbol})",
                color_discrete_sequence=px.colors.sequential.Reds
            )
            fig_expense.update_traces(texttemplate=f'{currency_symbol}%{{value:,.0f}}', textposition='outside')
            fig_expense.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_size=16
            )
            st.plotly_chart(fig_expense, use_container_width=True)
    
    with col2:
        # Bar Chart: Income vs Expenses
        comparison_data = pd.DataFrame({
            "Category": ["Total Income", "Total Expenses", "Net Income"],
            "Amount": [total_income_conv, total_expenses_conv, max(0, net_income_conv)]
        })
        
        fig_bar = px.bar(
            comparison_data,
            x="Category",
            y="Amount",
            title=f"📊 Income vs Expenses Comparison ({currency_symbol})",
            color="Category",
            color_discrete_map={
                "Total Income": "#00b4d8",
                "Total Expenses": "#ff6b6b",
                "Net Income": "#51cf66"
            }
        )
        fig_bar.update_traces(texttemplate=f'{currency_symbol}%{{y:,.0f}}', textposition='outside')
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_size=16,
            showlegend=False,
            yaxis_title=f"Amount ({currency_symbol})"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Assets breakdown
        assets_data = {
            "Cash": financial_data.get("Cash", 0),
            "Investments": financial_data.get("Investments", 0),
            "Property Value": financial_data.get("Property Value", 0)
        }
        assets_filtered = {k: v for k, v in assets_data.items() if v > 0}
        
        if assets_filtered:
            # Convert asset values
            assets_converted = {k: convert_amount(v, convert_from) for k, v in assets_filtered.items()}
            fig_assets = px.bar(
                x=list(assets_converted.keys()),
                y=list(assets_converted.values()),
                title=f"🏦 Assets Breakdown ({currency_symbol})",
                color=list(assets_converted.values()),
                color_continuous_scale="Viridis"
            )
            fig_assets.update_traces(texttemplate=f'{currency_symbol}%{{y:,.0f}}', textposition='outside')
            fig_assets.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_size=16,
                showlegend=False,
                yaxis_title=f"Amount ({currency_symbol})"
            )
            st.plotly_chart(fig_assets, use_container_width=True)

# ------------------------------------------------------------
# ✂️ TEXT SPLITTING
# ------------------------------------------------------------
if ENABLE_QA and raw_text:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(raw_text)
    st.info(f"🧩 Document split into {len(chunks)} parts for vectorization.")

# ------------------------------------------------------------
# 🔠 EMBEDDING FUNCTIONS (FIXED)
# ------------------------------------------------------------
@dataclass
class TextDocument:
    page_content: str

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)

def embed_text(text: str) -> List[float]:
    """Fixed embedding function that handles both dict and object responses"""
    try:
        res = genai.embed_content(model="models/text-embedding-004", content=text)
        
        # Handle dict response
        if isinstance(res, dict):
            embedding = res.get("embedding", [])
            if embedding:
                return list(embedding)
        
        # Handle object response
        if hasattr(res, "embedding"):
            embedding = getattr(res, "embedding", [])
            if embedding:
                return list(embedding)
        
        # Try accessing as dict key
        if hasattr(res, "__dict__"):
            return list(getattr(res, "__dict__", {}).get("embedding", []))
        
        # Last resort: try to get the first element if it's a list-like structure
        if isinstance(res, list) and len(res) > 0:
            if isinstance(res[0], dict):
                return list(res[0].get("embedding", []))
            return list(res[0]) if hasattr(res[0], "__iter__") else []
        
        raise ValueError(f"Could not extract embedding from response: {type(res)}")
    except Exception as e:
        st.error(f"Embedding error for text: {str(e)}")
        # Return zero vector as fallback
        return [0.0] * 768

def embed_documents(texts: List[str]) -> List[List[float]]:
    return [embed_text(t) for t in texts]

class SimpleEmbeddingRetriever:
    def __init__(self, docs: List[TextDocument], doc_vectors: List[List[float]], k: int = 3):
        self.docs = docs
        self.doc_vectors = doc_vectors
        self.k = k

    def get_relevant_documents(self, query: str) -> List[TextDocument]:
        query_vec = embed_text(query)
        scored = [( _cosine_similarity(query_vec, vec), doc) for vec, doc in zip(self.doc_vectors, self.docs)]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [doc for _, doc in scored[: self.k]]

# ------------------------------------------------------------
# ⚙️ BUILD RETRIEVER
# ------------------------------------------------------------
if ENABLE_QA and raw_text:
    try:
        docs = [TextDocument(page_content=c) for c in chunks]
        doc_vectors = embed_documents(chunks)
        retriever = SimpleEmbeddingRetriever(docs, doc_vectors, k=3)
        st.success("✅ Embeddings & retriever ready.")
    except Exception as e:
        st.error(f"❌ Embedding/retriever error: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        st.stop()

# ------------------------------------------------------------
# 🧠 SIMPLE QA MODEL
# ------------------------------------------------------------
class SimpleQA:
    def __init__(self, llm: Any, retriever: Any):
        self.llm = llm
        self.retriever = retriever

    def __call__(self, query: str) -> Dict[str, Any]:
        docs = self.retriever.get_relevant_documents(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = (
            "You are a helpful financial assistant. Use ONLY the provided context. "
            "If the answer isn't in the context, say 'I don't know.'\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        result = self.llm.generate_content(prompt)
        answer = getattr(result, "text", "")
        return {"result": answer, "source_documents": docs}

# ------------------------------------------------------------
# 🔮 INIT GEMINI MODEL (with fallback options)
# ------------------------------------------------------------
if ENABLE_QA and raw_text:
    # Try different model names in order of preference
    model_names = ["gemini-1.5-pro", "gemini-pro", "gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-pro"]
    llm = None
    
    for model_name in model_names:
        try:
            llm = genai.GenerativeModel(model_name)
            # Test the model with a simple call
            test_response = llm.generate_content("test")
            st.success(f"✅ Gemini model '{model_name}' initialized successfully!")
            break
        except Exception as e:
            continue
    
    if llm is None:
        st.error("❌ Could not initialize any Gemini model. Please check your API key and model availability.")
        st.info("💡 Try using: gemini-1.5-pro, gemini-pro, or gemini-1.5-flash")
        st.stop()
    
    try:
        qa_chain = SimpleQA(llm=llm, retriever=retriever)
    except Exception as e:
        st.error(f"❌ Error creating QA chain: {e}")
        st.stop()

# ------------------------------------------------------------
# 📊 CSV LOGGING & MANAGEMENT
# ------------------------------------------------------------
csv_file = "finance_agent_log.csv"
finance_csv_file = "finance_data.csv"

def initialize_csv():
    """Initialize CSV file with headers if it doesn't exist"""
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Question", "Answer"])

def initialize_finance_csv():
    """Initialize finance data CSV with headers if it doesn't exist"""
    if not os.path.exists(finance_csv_file):
        with open(finance_csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Currency", "Key", "Amount"])

def append_to_csv(timestamp: str, question: str, answer: str):
    """Append a new row to the CSV file"""
    try:
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, question, answer])
        return True
    except Exception as e:
        st.error(f"Error writing to CSV: {e}")
        return False

def read_csv_data():
    """Read all data from CSV file"""
    try:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            return df
        return pd.DataFrame(columns=["Timestamp", "Question", "Answer"])
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame(columns=["Timestamp", "Question", "Answer"])

# Initialize CSV on startup
initialize_csv()
initialize_finance_csv()

def append_finance_data(currency_code: str, records: List[Dict[str, Any]]):
    """Append finance records (list of {key, amount}) to finance_data.csv"""
    try:
        rows = []
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for rec in records:
            rows.append([ts, currency_code, rec["key"], rec["amount"]])
        if rows:
            with open(finance_csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            return True
        return False
    except Exception as e:
        st.error(f"Error writing finance data CSV: {e}")
        return False

def read_finance_csv():
    try:
        if os.path.exists(finance_csv_file):
            return pd.read_csv(finance_csv_file)
        return pd.DataFrame(columns=["Timestamp", "Currency", "Key", "Amount"])
    except Exception as e:
        st.error(f"Error reading finance CSV: {e}")
        return pd.DataFrame(columns=["Timestamp", "Currency", "Key", "Amount"])

## Q&A section disabled (ENABLE_QA=False)

# ------------------------------------------------------------
# 💰 AUTO SUMMARY TABLE
# ------------------------------------------------------------
if raw_text:
    st.markdown("---")
    st.markdown("### 📊 Finance Summary Table")
    
    try:
        lines = raw_text.splitlines()
        data = []
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().capitalize()
                try:
                    # Remove common currency symbols
                    value_str = value.strip().replace(",", "").replace("$", "").replace("₹", "").replace("€", "").replace("£", "").replace("¥", "").replace("C$", "").replace("A$", "").replace("S$", "").replace("د.إ", "")
                    value = float(value_str)
                    # Convert if needed
                    value = convert_amount(value, convert_from)
                except:
                    value = 0
                if value > 0:  # Only show non-zero values
                    data.append((key, value))
    
        if data:
            df = pd.DataFrame(data, columns=["Category", "Amount"])
            df = df.sort_values("Amount", ascending=False)
            
            # Style the dataframe with selected currency
            def format_amount(val):
                return format_currency(val)
            
            styled_df = df.style.format({"Amount": lambda x: format_currency(x)}).highlight_max(
                color="lightgreen", subset=["Amount"]
            ).highlight_min(
                color="lightcoral", subset=["Amount"]
            )
            
            st.dataframe(styled_df, use_container_width=True)

            # Allow saving current finance data to CSV
            with st.container():
                st.markdown("#### 💾 Save Finance Data")
                save_col1, save_col2 = st.columns([1, 4])
                with save_col1:
                    if st.button("Save Finance Data to CSV", use_container_width=True):
                        records = [{"key": row["Category"], "amount": row["Amount"]} for _, row in df.iterrows()]
                        ok = append_finance_data(selected_currency, records)
                        if ok:
                            st.success("✅ Finance data saved to finance_data.csv")
                        else:
                            st.warning("⚠️ No finance data to save or write failed.")
        else:
            st.info("📭 No numeric data detected to summarize.")
    except Exception as e:
        st.error(f"❌ Error parsing summary: {e}")

# ------------------------------------------------------------
# 📥 CSV DOWNLOAD & VIEWING SECTION
# ------------------------------------------------------------
st.markdown("---")
## Conversation history removed (Q&A disabled)

# Read and display CSV data
pass

# ------------------------------------------------------------
# 📦 Finance Data CSV & Chart Downloads
# ------------------------------------------------------------
st.markdown("---")
st.markdown("### 📦 Finance Data Export")

finance_df = read_finance_csv()

if not finance_df.empty:
    st.markdown("#### 📑 Saved Finance Data")
    st.dataframe(finance_df, use_container_width=True, height=260)

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        st.download_button(
            label="📥 Download Finance CSV",
            data=finance_df.to_csv(index=False).encode('utf-8'),
            file_name=f"finance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Chart downloads (if figures exist and kaleido available)
    def fig_to_png_bytes(fig):
        try:
            return fig.to_image(format="png", scale=2)
        except Exception:
            return None

    charts_available = any([fig_income, fig_expense, fig_bar, fig_assets])
    if charts_available:
        st.markdown("#### 🖼️ Download Charts as PNG")
        c1, c2, c3, c4 = st.columns(4)
        if fig_income:
            img = fig_to_png_bytes(fig_income)
            with c1:
                if img:
                    st.download_button("Income Pie", data=img, file_name="income_breakdown.png", mime="image/png")
                else:
                    st.caption("Install 'kaleido' to enable chart downloads")
        if fig_expense:
            img = fig_to_png_bytes(fig_expense)
            with c2:
                if img:
                    st.download_button("Expense Pie", data=img, file_name="expense_breakdown.png", mime="image/png")
        if fig_bar:
            img = fig_to_png_bytes(fig_bar)
            with c3:
                if img:
                    st.download_button("Income vs Expense", data=img, file_name="income_vs_expenses.png", mime="image/png")
        if fig_assets:
            img = fig_to_png_bytes(fig_assets)
            with c4:
                if img:
                    st.download_button("Assets Bar", data=img, file_name="assets_breakdown.png", mime="image/png")
else:
    st.info("📭 No saved finance data yet. Use 'Save Finance Data to CSV' above.")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.markdown("<div style='text-align:center; color:#999; padding: 2rem;'>Built and Created by Pramz <b>Pramzz</b> </div>", unsafe_allow_html=True)
