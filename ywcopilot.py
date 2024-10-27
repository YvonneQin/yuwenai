import os
import openai
import logging
import numpy as np  # 添加 numpy 导入
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import time
from transformers import BertTokenizer, BertModel
import torch
from langchain_openai import AzureChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 其余代码保持不变

# 加载环境变量和日志记录器
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure OpenAI 配置
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
azure_endpoint = os.getenv("AZURE_OPENAI_API_BASE")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
user_agent = os.getenv("USER_AGENT", "Mozilla/5.0 (compatible; MyBot/1.0)")

# 检查配置
if not all([azure_deployment, azure_endpoint, azure_api_version, azure_api_key]):
    raise ValueError("Missing required Azure OpenAI configuration.")

# 设置 OpenAI API 的配置
openai.api_type = "azure"
openai.api_base = azure_endpoint
openai.api_version = azure_api_version
openai.api_key = azure_api_key

# 初始化会话模型
def get_openai_model():
    return AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        openai_api_version=azure_api_version,
        openai_api_key=azure_api_key
    )

# Chinese-BERT-wwm 嵌入模型
class ChineseBERTEmbeddings:
    def __init__(self, model_name="hfl/chinese-bert-wwm-ext"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 使用 [CLS] token 的嵌入作为句子的整体嵌入
        embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return embeddings

# 初始化 Chinese-BERT-wwm 嵌入模型
bert_embeddings = ChineseBERTEmbeddings()

import faiss
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

# 抓取和嵌入网页内容
def embed_website_content():
    # 加载网页内容
    loader = WebBaseLoader("https://www.yuwen-qin.com/project_Construct.html")
    documents = loader.load()

    # 分割文本以便嵌入
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # 从分割文档中提取文本内容
    texts = [doc.page_content for doc in docs]

    # 使用 Chinese-BERT-wwm 嵌入生成向量
    embeddings = [bert_embeddings.embed_text(text) for text in texts]

    # 初始化 FAISS 索引
    dimension = len(embeddings[0])  # 嵌入的维度
    index = faiss.IndexFlatL2(dimension)  # 使用 L2 距离
    index.add(np.array(embeddings).astype("float32"))  # 添加嵌入到索引中

    return index, texts  # 返回索引和文本，以便在搜索时使用

# 获取网页内容的向量存储
vectorstore, texts = embed_website_content()

# 示例：搜索嵌入
def search_in_vectorstore(query_embedding):
    # 将 query_embedding 变为二维数组，满足 faiss 的输入要求
    query_embedding = np.array(query_embedding).reshape(1, -1).astype("float32")
    distances, indices = vectorstore.search(query_embedding, k=3)
    return [texts[i] for i in indices[0]]  # 返回找到的相关文本

# 生成聊天 prompt
def get_chat_prompt(user_input, context):
    return ChatPromptTemplate.from_messages([
        ("system", "You are a highly knowledgeable assistant."),
        ("system", f"Context: {context}"),
        ("user", "{user_input}")
    ]).format(user_input=user_input)

# 处理重试逻辑
import traceback

def chat_with_retries(model, user_input, retries=3, delay=2):
    for attempt in range(retries):
        try:
            logger.info(f"Attempt {attempt + 1} to call OpenAI API...")
            
            # 打印用户输入
            logger.info(f"User input: {user_input}")
            
            # 查询最相关的内容
            query_embedding = bert_embeddings.embed_text(user_input)
            relevant_texts = search_in_vectorstore(query_embedding)
            context = " ".join(relevant_texts)

            # 打印上下文
            logger.info(f"Context: {context}")

            prompt = get_chat_prompt(user_input, context)
            logger.info(f"Prompt: {prompt}")

            response = model.invoke([HumanMessage(content=prompt)], max_tokens=100, timeout=30)
            logger.info(f"Response received: {response}")

            return response.content
        except (openai.APIConnectionError, openai.OpenAIError) as e:
            if attempt < retries - 1:
                logger.warning(f"Error: {e}. Retrying...")
                time.sleep(delay)
            else:
                logger.error(f"All attempts failed: {e}")
                traceback.print_exc()  # 打印完整的异常堆栈信息
                raise e
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            traceback.print_exc()  # 打印完整的异常堆栈信息
            raise e

# 初始化 Flask 应用
app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get("question", "").strip()
        if not user_input:
            return jsonify({"error": "No question provided"}), 400
        reply = chat_with_retries(get_openai_model(), user_input)
        return jsonify({"response": reply})
    except openai.OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"OpenAI API error: {e}"}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        traceback.print_exc()
        return jsonify({"error": "An unexpected error occurred"}), 500

@app.route('/')
def index():
    return render_template('bot.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5002)), debug=True)