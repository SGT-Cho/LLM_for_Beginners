# LLM, Llama, Deepseek R1, LangChain, RAG, Vector DB, LLM Finetuning

This notebook has been created to **demonstrate how to experiment with a Local Large Language Model (LLM) in a local environment**.

---

## 1. LLM (Large Language Model)

An **LLM** is a language model with an extremely large number of parameters, showing **high performance** in various areas of **Natural Language Processing (NLP)**.  
- Examples: **GPT-4**, **Llama3**, **Phi4**, **Deepseek R1**, etc.

---

## 2.1 Llama

**Llama** is a **large language model** family released by Meta.  
- Pretrained **Llama models** can be run **locally** via Hugging Face Transformers.

## 2.2 Deepseek R1

**Deepseek R1** is an inference model based on reinforcement learning (RL).  
- DeepSeek is a Chinese AI company established in 2023 by the High-Flyer hedge fund. They have released various open-source large language models such as DeepSeek LLM, DeepSeek Coder, DeepSeek Math, and so on.  
  In particular, the recently released DeepSeek-V3 has gained attention for performance on par with Claude 3.5 Sonnet or Gemini 1.5 Pro.

- Recently, they open-sourced **DeepSeek-R1** and **DeepSeek-R1-Zero**, which maximize inference abilities through **reinforcement learning (RL)**. These two models were open-sourced on January 20, 2025.

---

## 3. LangChain

**LangChain** is a Python-based library that provides a variety of features to **systematically develop** and **scale** applications that utilize **large language models (LLMs)**.

- **Prompt Management**  
  - Organizes and **systematizes** multiple prompts for efficient reuse and management.

- **Chaining**  
  - Allows you to **sequentially connect** multiple steps of LLM tasks in a **pipeline** format.  
  - Example: Summarization → Q&A → Sentiment Analysis, etc. You can easily implement multi-step workflows.

- **Agents**  
  - Modules that can **dynamically perform tasks** by accessing specific tools (API, DB, etc.) and make **reasoning-based** decisions to **solve problems**.

- **Memory**  
  - A feature that manages **previous conversations or context** so that the model can “remember” them.

- **Tool Integration**  
  - Integrates easily with a wide range of external tools (databases, third-party APIs, web search, etc.) to provide **rich functionality**.

Because all of these capabilities can be **combined**, it is easy to **rapidly prototype** LLM-based applications and expand them into **production environments**.

---

## 4. RAG (Retrieval-Augmented Generation)

**RAG (Retrieval-Augmented Generation)** is a method in which a large language model does not rely solely on its **internally learned parameters** but also queries **external knowledge bases** (documents, databases, web search results, etc.) in **real time** to retrieve and use necessary information.

1. **Retrieve**  
   - Uses a search engine such as a **Vector DB** to find **relevant documents or information** corresponding to the user query (or conversation context).  
   - Through similarity search based on **embedding vectors**, the model **quickly obtains** the information it needs.

2. **Generate**  
   - The **LLM** **generates** the answer using the retrieved documents.  
   - Leveraging **specific details** from the documents, the model returns **factually rich** and **highly accurate** text.

**Advantages**  
- **Improved Accuracy**: Can utilize the latest data or knowledge not contained in the model’s training.  
- **Overcoming Memory Limits**: You do not have to store all knowledge within model parameters, allowing you to **maintain an efficient model size**.  
- **Flexibility**: Can combine various data types (text, image captions, DB content, etc.).

---

## 5. Vector DB

A **Vector DB** is a **specialized database** for storing **embedding vectors** (from text, images, etc.) and performing **fast searches** for vectors (documents, images) that have **high similarity**.

- **Key Features**  
  1. **Vector Insertion**  
     - Converts text/images to vectors using a pretrained model (e.g., SentenceTransformer, BERT, etc.) and stores them.  
  2. **Similarity Search (ANN Search)**  
     - Utilizes Approximate Nearest Neighbor Search techniques to **efficiently** find the **most similar vectors (documents)** among a large vector set.  
  3. **Scalability**  
     - Maintains **fast search speed** even for very large datasets through **scaling** and distributed processing.

- **Representative Vector DB examples**  
  - **FAISS**: A vector similarity search library developed by Meta (formerly Facebook).  
  - **Chroma**: A vector DB that can easily scale from personal projects to enterprise services.  
  - **Milvus**: A high-performance, large-scale vector search engine.  
  - **Pinecone**: A fully managed cloud-based vector DB service.

**Use Cases**  
- **RAG** (Retrieval-Augmented Generation) for **document search**  
- **Similarity-based recommendation** systems  
- **Multi-modal** search for images, audio, etc.

---

## 6. LLM Finetuning

**LLM Finetuning** is the process of **additional training** a pretrained large language model (e.g., GPT, BERT, etc.) to **customize** it for a specific task or domain.  
- You can **significantly improve** the model’s performance on **specific data** while reusing existing parameters.

### 6.1 Finetuning Methods

1. **Full Finetuning**  
   - Updating **all** model parameters, usually through **epoch-based** optimization.  
   - Used when you have plenty of data and computational resources.

2. **PEFT (LoRA, Prefix Tuning, etc.)**  
   - Approaches that **update only part of the parameters** or that enable **low-cost** additional training.  
   - **LoRA (Low-Rank Adaptation)**: Trains certain weight matrices in the model in a **low-rank** form to achieve finetuning effects with **less memory usage**.  
   - **Prefix Tuning**: Adds a **virtual prompt (prefix)** before the input tokens, guiding model performance improvement **without major changes** to the main model.

3. **Training Tools**  
   - **Hugging Face Transformers** library’s **Trainer API**  
   - **Deepspeed**, **Accelerate** for **distributed training** and memory optimization  
   - **PEFT library**: Apply LoRA and various other techniques easily

### 6.2 Considerations

- **Data Quality**: The **domain suitability** and **label quality** of the finetuning data are very important.  
- **Avoid Overfitting**: Simply increasing the learning rate or the number of epochs may cause the **model’s generative ability to degrade** or lead to **model bias**.  
- **Model Compatibility**: Some models have architectural constraints that limit finetuning, so you should consult **official documentation** or **community resources**.

---

By combining **LangChain**, **RAG**, **Vector DB**, and **LLM Finetuning**, you can:

1. Easily build an **LLM-based pipeline**,  
2. Strengthen the model’s responses with **accurate and rich knowledge**, and  
3. Create a **custom LLM** optimized for a **specific domain** or **work environment**.

In practical or research settings where LLMs are utilized:  
- Use **LangChain** to structure your **workflow**,  
- Enhance it with **RAG** for **real-time knowledge retrieval**,  
- Maximize **search performance** with a **Vector DB**,  
- And **LLM Finetuning** to adapt the model to **domain-specific** needs.  

This will enable you to build more **efficient** and **powerful** NLP solutions.

---

## Notebook Demonstration

In this notebook, we will explore the following **simple example code** to see how to use a local environment:

1. **Load a local Llama model (or similar) and perform simple inference**  
2. **Use LangChain + Vector DB to demonstrate RAG**  
3. (Simple version) **LLM Finetuning example**

```bash
pip install -r requirements.txt
```
Run this in your terminal to set up the environment.

Example: Loading a local LLM
Below is an example of downloading the Llama3.1-8b model from the Hugging Face model repository to your local machine and performing a simple inference.

On Apple Silicon (M1, M4, etc.), mps (Metal Performance Shaders) device support may be set automatically or require manual configuration.

⸻

Tokens and Tokenizer

In Natural Language Processing (NLP), tokenization is a very important preprocessing step that breaks text into specific units. The tool or library used in this process is the tokenizer. Let’s learn about this in more detail.
1.	What is a Token?

	A token is a small semantic unit obtained by splitting text (sentences, paragraphs, etc.). The way you split can vary, producing different types of token units.
	
	1.1 Word-level
		•A simple method that splits based on whitespace or punctuation.
		•Example: “나는 학교에 간다.” → [“나는”, “학교에”, “간다.”]
	
	1.2 Morpheme-level (used in Korean)
		•Often used for Korean.
		•Example: “나는 학교에 간다.” → [“나”, “는”, “학교”, “에”, “가”, “ᄂ다”, “.”]
	
	1.3 Subword-level
		•Uses algorithms like BPE, WordPiece, SentencePiece to split into smaller-than-word units.
		•Example: “unhappy” → [“un”, “happy”]
		•Example: “unbelievable” → [“un”, “believable”]

2.	What is a Tokenizer?

	A tokenizer is a tool (or library) that splits text into tokens according to certain rules or algorithms.
	
	2.1 Rule-based Tokenizer
		•Splits using predefined rules such as whitespace, punctuation, or patterns (regex).
		•Example: split(), simple regex-based splitting.
	
	2.2 Trained Tokenizer
		•Automatically creates a token vocabulary by training on a corpus of data, learning the rules.
		•BPE (Byte-Pair Encoding), WordPiece, SentencePiece are representative examples.
		•Also used in large language models (e.g., BERT, GPT).

3.	Why is Tokenization Important?
4.	Improved Accuracy
	•Splitting text into correct units increases the accuracy of subsequent tasks like morphological analysis or POS tagging.
5.	Vocabulary Management
	•If you only split by words, your vocabulary can become huge, but with subword methods, you can handle rare words and neologisms efficiently.
6.	Maximize Model Performance
	•Poor tokenization can make it difficult to train the model or degrade performance at inference time.
	•Models like BERT or GPT are trained on consistent tokenization rules.
7.	Considerations for Korean Tokenization
8.	The need for morphological analysis
	•Korean has various postpositions and endings, making it difficult to tokenize by whitespace alone.
9.	Various elements within a word
	•“학교에” → [“학교”, “에”]
	•“학교에서” → [“학교”, “에서”]
	•Must split postpositions, endings, etc. to get the desired accuracy.
10.	Irregular spacing
	•Because spacing is often not strictly followed, purely rule-based approaches have limitations.
11.	Representative Tokenizers

	•NLTK (English)
	•One of the most widely used NLP libraries in Python
	•Provides word tokenization, sentence tokenization, stopword removal, etc.
	•KoNLPy (Korean)
	•Integrates multiple morphological analyzers (Twitter, Kkma, Hannanum, etc.)
	•Specialized for Korean, offering morphological analysis, POS tagging, etc.
	•BPE, SentencePiece, WordPiece
	•Subword-based tokenizers
	•Used in large language models such as BERT, GPT, RoBERTa, etc.

6.	Summary

	•Token
	•The result of splitting text into small units (words, morphemes, subwords, etc.)
	•Tokenizer
	•A tool or algorithm for splitting text according to a specified criterion
	•Tokenization
	•A core preprocessing step in NLP. Good tokenization can significantly improve model performance and manage vocabulary effectively.
	•For Korean, morphological analysis or subword-based methods are commonly used.

In conclusion, proper tokenization can greatly improve data quality and model performance. It is important to choose a tokenizer suited to your NLP project and devise a tokenization strategy that reflects the linguistic characteristics of each language.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Example for a Llama model (In practice, you may need permission or a different model name in Hugging Face.)
# model_name = "meta-llama/Llama-3.1-8B"
model_name="../models/Llama-3.1-8b"  
# Check if the mps device is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map={"": device})

# Simple text generation example
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text:")
print(generated_text)

Using device: mps
...
Generated text:
Hello, how are you? Today we are going to talk about the new generation...

Explanation:
	•torch_dtype=torch.float16: float16 is optimal on Apple MPS.
	•device_map={"": device}: Automatically places the model on MPS (GPU) or CPU.
	•max_new_tokens=50: Limits response to 50 tokens.

Llama 3.1 8B + LangChain + Vector DB (RAG)

You can also integrate the Llama 3.1 8B model with LangChain and ChromaDB (a Vector DB) to perform Retrieval-Augmented Generation (RAG).

2.1 Install LangChain & ChromaDB

pip install langchain chromadb faiss-cpu sentence-transformers

2.2 RAG Code Example

# Required library imports
from transformers import pipeline  # <-- added here
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

# 1) Load embedding model
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# Sample documents
texts = [
    "LangChain is a framework for chaining LLMs.",
    "RAG stands for Retrieval-Augmented Generation.",
    "A Vector DB is a database for searching document embedding vectors."
]
documents = [Document(page_content=t) for t in texts]

# 2) Initialize a Chroma VectorStore
vectorstore = Chroma.from_documents(documents, embedding=embeddings, collection_name="example_collection")

# 3) Connect Llama 3.1 8B model to LangChain
generator_pipeline = pipeline(
    "text-generation",
    model=model,  # Llama 3.1 8B model
    tokenizer=tokenizer
)
llm = HuggingFacePipeline(pipeline=generator_pipeline)

# 4) Create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 5) Ask a question and generate RAG-based answer
query = "What is RAG?"
answer = qa_chain.run(query)
print(f"Q: {query}\nA: {answer}")

Output:

Q: What is RAG?
A: Retrieval-Augmented Generation...



⸻

What is an Embedding Model?

1) Concept

An Embedding Model is a model that converts data into fixed-size vectors.
It transforms various types of data (text, images, audio, etc.) into numerical vectors so that computers can understand and process them.

In other words, embedding is the process of converting high-dimensional data (words, sentences, documents, images, etc.) into dense vectors, and the model that generates these vectors is called an Embedding Model.

⸻

2) Why are embeddings needed?

Because computers can only understand numbers, they cannot directly process natural language (NLP) or images.
Hence, Embedding Models are needed for the following reasons:
	•	Convert strings to numbers: Text must be converted into numerical vectors for machine learning models to understand.
	•	Similar data have similar vector values: Words with similar meanings end up with similar vector representations.
	•	Compress high-dimensional data into lower dimensions: Embeddings help reduce the dimensionality for more optimized computations.

For example, “cat” and “dog” are similar in meaning, so their embedding vectors are close to each other in the vector space.
Meanwhile, “cat” and “car” have little connection, so they lie far apart in the vector space.

⸻

3) Types of Embedding Models

Embedding Models are used to vectorize text, images, audio, etc. Here are some representative types:

(1) Text Embeddings (Word/Sentence Embeddings)
	•Models that convert words, sentences, or documents into vectors for NLP
	•Uses: Chatbots, search, recommendation systems, document classification, RAG

Representative Models:
	•Word2Vec: A leading model for word vectorization
	•GloVe: Vectors that reflect statistical co-occurrence of words
	•FastText: An improvement on Word2Vec (can embed subwords)
	•BERT Embedding: A powerful model that considers context when creating vectors
	•Sentence-BERT (SBERT): A model that creates embeddings at the sentence level
	•sentence-transformers/all-MiniLM-L6-v2: A lightweight, fast sentence embedding model (often used with LangChain, RAG)

Example (using sentence-transformers to generate text embedding vectors):

from sentence_transformers import SentenceTransformer

# Load an SBERT-based sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example sentences
sentences = ["Cats are cute.", "Dogs are loyal.", "Cars are fast."]

# Convert to embeddings (vectors)
embeddings = model.encode(sentences)

# Print the vector for the first sentence
print(embeddings[0])



⸻

(2) Image Embeddings
	•Convert image data into vectors for tasks like image similarity search, object recognition, etc.
	•Uses: Image retrieval systems, style recommendations, computer vision

Representative Models:
	•ResNet-50, EfficientNet, CLIP: For image classification and feature extraction
	•DINOv2: An image embedding model recently introduced by Meta
	•OpenAI CLIP: Maps text and images into the same vector space (e.g. “dog photo” → close to dog-image vectors)

Example (using CLIP to generate image embedding vectors):

import torch
import clip
from PIL import Image

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load and transform the image
image = preprocess(Image.open("dog.jpg")).unsqueeze(0).to(device)

# Extract embedding vectors
with torch.no_grad():
    image_features = model.encode_image(image)

print(image_features.shape)  # e.g., [1, 512]



⸻

(3) Audio Embeddings
	•Convert audio data into vectors for tasks like speech recognition, emotion analysis, etc.
	•Uses: Speech-to-text, emotion analysis, noise filtering, music recommendation systems

Representative Models:
	•MFCC (Mel-Frequency Cepstral Coefficients): A method to extract feature vectors from audio signals
	•wav2vec 2.0 (by Facebook): Converts audio to vectors and can transcribe text
	•Whisper (by OpenAI): A multi-language speech recognition and embedding model

Example (using librosa to generate audio embedding vectors):

import librosa

# Load an audio file
y, sr = librosa.load("speech.wav")

# Extract MFCC feature vectors
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Print the MFCC vector for the first frame
print(mfccs[:, 0])



⸻

4) Applications of Embedding Vectors

Using embedding models, you can achieve strong performance in tasks such as document search, chatbots, recommendation systems, and RAG models.
	•Document Retrieval System
	•Convert user queries into vectors and search for the most similar documents.
	•Example: LangChain + ChromaDB for RAG (Retrieval-Augmented Generation)
	•Chatbots and LLM Applications
	•During chatbot response generation, utilize embedding-based searches.
	•Example: Convert user input into embeddings and retrieve the most relevant answers from a DB.
	•Recommendation Systems
	•Platforms like Netflix or Spotify use embedding vectors for user preference and content.
	•Example: Embedding vectors for movies/music the user has consumed, then recommend similar content.
	•Healthcare and Bioinformatics
	•Vectorize gene data, medical publications, protein structures, etc. for analysis.
	•Example: Drug discovery, genome analysis

⸻

5) Conclusion

Embedding models convert text, images, audio into vector representations, enabling similarity search, chatbots, recommendation systems, and more.
Transformer-based embedding models (e.g., BERT, CLIP, wav2vec 2.0, etc.) are becoming more sophisticated, enabling richer semantic representations.

🚀 By combining LangChain + Vector DB (Chroma, FAISS) + LLM (RAG), you can build even more powerful AI applications!

⸻

3) Llama 3.1 8B Model Finetuning (LoRA Application)

To finetune the Llama 3.1 8B model using LoRA (Low-Rank Adaptation), you can use PEFT (Parameter-Efficient Fine-Tuning).

3.1 Install LoRA

pip install peft bitsandbytes datasets

3.2 LoRA Code Example

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# Load model and tokenizer
model_name = "../models/Llama-3.1-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # ✅ using bfloat16 instead of float16
    device_map="auto"
).to(device)  # explicitly move model to device

# LoRA Configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"]  # Adjust core weights of Llama
)

# Apply LoRA to the model
lora_model = get_peft_model(model, lora_config)

# ✅ Prepare input data
train_texts = [
    "Question: What is the Llama model?\nAnswer: It is a large language model by Meta.",
    "Question: What is RAG?\nAnswer: It is a retrieval-based generation model."
]

# ✅ Modified tokenization function
def tokenize_fn(text):
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=64)
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"]  # GPT-style models use labels = input_ids
    }

# ✅ Convert to Hugging Face Dataset
train_dataset = Dataset.from_dict({"text": train_texts})
train_dataset = train_dataset.map(lambda x: tokenize_fn(x["text"]), batched=True, remove_columns=["text"])

# ✅ TrainingArguments (disable removal of unused columns)
training_args = TrainingArguments(
    output_dir="finetuned-llama3",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    save_steps=10,
    logging_steps=5,
    remove_unused_columns=False,  # <--- Important
)

# ✅ Trainer setup
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
)

# ✅ Start training
trainer.train()

# ✅ Save the trained model
trainer.save_model("finetuned-llama3")

# Save the LoRA adapter checkpoint
lora_model.save_pretrained("finetuned-llama3")
# Optionally save the tokenizer too
tokenizer.save_pretrained("finetuned-llama3")
print("Adapter and tokenizer saved successfully!")

print("Finetuning Complete!")

Using device: mps
...
Finetuning Complete!

Why LoRA?
	•Typical training for Llama 3.1 8B might require over 100GB of RAM.
	•By adjusting only certain weights (Q, V projections) using LoRA, you can reduce memory usage.
	•Can run even on an M4 Pro with 64GB RAM.

4) Summary
	•Llama 3.1 8B model usage
	1.Use Hugging Face Transformers
		•Use mps (Apple Metal)
		•Use float16 for memory optimization
		•Integrate with LangChain + Vector DB (RAG)
	2.Use ChromaDB to store documents
		•Build retrieval-answer systems with LangChain
		•Finetune Llama 3.1 8B (LoRA)
	3.Apply PEFT (LoRA) for lightweight training
		•Possible on Apple Silicon

How to Use the Trained Model

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the base model from its original checkpoint
base_model_name = "../models/Llama-3.1-8b"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Now load the adapter from your local directory
model = PeftModel.from_pretrained(base_model, "finetuned-llama3", local_files_only=True)

# Load the tokenizer (either from your adapter folder or the base model)
tokenizer = AutoTokenizer.from_pretrained("finetuned-llama3")

prompt = "Question: What is the Llama model?\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Response:")
print(generated_text)

Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Generated Response:
Question: What is the Llama model?
Answer: B

It seems the model did not learn well—likely because the training dataset was very small. If you increase the dataset size, results will likely improve.

For example, you could try a Korean food dataset from:
https://huggingface.co/datasets/SGTCho/korean_food

And follow the steps here:
https://github.com/SGT-Cho/LLM/tree/main/Finetuning




# LLM, Llama, Deepseek R1, LangChain, RAG, Vector DB, LLM Finetuning

본 노트북은 **로컬 환경에서 LLM(Local Large Language Model)을 활용하여 실험**하는 방법을 데모하기 위해 작성되었습니다.  

---

## 1. LLM (Large Language Model)

**LLM**은 방대한 파라미터를 갖춘 언어 모델로, **자연어 처리(NLP)** 전반에 걸쳐 **높은 성능**을 보입니다.  
- 예: **GPT-4**, **Llama3**, **Phi4**, **Deepseek R1** 등

---

## 2.1 Llama

**Llama**는 Meta에서 공개한 **대규모 언어 모델** 계열입니다.  
- 사전에 학습된 **Llama 모델**을 Hugging Face Transformers를 통해 **로컬에서 실행** 가능합니다.
  
## 2.2 Deepseek R1  
  
**Deepseek R1**은 강화 학습 기반 추론 모델입니다.
- DeepSeek 는 중국 헤지 펀드인 하이-플라이어 (High-Flyer) 가 2023년에 설립한 중국의 인공지능 회사입니다. DeepSeek LLM, DeepSeek Coder, DeepSeek Math 등 다양한 대형 언어 모델을 오픈소스로 공개해 왔습니다.  
특히, 최근 공개된 DeepSeek-V3는 Claude 3.5 Sonnet 및 Gemini 1.5 Pro와 견줄 만한 성능으로 주목받고 있습니다.  

- 이번에는 강화 학습 (RL, Reinforcement Learning) 을 통해 추론 능력을 극대화한 새로운 모델, DeepSeek-R1 과 DeepSeek-R1-Zero 를 공개하였습니다. 이 두 모델은 2025년 1월 20일에 오픈소스로 공개되었습니다.
---

## 3. LangChain

**LangChain**은 파이썬 기반 라이브러리로, **대규모 언어 모델(LLM)을 활용한 애플리케이션**을 **체계적으로 개발**하고 **확장**하기 위한 다양한 기능을 제공합니다.

- **Prompt 관리**  
  - 여러 프롬프트를 효율적으로 관리하고 재사용할 수 있도록 **체계화**합니다.

- **체이닝(Chaining)**  
  - 여러 스텝의 LLM 작업을 **순차적으로 연결**해 **파이프라인** 형태로 구성할 수 있습니다.  
  - 예) 요약 → 질의응답 → 감성 분석 등, 다단계 워크플로우를 간단히 구현

- **에이전트(Agent)**  
  - 특정 툴(API, DB 등)에 접근해 **동적으로 작업**을 수행하고, **추론(Reasoning)** 기반 의사결정을 내려 **문제 해결**을 돕는 모듈

- **메모리(Memory)**  
  - 모델이 **이전 대화나 컨텍스트**를 **기억**할 수 있도록 관리하는 기능

- **툴 연동**  
  - 데이터베이스, 서드파티 API, 웹 검색 등 다양한 **외부 툴**과 손쉽게 연동해 **풍부한 기능**을 제공

이 모든 기능을 **통합적으로** 사용할 수 있어, **LLM을 활용한 애플리케이션**을 빠르게 **프로토타이핑**하고, **생산 환경**으로 확장하기 쉽습니다.

---

## 4. RAG (Retrieval-Augmented Generation)

**RAG(Retrieval-Augmented Generation)**는 대규모 언어 모델이 기존에 학습된 **파라미터**에만 의존하지 않고, 모델 **외부**의 지식 베이스(문서, 데이터베이스, 웹 검색 결과 등)에서 **실시간**으로 정보를 검색해 활용하는 방식입니다.

1. **Retrieve (검색)**  
   - 사용자의 질의(또는 대화 맥락)에 대응되는 **관련 문서나 정보**를 찾기 위해, **Vector DB** 등의 검색 엔진을 사용  
   - **임베딩 벡터** 기반의 유사도 검색을 통해, 모델이 필요한 정보를 **빠르게 획득**

2. **Generate (생성)**  
   - 검색된 문서를 토대로 **LLM**이 **응답**을 생성  
   - 모델은 **문서의 구체적인 내용**을 바탕으로, **사실적으로 풍부**하고 **정확도 높은** 텍스트를 반환

**장점**  
- **정확도 향상**: 최신 정보나 모델이 학습하지 못한 지식을 활용 가능  
- **메모리 제한 극복**: 모든 지식을 모델 파라미터에 담지 않아도 되므로, 모델 크기를 **효율적으로 유지**할 수 있음  
- **유연성**: 다양한 형태의 데이터(텍스트, 이미지 설명, DB 내용 등)와 결합 가능

---

## 5. Vector DB

**Vector DB**는 텍스트나 이미지를 **임베딩 벡터**로 변환하여 저장하고, 이와 **유사도가 높은** 벡터(문서, 이미지 등)를 **빠르게 검색**하기 위한 **특화된 데이터베이스**입니다.

- **주요 기능**  
  1. **벡터 삽입**  
     - 사전 학습된 모델(예: SentenceTransformer, BERT 등)로 **텍스트/이미지 → 벡터**로 변환하여 저장  
  2. **유사도 검색(ANN Search)**  
     - Approximate Nearest Neighbor Search 기법을 활용해, 대규모 벡터 집합 내에서 **비슷한 벡터(문서)**를 **효율적으로** 찾음  
  3. **확장성**  
     - 데이터가 매우 많아도 **확장(Scaling)**과 분산 처리를 통해 **빠른 검색 속도**를 유지

- **대표적인 Vector DB 예시**  
  - **FAISS**: 메타(구 페이스북)에서 개발한 벡터 유사도 검색 라이브러리  
  - **Chroma**: 개인 프로젝트부터 대규모 서비스까지 쉽게 확장 가능한 벡터 DB  
  - **Milvus**: 고성능, 대규모 벡터 검색 엔진  
  - **Pinecone**: 클라우드 기반의 완전관리형 벡터 DB 서비스

**활용 사례**  
- **RAG**(Retrieval-Augmented Generation)에서 **문서 검색**  
- **유사도 기반 추천** 시스템  
- 이미지 검색, 음성 검색 등 **멀티모달 검색**

---

## 6. LLM Finetuning

**LLM Finetuning**은 사전 학습된 대규모 언어 모델(예: GPT, BERT 등)을 특정 태스크나 도메인에 **맞춤화**하기 위해 추가 학습하는 과정입니다.  
- 기존 파라미터를 재활용하면서, **특정 데이터**에 대한 모델 성능을 **크게 개선**할 수 있습니다.

### 6.1 파인튜닝 방법

1. **전체 파라미터 업데이트(Full Finetuning)**  
   - 모델의 모든 파라미터를 대상으로, **에포크(epoch)** 단위로 **최적화(Optimization)** 진행  
   - 데이터와 계산 자원이 풍부한 경우에 사용

2. **PEFT(LoRA, Prefix Tuning 등)**  
   - **부분 파라미터만 업데이트**하거나, **저비용**으로 추가 학습하는 방식  
   - **LoRA(Low-Rank Adaptation)**: 모델 내부의 특정 가중치 행렬을 **저랭크(축소된 차원)** 형태로 학습해, **적은 메모리**로도 파인튜닝 효과를 얻을 수 있음  
   - **Prefix Tuning**: 입력 토큰 앞에 **가상 프롬프트(prefix)**를 추가 학습해, **모델 본체는 크게 건드리지 않고** 성능 향상을 유도

3. **훈련 도구**  
   - **Hugging Face Transformers** 라이브러리의 **Trainer API**  
   - **Deepspeed**, **Accelerate** 등 **분산 훈련**, 메모리 최적화 라이브러리  
   - **PEFT 라이브러리**: LoRA 등 다양한 기법을 간단하게 적용

### 6.2 주의사항

- **데이터 품질**: 파인튜닝에 사용되는 데이터의 **도메인 적합성**과 **레이블 품질**이 매우 중요  
- **오버피팅 방지**: 무작정 학습률이나 에포크를 높이면, **생성 능력이 단순화**되거나 **모델 편향**이 발생할 수 있음  
- **모델 호환성**: 일부 모델은 아키텍처 특성상 파인튜닝이 제한될 수 있으므로, **공식 문서**나 **커뮤니티 정보**를 확인

---

이러한 **LangChain**, **RAG**, **Vector DB**, **LLM Finetuning** 기법들을 조합하면,

1. **LLM 활용 파이프라인**을 손쉽게 구성할 수 있고,  
2. **정확하고 풍부한 지식**을 기반으로 모델의 응답을 **강화**하며,  
3. **특정 도메인**이나 **업무 환경**에 최적화된 **커스텀 LLM**을 구축할 수 있습니다.

실무나 연구에서 LLM을 활용할 때,  
- **LangChain**으로 **워크플로우**를 체계화하고  
- **RAG**를 통해 **실시간 지식 검색**을 더하며  
- **Vector DB**로 **검색 성능**을 극대화하고  
- **LLM Finetuning**으로 **도메인 특화 모델**을 만들면,  
더욱 **효율적**이고 **강력**한 NLP 솔루션을 구현할 수 있을 것입니다.

---

## 노트북 시연 내용

이 노트북에서는 **간단한 예시 코드**를 통해 로컬 환경에서 다음을 살펴봅니다:

1. **로컬 Llama 모델(또는 유사 모델) 로드 및 간단한 추론**  
2. **LangChain + Vector DB를 이용한 RAG 예시**  
3. (**간단 버전**) **LLM 파인튜닝 예시**

pip install -r requirements.txt
  
터미널에서 실행하여 환경 세팅

예시: 로컬 LLM 로드
아래 예시는 Hugging Face 모델 저장소로부터 Llama3.1-8b를 로컬에 다운로드받고,
간단 추론을 수행하는 코드 예시입니다.

Apple 실리콘(M1, M4 등)에서는 PyTorch `mps` 디바이스가 자동으로 잡히거나 수동 설정이 필요할 수 있습니다.


---
토큰(Token)과 토크나이저(Tokenizer)

자연어 처리(NLP)에서 **토큰화(Tokenization)**는 텍스트를 특정 단위로 분할하는 매우 중요한 전처리 과정입니다. 그 과정에서 활용되는 **토크나이저(Tokenizer)**에 대해 자세히 알아보겠습니다.

1. 토큰(Token)이란?

**토큰(Token)**은 텍스트(문장, 문단 등)를 작은 의미 단위로 나눈 결과물입니다. 어떤 기준으로 나누느냐에 따라 다양한 토큰 단위를 얻을 수 있습니다.

1.1 단어 단위
	•	공백(whitespace)이나 구두점(punctuation)을 기준으로 분리하는 간단한 방식
	•	예) “나는 학교에 간다.” → ["나는", "학교에", "간다."]

1.2 형태소 단위
	•	한국어에서 자주 사용하는 방식
	•	예) “나는 학교에 간다.” → ["나", "는", "학교", "에", "가", "ᄂ다", "."]

1.3 서브워드(Subword) 단위
	•	BPE, WordPiece, SentencePiece 등의 알고리즘을 사용하여 단어보다 작은 단위로 나눔
	•	예) “unhappy” → ["un", "happy"]
	•	예) “unbelievable” → ["un", "believable"]

2. 토크나이저(Tokenizer)란?

**토크나이저(Tokenizer)**는 텍스트를 특정 규칙 혹은 알고리즘을 사용해 토큰으로 분할하는 도구(또는 라이브러리)입니다.

2.1 규칙 기반 토크나이저
	•	공백, 구두점, 특정 패턴(정규 표현식) 등을 기준으로 미리 정해진 규칙에 따라 분리
	•	예) split(), 정규식(Regex)을 통한 단순 분할

2.2 학습 기반 토크나이저
	•	훈련 데이터(코퍼스)에 맞춰 토큰 사전을 자동 생성하고, 해당 규칙을 학습
	•	BPE(Byte-Pair Encoding), WordPiece, SentencePiece 등이 대표적
	•	대규모 언어 모델(예: BERT, GPT 등)에서도 사용

3. 토큰화 과정이 중요한 이유
	1.	정확도 향상
	•	올바른 단위로 텍스트를 나누어야 형태소 분석, 품사 태깅 등 후속 작업의 정확도가 높아집니다.
	2.	어휘 사전 관리
	•	단어 단위로만 나누면 어휘 사전이 너무 커질 수 있으나, 서브워드 방식을 사용하면 희귀 단어나 신조어도 효율적으로 처리할 수 있습니다.
	3.	모델 성능 극대화
	•	토큰화가 잘못되면 모델이 학습에 어려움을 겪거나, 추론 시 성능이 떨어집니다.
	•	BERT나 GPT 같은 모델들도 일관된 토크나이징 규칙을 기반으로 학습됩니다.

4. 한국어 토크나이징 시 고려사항
	1.	형태소 분석의 필요성
	•	한국어는 조사, 어미 변화 등이 다양해 단순 공백으로만 토큰화하기 어렵습니다.
	2.	어절 내부의 다양한 요소
	•	“학교에” → ["학교", "에"]
	•	“학교에서” → ["학교", "에서"]
	•	조사와 어간 등을 분리해줘야 원하는 정확도를 얻을 수 있습니다.
	3.	불규칙한 띄어쓰기
	•	띄어쓰기를 정확히 지키지 않는 경우가 많으므로 규칙 기반만으로는 처리에 한계가 있습니다.

5. 대표적인 토크나이저 예시
	•	NLTK (영어)
	•	파이썬에서 가장 많이 사용되는 NLP 라이브러리 중 하나
	•	단어 토큰화, 문장 토큰화, 스톱워드 제거 등 다양한 기능 제공
	•	KoNLPy (한국어)
	•	트위터(Twitter), 꼬꼬마(Kkma), 한나눔(Hannanum) 등 여러 형태소 분석기 연동 가능
	•	형태소 분석, 품사 태깅 등 한국어 전용 처리 기능
	•	BPE, SentencePiece, WordPiece
	•	서브워드(subword) 기반 토크나이저
	•	대규모 언어 모델(예: BERT, GPT, RoBERTa 등)에서 사용

6. 정리
	•	토큰(Token)
	•	텍스트를 작은 단위(단어, 형태소, 서브워드 등)로 분할한 결과물
	•	토크나이저(Tokenizer)
	•	텍스트를 원하는 기준으로 분할하기 위한 도구 또는 알고리즘
	•	토큰화(Tokenization)
	•	자연어 처리의 핵심 전처리 단계이며, 올바른 토큰화를 통해 모델의 성능을 높이고 어휘 사전을 효율적으로 관리할 수 있음
	•	한국어는 조사, 어미 변화 등이 복잡하므로 형태소 분석 또는 서브워드 방식을 고려하는 것이 일반적

결론적으로, 적절한 토크나이징을 통해 데이터 품질과 모델 성능을 크게 향상시킬 수 있습니다. NLP 프로젝트에 맞는 토크나이저를 선택하고, 언어별 특성을 반영한 토큰화 전략을 수립하는 것이 중요합니다.


```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Llama 모델 예시 (실제 사용 시, Hugging Face에서 접근 권한이 필요하거나 모델 이름이 다를 수 있습니다.)
#model_name = "meta-llama/Llama-3.1-8B"
model_name="../models/Llama-3.1-8b"  
# mps 디바이스 사용 가능 여부 확인
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map={"": device})

# 간단한 텍스트 생성 예시
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text:")
print(generated_text)

```

    Using device: mps



    Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Generated text:
    Hello, how are you? Today we are going to talk about the new generation of smart watches, which are very popular in the world of technology and are very useful for our daily life.
    The smart watch is a device that is in charge of tracking the physical activity of its user


설명:

- torch_dtype=torch.float16: Apple MPS에서는 float16이 최적입니다.
- device_map={"": device}: 모델을 자동으로 MPS(GPU) 또는 CPU에 배치합니다.
- max_new_tokens=50: 응답 길이를 50개 토큰으로 제한합니다.

Llama 3.1 8B 모델 + LangChain + Vector DB(RAG) 적용
Llama 3.1 8B 모델을 LangChain 및 ChromaDB(Vector DB)와 함께 사용하여 Retrieval-Augmented Generation(RAG) 을 수행할 수도 있습니다.

2.1 LangChain & ChromaDB 설치

```bash
pip install langchain chromadb faiss-cpu sentence-transformers
```

2.2 RAG 적용 코드


```python
# 필요한 라이브러리 임포트
from transformers import pipeline  # <-- 여기에 추가
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

# 1) 임베딩 모델 로드
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# 예제 문서
texts = [
    "LangChain은 LLM을 체인으로 연결하는 프레임워크입니다.",
    "RAG는 Retrieval-Augmented Generation의 약어입니다.",
    "Vector DB는 문서 임베딩 벡터를 검색하기 위한 데이터베이스입니다."
]
documents = [Document(page_content=t) for t in texts]

# 2) Chroma VectorStore 초기화
vectorstore = Chroma.from_documents(documents, embedding=embeddings, collection_name="example_collection")

# 3) Llama 3.1 8B 모델을 LangChain에 연결
generator_pipeline = pipeline(
    "text-generation",
    model=model,  # Llama 3.1 8B 모델
    tokenizer=tokenizer
)
llm = HuggingFacePipeline(pipeline=generator_pipeline)

# 4) RetrievalQA 체인 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 5) 질문 입력 및 RAG 기반 답변 생성
query = "RAG는 무엇인가요?"
answer = qa_chain.run(query)
print(f"Q: {query}\nA: {answer}")

```

    Device set to use mps
    /var/folders/z8/94fh0xbx5cv85y4f8dm4_nch0000gn/T/ipykernel_1491/842959494.py:30: LangChainDeprecationWarning: The class `HuggingFacePipeline` was deprecated in LangChain 0.0.37 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-huggingface package and should be used instead. To use it run `pip install -U :class:`~langchain-huggingface` and import as `from :class:`~langchain_huggingface import HuggingFacePipeline``.
      llm = HuggingFacePipeline(pipeline=generator_pipeline)
    /var/folders/z8/94fh0xbx5cv85y4f8dm4_nch0000gn/T/ipykernel_1491/842959494.py:41: LangChainDeprecationWarning: The method `Chain.run` was deprecated in langchain 0.1.0 and will be removed in 1.0. Use :meth:`~invoke` instead.
      answer = qa_chain.run(query)
    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Q: RAG는 무엇인가요?
    A: Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    RAG는 Retrieval-Augmented Generation의 약어입니다.
    
    RAG는 Retrieval-Augmented Generation의 약어입니다.
    
    RAG는 Retrieval-Augmented Generation의 약어입니다.
    
    LangChain은 LLM을 체인으로 연결하는 프레임워크입니다.
    
    Question: RAG는 무엇인가요?
    Helpful Answer: Retrieval-Augmented Generation의 약어입니다.


## **임베딩 모델(Embedding Model)이란?**

### 1️⃣ **개념**
임베딩 모델(Embedding Model)이란, 데이터를 **고정된 크기의 벡터(vector) 형태로 변환하는 모델**을 의미합니다.  
텍스트, 이미지, 오디오 등 다양한 데이터 유형을 **수치 벡터로 변환**하여 컴퓨터가 이해하고 연산할 수 있도록 만들어 줍니다.

즉, **임베딩(Embedding)**은 고차원 데이터(예: 단어, 문장, 문서, 이미지 등)를 **밀집 벡터(Dense Vector)**로 변환하는 과정이며,  
이 벡터를 생성하는 모델을 **임베딩 모델(Embedding Model)**이라고 합니다.

---

### 2️⃣ **왜 임베딩이 필요한가?**
컴퓨터는 숫자만 이해할 수 있으므로, 자연어(NLP)나 이미지 데이터를 직접 처리할 수 없습니다.  
따라서, 다음과 같은 이유로 **임베딩 모델**이 필요합니다.

✅ **문자열을 숫자로 변환**: 텍스트를 숫자 벡터로 변환해야 머신러닝 모델이 이해할 수 있음  
✅ **유사한 데이터끼리 가까운 벡터 값을 가짐**: 의미적으로 비슷한 단어는 유사한 벡터로 변환됨  
✅ **고차원 데이터를 저차원으로 압축**: 차원이 높은 데이터를 낮은 차원의 벡터로 변환하여 연산을 최적화  

> 예를 들어, "고양이(cat)"와 "강아지(dog)"는 **비슷한 의미**를 가지므로, 임베딩 벡터 공간에서 서로 가까운 위치에 있음.  
> 반면, "고양이(cat)"와 "자동차(car)"는 관련성이 적으므로 벡터 공간에서 멀리 떨어져 있음.

---

### 3️⃣ **임베딩 모델의 종류**
임베딩 모델은 **텍스트, 이미지, 오디오** 등 다양한 데이터를 벡터화하는 데 사용됩니다.  
대표적인 임베딩 모델 종류는 다음과 같습니다.

#### 🔹 **(1) 텍스트 임베딩 (Word/Sentence Embedding)**
- 자연어 처리(NLP)에서 단어, 문장, 문서를 벡터로 변환하는 모델
- **사용처**: 챗봇, 검색, 추천 시스템, 문서 분류, RAG(Retrieval-Augmented Generation)

📌 **대표적인 모델**
- `Word2Vec`: 단어를 벡터화하는 대표적인 모델
- `GloVe(Global Vectors for Word Representation)`: 단어 간의 통계적 연관성을 반영한 벡터
- `FastText`: Word2Vec 개선 버전 (부분 단어까지 임베딩 가능)
- `BERT Embedding`: 문맥을 고려하여 단어/문장을 벡터로 변환하는 강력한 모델
- `Sentence-BERT (SBERT)`: 문장 단위 임베딩을 생성하는 모델
- `sentence-transformers/all-MiniLM-L6-v2`: 가볍고 빠른 문장 임베딩 모델 (LangChain, RAG에서 많이 사용됨)

✅ **예제**: `sentence-transformers`를 사용하여 텍스트 임베딩 벡터 생성  
```python
from sentence_transformers import SentenceTransformer

# SBERT 기반 문장 임베딩 모델 로드
model = SentenceTransformer("all-MiniLM-L6-v2")

# 예제 문장
sentences = ["고양이는 귀엽다.", "강아지는 충성스럽다.", "자동차는 빠르다."]

# 임베딩 변환 (벡터 생성)
embeddings = model.encode(sentences)

# 출력 (첫 번째 문장의 벡터 값)
print(embeddings[0])
```

---

#### 🔹 **(2) 이미지 임베딩 (Image Embedding)**
- 이미지 데이터를 벡터로 변환하여 비슷한 이미지 검색, 객체 인식 등에 활용
- **사용처**: 이미지 검색 시스템, 스타일 추천, 컴퓨터 비전

📌 **대표적인 모델**
- `ResNet-50`, `EfficientNet`, `CLIP`: 이미지 분류 및 특징 벡터 추출
- `DINOv2`: 최근 Meta에서 발표한 이미지 임베딩 모델
- `OpenAI CLIP`: 텍스트와 이미지를 동일한 벡터 공간에 매핑하는 모델 (예: "강아지 사진" → 강아지 이미지 벡터와 유사한 위치)

✅ **예제**: `CLIP`을 사용하여 이미지 임베딩 벡터 생성
```python
import torch
import clip
from PIL import Image

# CLIP 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 이미지 로드 및 변환
image = preprocess(Image.open("dog.jpg")).unsqueeze(0).to(device)

# 임베딩 벡터 추출
with torch.no_grad():
    image_features = model.encode_image(image)

print(image_features.shape)  # [1, 512] 크기의 벡터 출력
```

---

#### 🔹 **(3) 오디오 임베딩 (Audio Embedding)**
- 오디오 데이터를 벡터로 변환하여 음성 인식, 감정 분석 등에 활용
- **사용처**: 음성 인식(Speech-to-Text), 감정 분석, 노이즈 필터링, 음악 추천 시스템

📌 **대표적인 모델**
- `MFCC (Mel-Frequency Cepstral Coefficients)`: 음성 데이터를 특징 벡터로 변환
- `wav2vec 2.0 (by Facebook)`: 음성을 벡터화하고 텍스트로 변환하는 모델
- `Whisper (by OpenAI)`: 다국어 음성 인식 및 임베딩 모델

✅ **예제**: `librosa`를 사용하여 오디오 임베딩 벡터 생성
```python
import librosa

# 오디오 파일 로드
y, sr = librosa.load("speech.wav")

# MFCC 특징 벡터 추출
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# 출력 (첫 번째 프레임의 MFCC 벡터)
print(mfccs[:, 0])
```

---

### 4️⃣ **임베딩 벡터의 활용**
임베딩 모델을 활용하면 **문서 검색, 챗봇, 추천 시스템, RAG 모델** 등에서 강력한 성능을 발휘할 수 있습니다.

✅ **문서 검색 시스템**  
- 사용자가 입력한 질문을 벡터로 변환하여 **가장 유사한 문서**를 검색  
- 예: **LangChain + ChromaDB** 를 활용한 **RAG (Retrieval-Augmented Generation)**  

✅ **챗봇 및 LLM 응용**  
- 챗봇이 답변을 생성할 때 **임베딩 벡터 기반 검색**을 수행  
- 예: 사용자의 입력을 임베딩 후, **가장 관련성이 높은 답변을 DB에서 검색하여 응답**  

✅ **추천 시스템**  
- Netflix, Spotify 같은 추천 시스템에서 **사용자의 취향을 임베딩 벡터로 변환**  
- 예: 사용자가 시청한 영화/음악의 임베딩 벡터를 활용하여 **비슷한 콘텐츠 추천**  

✅ **의료 및 생물정보학**  
- 유전자 데이터, 의학 논문, 단백질 구조 데이터를 벡터화하여 분석  
- 예: 신약 개발, 유전체 분석  

---

### 5️⃣ **결론**
임베딩 모델은 데이터(텍스트, 이미지, 오디오)를 **벡터 표현으로 변환하여** 유사도 검색, 챗봇, 추천 시스템 등 다양한 분야에서 활용됩니다.  
최근에는 **Transformer 기반의 강력한 임베딩 모델(BERT, CLIP, wav2vec 2.0 등)이 등장**하면서, **더 정교한 의미 표현이 가능**해졌습니다.

> 🚀 **LangChain + Vector DB (Chroma, FAISS) + LLM(RAG)** 을 활용하면 더욱 강력한 AI 응용 시스템을 구축할 수 있습니다!

---

### 3)Llama 3.1 8B 모델 Finetuning (LoRA 적용)
Llama 3.1 8B 모델을 LoRA (Low-Rank Adaptation) 방식으로 파인튜닝하려면 PEFT (Parameter-Efficient Fine-Tuning) 를 사용하면 됩니다.

3.1 LoRA 설치

```bash
pip install peft bitsandbytes datasets
```

3.2 LoRA 적용 코드


```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# 모델 및 토크나이저 로드
model_name = "../models/Llama-3.1-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # ✅ float16 대신 bfloat16 사용
    device_map="auto"
).to(device)  # ✅ 명시적으로 디바이스로 이동

# LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"]  # Llama 모델의 핵심 가중치 조정
)

# LoRA 모델 적용
lora_model = get_peft_model(model, lora_config)

# ✅ 입력 데이터 준비
train_texts = [
    "Question: Llama 모델은?\nAnswer: Meta의 대형 언어 모델입니다.",
    "Question: RAG란?\nAnswer: 검색 기반 생성 모델입니다."
]

# ✅ 토큰화 함수 수정
def tokenize_fn(text):
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=64)
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"]  # GPT 모델은 labels=input_ids 사용
    }

# ✅ 데이터셋 변환 (Hugging Face Dataset)
train_dataset = Dataset.from_dict({"text": train_texts})
train_dataset = train_dataset.map(lambda x: tokenize_fn(x["text"]), batched=True, remove_columns=["text"])

# ✅ TrainingArguments 설정 (disable removal of unused columns)
training_args = TrainingArguments(
    output_dir="finetuned-llama3",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    save_steps=10,
    logging_steps=5,
    remove_unused_columns=False,  # <--- Added line to fix the error
)

# ✅ Trainer 설정
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
)

# ✅ 학습 시작
trainer.train()

# ✅ 학습 완료 후 모델 저장
trainer.save_model("finetuned-llama3")

# Save the adapter checkpoint (this will save adapter_config.json, adapter weights, etc.)
lora_model.save_pretrained("finetuned-llama3")
# Optionally, save the tokenizer as well.
tokenizer.save_pretrained("finetuned-llama3")
print("Adapter and tokenizer saved successfully!")


print("Finetuning Complete!")

```

    Using device: mps



    Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]


    'NoneType' object has no attribute 'cadam32bit_grad_fp32'


    /opt/anaconda3/envs/guide/lib/python3.9/site-packages/bitsandbytes/cextension.py:34: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
      warn("The installed version of bitsandbytes was compiled without GPU support. "



    Map:   0%|          | 0/2 [00:00<?, ? examples/s]




    <div>

      <progress value='2' max='2' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [2/2 00:00, Epoch 1/1]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table><p>


    Adapter and tokenizer saved successfully!
    Finetuning Complete!


LoRA를 활용하는 이유:

- 일반적인 Llama 3.1 8B 모델 학습은 RAM 100GB 이상 필요
- LoRA를 사용하면 특정 가중치(Q, V 프로젝션)만 조정하여 메모리 사용량을 줄임
- M4 Pro(64GB RAM) 환경에서도 실행 가능

### 4) 요약
- Llama 3.1 8B 모델 실행

1. Hugging Face transformers 로드
    - mps(Apple Metal Performance Shader) 사용
    - float16으로 메모리 최적화
    - LangChain + Vector DB 적용 (RAG)

2. ChromaDB로 문서 저장
    - LangChain을 사용하여 검색-응답 시스템 구축
    - Llama 3.1 8B Finetuning (LoRA)

3. PEFT(LoRA) 활용하여 가벼운 학습 진행
    - Apple Silicon 환경에서도 가능

### 학습된 모델 사용 방법


```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the base model from its original checkpoint
base_model_name = "../models/Llama-3.1-8b"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Now load the adapter from your local directory
model = PeftModel.from_pretrained(base_model, "finetuned-llama3", local_files_only=True)

# Load the tokenizer (you can load from your adapter folder if saved or from the base model)
tokenizer = AutoTokenizer.from_pretrained("finetuned-llama3")

prompt = "Question: Llama 모델은?\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Response:")
print(generated_text)

```


    Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]


    Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.


    Generated Response:
    Question: Llama 모델은?
    Answer: B


Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Generated Response:
Question: Llama 모델은?
Answer: B


  
잘 학습되진 않은 모습이다.
아마 너무 적은 크기의 데이터셋을 이용해서 그런 것 같다
크기를 늘려서 시도해보면 잘 될 것 같습니다

https://huggingface.co/datasets/SGTCho/korean_food
  
에서 한식 관련 데이터셋을 받은후

https://github.com/SGT-Cho/LLM/tree/main/Finetuning
  
따라해보면 될 것 같습니다


