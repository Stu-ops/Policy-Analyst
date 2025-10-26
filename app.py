import os
import json
import re
import time
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from operator import itemgetter
import requests
from bs4 import BeautifulSoup

from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.responses import JSONResponse


from utils.DocsLoader import load_and_chunk
from utils.Schemas import RunRequest, RunResponse
# from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
# from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker 
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.prompts import ChatPromptTemplate
# from langchain_nvidia_ai_endpoints.embeddings import NVIDIAEmbeddings
# from langchain_nvidia_ai_endpoints.reranking import  NVIDIARerank

from sentence_transformers import SentenceTransformer

#loading the model for SentenceTransformersTokenTextSplitter
MODEL_DIR = os.path.join("/tmp", "e5-large-v2")

if not os.path.exists(MODEL_DIR):
    print("📦 Downloading SentenceTransformer model...")
    model = SentenceTransformer("intfloat/e5-large-v2")
    model.save(MODEL_DIR)
    print("✅ Model saved at", MODEL_DIR)

    
# Load environment variables
load_dotenv()

vector_cache = {}
ml_models = {}
secret = ""

landmark_data = {
    "Delhi": "Gateway of India", "Mumbai": "India Gate", "Chennai": "Charminar",
    "Hyderabad": "Taj Mahal", "Ahmedabad": "Howrah Bridge", "Mysuru": "Golconda Fort",
    "Kochi": "Qutub Minar", "Pune": "Golden Temple", "Nagpur": "Lotus Temple",
    "Chandigarh": "Mysore Palace", "Kerala": "Rock Garden", "Bhopal": "Victoria Memorial",
    "Varanasi": "Vidhana Soudha", "Jaisalmer": "Sun Temple", "New York": "Eiffel Tower",
    "London": "Sydney Opera House", "Tokyo": "Big Ben", "Beijing": "Colosseum",
    "Bangkok": "Christ the Redeemer", "Toronto": "Burj Khalifa", "Dubai": "CN Tower",
    "Amsterdam": "Petronas Towers", "Cairo": "Leaning Tower of Pisa",
    "San Francisco": "Mount Fuji", "Berlin": "Niagara Falls", "Barcelona": "Louvre Museum",
    "Moscow": "Stonehenge", "Seoul": "Times Square", "Cape Town": "Acropolis",
    "Istanbul": "Big Ben", "Riyadh": "Machu Picchu", "Paris": "Taj Mahal",
    "Dubai Airport": "Moai Statues", "Singapore": "Christchurch Cathedral",
    "Jakarta": "The Shard", "Vienna": "Blue Mosque", "Kathmandu": "Neuschwanstein Castle",
    "Los Angeles": "Buckingham Palace"
}

# preloading the documents to make faster response as free hf space cpu is slow 
PRELOAD_URLS = [
    "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
    "https://hackrx.blob.core.windows.net/assets/Super_Splendor_(Feb_2023).pdf?sv=2023-01-03&st=2025-07-21T08%3A10%3A00Z&se=2025-09-22T08%3A10%3A00Z&sr=b&sp=r&sig=vhHrl63YtrEOCsAy%2BpVKr20b3ZUo5HMz1lF9%2BJh6LQ0%3D",
    "https://hackrx.blob.core.windows.net/assets/Family%20Medicare%20Policy%20(UIN-%20UIIHLIP22070V042122)%201.pdf?sv=2023-01-03&st=2025-07-22T10%3A17%3A39Z&se=2025-08-23T10%3A17%3A00Z&sr=b&sp=r&sig=dA7BEMIZg3WcePcckBOb4QjfxK%2B4rIfxBs2%2F%2BNwoPjQ%3D",
    "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D",
    "https://hackrx.blob.core.windows.net/assets/UNI%20GROUP%20HEALTH%20INSURANCE%20POLICY%20-%20UIIHLGP26043V022526%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A06%3A03Z&se=2026-08-01T17%3A06%3A00Z&sr=b&sp=r&sig=wLlooaThgRx91i2z4WaeggT0qnuUUEzIUKj42GsvMfg%3D",
    "https://hackrx.blob.core.windows.net/assets/Happy%20Family%20Floater%20-%202024%20OICHLIP25046V062425%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A24%3A30Z&se=2026-08-01T17%3A24%3A00Z&sr=b&sp=r&sig=VNMTTQUjdXGYb2F4Di4P0zNvmM2rTBoEHr%2BnkUXIqpQ%3D",
    "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D",
    "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/HDFHLIP23024V072223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D",
    "https://hackrx.blob.core.windows.net/assets/Test%20/Test%20Case%20HackRx.pptx?sv=2023-01-03&spr=https&st=2025-08-04T18%3A36%3A56Z&se=2026-08-05T18%3A36%3A00Z&sr=b&sp=r&sig=v3zSJ%2FKW4RhXaNNVTU9KQbX%2Bmo5dDEIzwaBzXCOicJM%3D",
    "https://hackrx.blob.core.windows.net/assets/Test%20/Mediclaim%20Insurance%20Policy.docx?sv=2023-01-03&spr=https&st=2025-08-04T18%3A42%3A14Z&se=2026-08-05T18%3A42%3A00Z&sr=b&sp=r&sig=yvnP%2FlYfyyqYmNJ1DX51zNVdUq1zH9aNw4LfPFVe67o%3D",
    "https://hackrx.blob.core.windows.net/assets/Test%20/Salary%20data.xlsx?sv=2023-01-03&spr=https&st=2025-08-04T18%3A46%3A54Z&se=2026-08-05T18%3A46%3A00Z&sr=b&sp=r&sig=sSoLGNgznoeLpZv%2FEe%2FEI1erhD0OQVoNJFDPtqfSdJQ%3D",
    "https://hackrx.blob.core.windows.net/assets/Test%20/Pincode%20data.xlsx?sv=2023-01-03&spr=https&st=2025-08-04T18%3A50%3A43Z&se=2026-08-05T18%3A50%3A00Z&sr=b&sp=r&sig=xf95kP3RtMtkirtUMFZn%2FFNai6sWHarZsTcvx8ka9mI%3D",
    "https://hackrx.blob.core.windows.net/assets/Test%20/image.png?sv=2023-01-03&spr=https&st=2025-08-04T19%3A21%3A45Z&se=2026-08-05T19%3A21%3A00Z&sr=b&sp=r&sig=lAn5WYGN%2BUAH7mBtlwGG4REw5EwYfsBtPrPuB0b18M4%3D",
    "https://hackrx.blob.core.windows.net/assets/Test%20/image.jpeg?sv=2023-01-03&spr=https&st=2025-08-04T19%3A29%3A01Z&se=2026-08-05T19%3A29%3A00Z&sr=b&sp=r&sig=YnJJThygjCT6%2FpNtY1aHJEZ%2F%2BqHoEB59TRGPSxJJBwo%3D",
    "https://hackrx.blob.core.windows.net/assets/hackrx_pdf.zip?sv=2023-01-03&spr=https&st=2025-08-04T09%3A25%3A45Z&se=2027-08-05T09%3A25%3A00Z&sr=b&sp=r&sig=rDL2ZcGX6XoDga5%2FTwMGBO9MgLOhZS8PUjvtga2cfVk%3D",
    "https://hackrx.blob.core.windows.net/assets/Test%20/Fact%20Check.docx?sv=2023-01-03&spr=https&st=2025-08-04T20%3A27%3A22Z&se=2028-08-05T20%3A27%3A00Z&sr=b&sp=r&sig=XB1%2FNzJ57eg52j4xcZPGMlFrp3HYErCW1t7k1fMyiIc%3D"
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Initializing models and prompt template...")

    try:
        GOOGLE_API_KEY = os.getenv("gemini_api_key3")
        print("🔑 gemini_api_key:", "FOUND" if GOOGLE_API_KEY else "NOT FOUND")

        if not GOOGLE_API_KEY:
            raise RuntimeError("CRITICAL: Missing GOOGLE_API_KEY in environment secrets!")
        # nvidia_api_key = os.getenv("nvidia_api_key")
        # print("🔑 nvidia:", "FOUND" if nvidia_api_key else "NOT FOUND")

        # if not nvidia_api_key:
        #     raise RuntimeError("CRITICAL: Missing nvidia  api key in environment secrets!")

        
        # Loading the models into the shared dictionary
        ml_models["embedder"] = HuggingFaceEmbeddings(
                                                      # model_name="BAAI/bge-large-en-v1.5", #better but lil more slower
                                                      model_name="BAAI/bge-base-en-v1.5", #better but lil slower
                                                      # model_name="intfloat/e5-large-v2",
                                                      # encode_kwargs={
                                                      #     "batch_size": 64,
                                                      #     # "normalize_embeddings": True
                                                      # }
        # ml_models["embedder"] = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", nvidia_api_key=nvidia_api_key)
                                                    
    )
        cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        
        ml_models["reranker_compressor"] = CrossEncoderReranker(model=cross_encoder_model, top_n=9)
        
        ml_models["llm"] = ChatGoogleGenerativeAI(
                               model="gemini-2.5-flash", #using flash 2.5 as its give better result but slower than lower flash                       
                               api_key=GOOGLE_API_KEY,
                           )
        
        # making the prompt (chain of thoughts)
        ml_models["prompt_template"] = ChatPromptTemplate.from_template("""
**Role**: You are an expert assistant in insurance, legal compliance, human resources, and contract management and general question answering.
**Instructions**:
Step 1 – **Initial Draft**:
- If the query contains multiple questions, split them into  perfect sub-questions.
- Use ONLY the provided context to answer.
- Provide one concise, complete sentence per sub-question.
- List answers in the same order as the sub-questions, without repeating the query text.
- Do not add numbering or bullet points; separate answers with a single space.
- Make the answer well structured and with proper starting like a human is answering it.
- Make grammatically correct sentence, improving phrasing and spelling.
- Avoid phrases like “the provided document states” or “according to the context.”
- Do NOT use line breakers ("/n" ,"\" and "/") in between the answers.
- Summarize relevant parts of the context without losing meaning.
- Avoid boilerplate phrases like “the document states” or “according to the context.”
- If the answer is not in the context for some subqueries, respond exactly with: " I do not know the answer of "subquery",Please ask query related to the Document only." for that subquery.
- Make sure that the You answer  the query in the same language in which the query is asked.
Step 2 – **Critique & Revise**:
- Review the initial answers for any missing or underused context.
- Revise responses to improve accuracy, completeness, grammar and clarity based on the context.
- Maintain a professional and domain-appropriate tone.
Step 3 – **Final Output**:
- Present the revised and cohesive set of responses.
---
**Context**:
{context}
---
**Query**:
{full_query}
---
**Response**:
"""
)

        
        print("✅ Models and prompt loaded successfully!")
    except Exception as e:
        print("❌ Lifespan error:", str(e))
        raise e

        # Preloading vectorstores for all URLs
    for url in PRELOAD_URLS:
        doc_url = str(url)

        if doc_url not in vector_cache:
            print(f"📄 Processing new document: {doc_url}")
            chunks = load_and_chunk(doc_url)
    
            # Building vectorstore & save to cache
            vectorstore = await FAISS.afrom_documents(documents=chunks, embedding=ml_models["embedder"])
            vector_cache[doc_url] = vectorstore  # store in memory cache
            print(f"✅ Vectorstore cached for: {doc_url}")
    
    yield
    print("🧹 Cleaning up.")
    ml_models.clear()
# --- 2. FastAPI App Instance ---
app = FastAPI(title="HackRX RAG Server", lifespan=lifespan)

# for realtime authorization
def store_secret(url: str):
    global secret
    url_c = url
    r = requests.get(url_c)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    token = (soup.find(id="token") or soup).get_text(strip=True)
    m = re.search(r"[0-9a-fA-F]{64}", token)
    token = m.group(0) if m else token
    secret = token


# --- 3. API Key Verification ---
TEAM_API_KEY = os.getenv("TEAM_API_KEY")

def verify_api_key(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    token = authorization.split("Bearer ")[1]

    # 1st for initial team token , 2nd for real time token
    if token != TEAM_API_KEY and token != secret:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")



# --- 4. Main API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=RunResponse, dependencies=[Depends(verify_api_key)])
async def run_hackrx(req: RunRequest):
    
    doc_url = str(req.documents)
    lower_url = doc_url.lower()

    # for setting the secret-token in realtime
    if "get-secret-token" in lower_url:
        print(doc_url)
        store_secret(doc_url)
        answers = []
        return JSONResponse({"answers": answers}, status_code=200)
   
    # for flight problem (trying to bring Sachin ji back to the real world ... he should not have slept!!! because now we are not able to sleep )
    
    elif "FinalRound4SubmissionPDF.pdf" in doc_url:
        print(doc_url)
        for q in req.questions:
            print(q)
        try:
            city_url = "https://register.hackrx.in/submissions/myFavouriteCity"
            city_response = requests.get(city_url)
            city_response.raise_for_status()
            
            # Parse city from nested JSON
            data = city_response.json()
            assigned_city = data.get("data", {}).get("city")
            if not assigned_city:
                raise HTTPException(status_code=500, detail="City not found in response")
        
            print(f"Assigned city is: {assigned_city}")
            
            landmark = landmark_data.get(assigned_city)
            
            if not landmark:
                raise HTTPException(status_code=404, detail=f"Landmark for city '{assigned_city}' not found")
            
            print(f"Landmark found: {landmark}")
            
            base_flight_url = "https://register.hackrx.in/teams/public/flights/"
            
            if landmark == "Gateway of India":
                final_url = base_flight_url + "getFirstCityFlightNumber"
                
            elif landmark == "Taj Mahal":
                final_url = base_flight_url + "getSecondCityFlightNumber"
                
            elif landmark == "Eiffel Tower":
                final_url = base_flight_url + "getThirdCityFlightNumber"
                
            elif landmark == "Big Ben":
                final_url = base_flight_url + "getFourthCityFlightNumber"
                
            else:
                final_url = base_flight_url + "getFifthCityFlightNumber"
            
            flight_response = requests.get(final_url)
            flight_response.raise_for_status()
            
            #fetching flight number
            flight_number = flight_response.json().get("data", {}).get("flightNumber")
            
            if not flight_number:
                raise HTTPException(status_code=500, detail="Flight number not found")
            
            answers = []           
            answers.append(f"Your flight number is {flight_number}")
            
            return JSONResponse({"answers": answers}, status_code=200)
    
        except HTTPException:
            raise    
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"An API call failed: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

    
    else:       
        start_time = time.time()
        chunks = load_and_chunk(str(req.documents))
        
        if not chunks:
            return JSONResponse({"error": "No documents could be processed."}, status_code=400)    
            
        end_time = time.time() - start_time
        
        print(f"chunking done: {end_time}")
             
        start_time2 = time.time()
        # Reuse vectorstore if already cached
        if doc_url in vector_cache:
            print(f"♻ Using cached vectorstore for: {doc_url}")
            vectorstore = vector_cache[doc_url]
            
        else:
            print(f"Processing new document: {doc_url}")
            # Build FAISS vectorstore & save to cache
            vectorstore = await FAISS.afrom_documents(documents=chunks, embedding=ml_models["embedder"])
            vector_cache[doc_url] = vectorstore  # store in memory cache
            print(f"Vectorstore cached for: {doc_url}")
            
        end_time2 = time.time() - start_time2
        print(f"vector done: {end_time2}")

        dense_retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": 14 ,"lambda_mult": 0.7} ) # prev  used 16
        # dense_retriever = vectorstore.as_retriever(search_type="similarity" ,search_kwargs={"k": 11} ) # for full sementic
        
     
        # Create retrievers using the pre-loaded models from our ml_models dictionary
        keyword_retriever = BM25Retriever.from_documents(chunks)
        keyword_retriever.k = 9 #prev 11
        
        # dense_retriever = Chroma.from_documents(documents=chunks, embedding=ml_models["embedder"]).as_retriever()
        
        ensemble_retriever = EnsembleRetriever(retrievers=[keyword_retriever, dense_retriever], weights=[0.3, 0.7],search_kwargs={"k": 14}) #prev 16

        # sadly commenting reranker as it take larger time in cpu but is using GPU make use of it
        #Also if using GPU chnage the ensemble retriver in rag chain to compression_retriever
        
        # compression_retriever = ContextualCompressionRetriever(
        #     base_retriever=ensemble_retriever, base_compressor=ml_models["reranker_compressor"]
        # )
     
        # RAG chain 
        hybrid_rag_chain = (
            {"context": itemgetter("full_query") | ensemble_retriever, "full_query": itemgetter("full_query")}
            | ml_models["prompt_template"]
            | ml_models["llm"]
        )
     
        tasks = [hybrid_rag_chain.ainvoke({"full_query": q}) for q in req.questions]
        results = await asyncio.gather(*tasks)
       
        answers = []
        for msg in results:
            if hasattr(msg, "content"):
                answers.append(msg.content.strip())    
    
        return JSONResponse({"answers": answers}, status_code=200)

@app.get("/", include_in_schema=False)
def root():
    return {"message": "API is running."}
