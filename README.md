AyurGenix — Intelligent Ayurvedic Agentic RAG System
A Multimodal, Agentic AI System for Personalized Ayurveda-Based Health Insights, Diet Plans & Lifestyle Recommendations

# Overview

AyurGenix is an AI-powered Agentic RAG system built to transform unstructured classical Ayurvedic literature—often found in noisy scanned Sanskrit–English mixed PDFs—into an intelligent, queryable knowledge system capable of generating contextual, grounded, and medically faithful responses.

The project handles every stage of the pipeline:

 Document ingestion (scanned & text PDFs)
 OCR + noise removal
 Sanskrit–English language classification
 Spelling correction for Sanskrit & English
 Text normalization + chunking
 Transformer-based embedding
 Vector storage using Pinecone
 Multi-agent reasoning + retrieval
 FastAPI backend
 Streamlit frontend chatbot

# Project Objective

AyurGenix is an advanced Agentic Retrieval-Augmented Generation (RAG) system designed to convert classical Ayurvedic medical texts (Charaka Samhita, Sushruta Samhita, Ashtanga Hridaya, Madhava Nidana, etc.) into an interactive intelligent assistant capable of:

 Understanding user health symptoms
 Recommending daily lifestyle routines (Dinacharya / Ritucharya)
 Suggesting Ayurvedic diet plans based on dosha constitution
 Explaining herbs, treatments, formulations (Yoga, Dravya, Rasayana)
 Providing personalized wellness guidance grounded strictly in classical texts

All recommendations are contextual, evidence-backed, and retrieved directly from authoritative Ayurvedic literature using a robust vector-search pipeline.

# Key Features
 1. High-Fidelity OCR + Cleaning Pipeline

Handles noisy scanned Sanskrit–English PDFs:

Image cleaning & deskewing

OCR extraction

Unicode normalization

Removal of artefacts, broken characters, page numbers

 2. Sanskrit–English Token Classifier (Char-CNN)

Custom Character-CNN model trained on thousands of Sanskrit, English, and OCR-junk tokens.

Used for:

Removing gibberish

Correcting Sanskrit spelling

Correcting English words

Improving embedding quality

 3. Ayurveda-Specific Spell Correction

Two modules:

Sanskrit Spell Checker (fuzzy + IAST normalization + dictionary matching)

English Spell Checker (SymSpell + wordlist)

 4. Sentence-Aware Chunking

Smart chunking of Ayurvedic scripture into clean, overlapping segments optimized for retrieval.

 5. Transformer Embedding Pipeline

Embeddings generated using:

all-MiniLM-L6-v2 (384-dim)

Mean pooling

GPU-accelerated inference

 6. Vector Store (Pinecone)

Stores millions of high-quality chunks with metadata:

Source

Page number

Document ID

Cleaned text

Enables fast, semantically rich retrieval.

 7. Agentic RAG System

A multi-agent pipeline consisting of:

Retrieval Agent

Reranking Agent

Context Constructor

Answer-Generation Agent

Ensures:

Hallucination-free generation

Answer grounded in classical references

Context-aware recommendations

 8. FastAPI Backend + Streamlit Frontend

Fully deployable REST API

Chat-style frontend enabling real-time conversation

Response includes sources and page numbers for transparency

# What it Can Do
 Personalized Health Guidance

Users can ask:

“I feel heavy after meals; what does Ayurveda suggest?”

“What are symptoms of aggravated Pitta?”

“How to manage constipation naturally?”

The system returns:

Explanation from classical texts

Herbal recommendations

Diet restrictions

Lifestyle routines

 Diet Recommendations

Food compatibility

Seasonal diets

Dosha-balancing meals

Contraindicated foods

 Lifestyle & Daily Routines

Dinacharya (daily routine)

Ritucharya (seasonal regimen)

Sleep, exercise, meditation practices

 Treatments & Home Remedies

Sourced strictly from ancient scriptures:

Ghee-based therapies

Decoctions (Kashaya)

Oils for massage (Abhyanga)

Traditional formulations

# Tech Stack Overview

AyurGenix is built using a modern, production-grade AI engineering stack enabling efficient data processing, scalable retrieval, and agentic reasoning:

Python 3.12 — Core development language powering preprocessing, modeling, and APIs.

PyTorch — Used for transformer embeddings, Char-CNN classification, and GPU-accelerated inference.

Transformers (HuggingFace) — Provides MiniLM embedding model, tokenizers, and model utilities.

Sentence-Transformers — Efficient embedding backend for generating low-latency semantic vectors.

Pinecone Serverless Vector DB — Stores millions of embeddings with fast cosine similarity search.

FastAPI — Backend service delivering high-performance, async RAG endpoints.

Streamlit — Clean, interactive chat-style frontend UI interface.

OpenCV + Tesseract — For OCR extraction and scanned document preprocessing.

ftfy, Unidecode, Regex — Text normalization and OCR artifact correction tools.

Scikit-Learn — Label encoding and utility support for Char-CNN preprocessing.

Cologne Sanskrit Dictionaries — Enriched linguistic dataset powering Sanskrit spell correction.

SymSpell — Fast English word spell-correction system.

Pickle / Joblib — For storing trained models, tokenizers, and preprocessing pipelines.

JSONLines — Handles large-scale document streaming for cleaned Ayurvedic texts.

TQDM — Real-time progress bars for large-scale embedding runs.

Numpy / Pandas — Vector operations and dataset manipulation.