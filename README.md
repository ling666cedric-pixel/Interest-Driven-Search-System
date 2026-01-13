# Interest-Driven Personalized Search System

## 📖 Introduction
This is a course project for **Information Retrieval (IR)**. The goal is to build a personalized search system that adjusts retrieval results based on user behavior history to achieve **interest-driven ranking**.

Unlike traditional search engines that return the same results for everyone, this system utilizes user profiles (derived from historical ratings) to re-rank search results, delivering content that matches specific user interests (e.g., boosting "Animation" movies for animation lovers).

## 🚀 Features
- **Data Source**: Powered by the **MovieLens Latest Small Dataset** (9,000+ movies, 100,000+ ratings).
- **Retrieval Model**: Implemented **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Vector Space Model (VSM)** for accurate document retrieval.
- **Personalization**:
  - **User Profiling**: Automatically builds user interest profiles based on high-rated history.
  - **Re-ranking Algorithm**: Linearly boosts scores for documents matching user interest tags.
- **User Interface**: Interactive Command Line Interface (CLI) supporting multiple user simulations.

## 📂 Project Structure
```text
.
├── ir_system.py          # Main source code (System logic & UI)
├── movies.csv            # Movie metadata (Title, Genres)
├── ratings.csv           # User rating logs (UserId, MovieId, Rating)
├── requirements.txt      # Python dependencies
