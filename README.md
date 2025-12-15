# Multi-Dimensional Artist Profiling

## Project Summary
This project builds comprehensive, data-driven profiles for six globally popular artists—**Drake, Kendrick Lamar, The Weeknd, Billie Eilish, Taylor Swift, and Bad Bunny**—to show that no single metric (like sentiment or popularity) can fully capture an artist’s identity. Instead, we combine **lyrical content**, **thematic structure**, **public perception**, and **temporal fame dynamics** into a unified analytical framework.

## Youtube Links
**Full Presentation:** https://youtu.be/75aeonsjKK0

**Code demos of each part:**
Lyric Sentiment: https://youtu.be/ZwsHOmrfXFE  
Theme analysis: https://youtu.be/l7HYH53H9nw 
Public Perception: https://youtu.be/GlcwNCH_gfM  
Fame over time: https://youtu.be/BruqunChwj8   


## Core Components
1. **Lyric Sentiment Analysis**  
   - Dataset: Genius Song Lyrics  
   - Model: `distilbert-base-uncased` (fine-tuned on SST-2)  
   - Produces normalized, song-level sentiment distributions per artist.

2. **Common Themes in Songs**  
   - Dataset: Genius Song Lyrics  
   - Models: Sentence-BERT (`all-mpnet-base-v2`) + K-Means  
   - Lyrics are split into sentences, embedded, clustered, and manually labeled into interpretable themes (e.g., love, introspection, confidence, industry references).

3. **Public Perception via Media**  
   - Datasets: Twitter, Reddit, News  
   - Models: `twitter-roberta-base-sentiment-latest`, Sentence-BERT, FAISS  
   - Includes sentiment aggregation and a Retrieval-Augmented Generation (RAG) pipeline for question-answering over media content, evaluated with DeepEval metrics.

4. **Fame Over Time**  
   - Dataset: Spotify Global Music Dataset (2009–2025)  
   - Model: SARIMAX  
   - Tracks historical popularity, identifies fame spikes, and forecasts artist popularity over a 5-year horizon.

## Key Insight
By integrating NLP, clustering, RAG, and time-series forecasting, this project demonstrates that an artist’s “identity” emerges from the interaction between what they create, how audiences respond, and how their influence evolves over time.


