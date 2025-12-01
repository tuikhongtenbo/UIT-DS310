# Data for the VLSP 2025 DRiLL: The challenge of Deep Retrieval in the expansive Legal Landscape
- The contents of the legal passages whose `id` is mentioned in the training are provided in `legal_corpus.json`.
- The data is provided in JSON format and UTF-8 encoding.
- Please refer to https://vlsp.org.vn/vlsp2025/eval/drill for more information. 

## File Overview and Structure

### 1. `train.json`
- **Decription**: Annotated training dataset.  
- **Schema**:
  ```json
  {
    "qid": <integer>,             // Unique question identifier
    "question": "<string>",       // Text of the question
    "relevant_laws": [<integer>],  // List of `aid` values (article IDs) from `legal_corpus.json`
    "answer": "<string>"           // Text of the answer
  }

### 2. `legal_corpus.json`
- **Purpose**: Legal articlecorpus.
- **Structure**:
  ```json
  {
    "id": <integer>,               // Unique document identifier
    "law_id": <integer>,           // Official law number
    "content": [
      {
        "aid": <integer>,          // Article identifier within this document
        "content_Article": "<string>" // Full text of the article
      }
      // ... additional articles
    ]
  }