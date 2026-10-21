"""Loading source pages into the pgvector passage store (rag_documents / rag_chunks).

config.py     reads data/collections.yaml
documents.py  turns raw .txt files into SourceDocuments (real title, URL, sha256)
pipeline.py   plans an incremental run (new / changed / unchanged / removed),
              chunks, counts tokens, embeds with retries and writes to a store
stores.py     where passages are written: Postgres directly (SqlRagStore) or
              through Supabase's REST API (RestRagStore)

scripts/ingest.py is the command line around this.
"""
