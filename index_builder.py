from document_uploader import build_indexes, ingest_documents

# Example usage
if __name__ == "__main__":
    file_path = "your_pdf_text_extracted.txt"  # Replace with your extracted text file
    embeddings, documents = ingest_documents(file_path)
    vector_index = build_indexes(embeddings)
    print("Indexes created and stored locally.")


