from groq import Groq
from typing import List, Tuple, Dict
from utils.vector_database import similarity_search

def create_medical_prompt_template():
    return """You are an expert medical assistant with access to comprehensive medical knowledge.

Based on the provided medical context, answer the user's question clearly and accurately.

GUIDELINES:
1) Provide clear, well-structured responses.
2) Use bullet points or numbered lists for complex information.
3) Explain medical terms in simple language.
4) Be specific and factual based on the context.
5) If the context doesn't contain relevant information, clearly state this.
6) Don't use + for bullet points and use • .

MEDICAL CONTEXT:
{context}

USER QUESTION: {question}

RESPONSE:"""

def retrieve_relevant_context(query: str, top_k: int = 5) -> str:
    print(f"Retrieving relevant context for: '{query[:50]}...'")
    
    results = similarity_search(query, k=top_k)
    
    if not results:
        print("No relevant context found")
        return "No relevant medical information found in the knowledge base."
    
    # Format context with relevance scores
    contexts = []
    for i, (text, score, metadata) in enumerate(results, 1):
        context_piece = f"[Source {i} - Relevance: {score:.2f}]\n{text.strip()}"
        contexts.append(context_piece)
    
    formatted_context = "\n\n" + "="*50 + "\n\n".join(contexts)
    print(f"Retrieved {len(results)} relevant contexts")
    return formatted_context

def generate_medical_response(query: str, groq_client: Groq) -> str:
    """Generate medical response using RAG approach"""
    print(f"Generating response for query...")
    
    try:
        # Step 1: Retrieve relevant context
        context = retrieve_relevant_context(query, top_k=4)
        
        # Step 2: Create prompt
        prompt_template = create_medical_prompt_template()
        prompt = prompt_template.format(context=context, question=query)
        
        # Step 3: Generate response
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # Lower temperature for more accurate medical responses
            max_tokens=1200,
            top_p=0.9,
            stream=False
        )
        
        response = completion.choices[0].message.content
        
        # Step 4: Clean and format response
        formatted_response = format_medical_response(response)
        
        print("Medical response generated successfully")
        return formatted_response
        
    except Exception as e:
        print(f"Error generating medical response: {e}")
        return "I apologize, but I encountered an error while processing your medical question. Please try rephrasing your question or consult a healthcare professional directly."

def format_medical_response(response: str) -> str:
    # Remove excessive formatting
    response = response.replace("**", "")
    response = response.replace("*", "")
    
    # Ensure proper spacing
    response = response.replace("\n\n\n", "\n\n")
    response = response.strip()
    
    # Add medical disclaimer at the bottom
    disclaimer = ""
    
    if not response.endswith(disclaimer.strip()):
        response += disclaimer
    
    return response

def validate_medical_query(query: str) -> bool:
    medical_indicators = [
        'symptom', 'disease', 'condition', 'treatment', 'medicine', 'medication',
        'diagnosis', 'doctor', 'hospital', 'pain', 'fever', 'infection',
        'medical', 'health', 'illness', 'therapy', 'surgery', 'procedure',
        'blood', 'heart', 'lung', 'brain', 'liver', 'kidney', 'cancer',
        'diabetes', 'hypertension', 'pneumonia', 'what is', 'how to treat',
        'causes of', 'prevention', 'cure', 'relief'
    ]
    
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in medical_indicators)
