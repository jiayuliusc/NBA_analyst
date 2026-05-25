import ollama
import re
from .retrieval import retrieve_info_v2

def clean_deepseek_output(text):
    # Remove the <think>...</think> blocks to show only the final answer
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def generate_answer(user_question):
    # Retrieve Context
    retrieved_docs = retrieve_info_v2(user_question, k=5)
    
    if not retrieved_docs:
        return "I couldn't find any specific game logs matching your question."

    context_block = "\n".join(retrieved_docs)

    # R1 models prefer direct instructions over complex personas, 
    # but we still guide it to use the context.
    prompt = f"""
    Instruction: Answer the user's question using ONLY the provided game logs. 
    If the answer is not in the logs, state that you do not have the data.
    Cite specific stats, dates, and opponents.
    
    === GAME LOGS ===
    {context_block}

    === QUESTION ===
    {user_question}
    """

    # Call DeepSeek
    try:
        response = ollama.chat(model='deepseek-r1:1.5b', messages=[
            {'role': 'user', 'content': prompt},
        ])
        
        raw_content = response['message']['content']
        
        # Clean the output
        final_answer = clean_deepseek_output(raw_content)
        
        return final_answer
        
    except Exception as e:
        return f"Error generating answer: {e}"

if __name__ == "__main__":
    # Quick Test
    print(generate_answer("How did Anthony Edwards play in his last game?"))
