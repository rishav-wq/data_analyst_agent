# utils/llm_handler.py
import os
import time
import together
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY not found in .env file or environment variables.")

# Use environment variable instead of deprecated together.api_key
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

# Primary model - if rate limited, we'll fall back to alternatives
MODEL_NAME = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
# Fallback models with higher rate limits
FALLBACK_MODELS = [
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
]

def get_llm_response(prompt, conversation_history=None, max_tokens=1024, temperature=0.7, top_p=0.7, top_k=50, repetition_penalty=1, retry_with_fallback=True):
    """
    Gets a response from the Together AI LLM.

    Args:
        prompt (str): The user's current prompt/question.
        conversation_history (list, optional): List of previous messages for context.
                                              Each message is a dict: {"role": "user/assistant", "content": "message text"}
        max_tokens (int): Max tokens for the response.
        temperature (float): Controls randomness. Lower is more deterministic.
        top_p (float): Nucleus sampling.
        top_k (int): Top-k sampling.
        repetition_penalty (float): Penalizes repeated tokens.
        retry_with_fallback (bool): Whether to try fallback models if rate limited.

    Returns:
        str: The LLM's response.
    """
    messages = []
    if conversation_history:
        messages.extend(conversation_history)
    
    messages.append({"role": "user", "content": prompt})

    # Try primary model first, then fallbacks if rate limited
    models_to_try = [MODEL_NAME]
    if retry_with_fallback:
        models_to_try.extend(FALLBACK_MODELS)
    
    for model in models_to_try:
        try:
            print(f"Trying model: {model}")
            
            # Use the newer client approach
            client = together.Together()
            
            response = client.completions.create(
                model=model,
                prompt=_format_messages_for_llama(messages),
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop=['</s>', '[/INST]', '<|im_end|>']
            )
            
            if response.choices and len(response.choices) > 0:
                result = response.choices[0].text.strip()
                print(f"✅ Success with model: {model}")
                return result
            else:
                print(f"❌ No choices in response for model {model}:", response)
                continue
                
        except Exception as e:
            error_str = str(e)
            print(f"❌ Error with model {model}: {error_str}")
            
            # Check if it's a rate limit error
            if "429" in error_str or "rate_limit" in error_str.lower():
                print(f"Rate limited on {model}, trying next model...")
                if model != models_to_try[-1]:  # Not the last model
                    time.sleep(1)  # Brief pause before trying next model
                    continue
                else:
                    return "Error: All models are rate limited. Please try again later."
            else:
                # For non-rate-limit errors, try next model immediately
                continue
    
    return "Error: All models failed to respond."

def _format_messages_for_llama(messages):
    """
    Format messages for Llama-based models using the chat template format.
    """
    formatted_prompt = ""
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            formatted_prompt += f"[INST] {content} [/INST]"
        elif role == "assistant":
            formatted_prompt += f" {content} </s><s>"
    
    return formatted_prompt

# Alternative version using the newer Together AI client
def get_llm_response_v2(prompt, conversation_history=None, max_tokens=1024, temperature=0.7, top_p=0.7, top_k=50, repetition_penalty=1):
    """
    Simplified version that uses chat completions API instead of completions.
    This might work better for some models.
    """
    messages = []
    if conversation_history:
        messages.extend(conversation_history)
    
    messages.append({"role": "user", "content": prompt})

    try:
        # Try using chat completions instead
        client = together.Together()
        
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",  # Use a more reliable model
            messages=messages,  # Use messages directly instead of formatted prompt
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content.strip()
        else:
            print("No choices in response:", response)
            return "Error: No response generated."
            
    except Exception as e:
        print(f"Error calling Together AI API (v2): {e}")
        return f"Error: API call failed. Details: {str(e)}"

def add_rate_limit_delay():
    """Add a delay to help with rate limiting."""
    print("⏱️  Adding delay to avoid rate limits...")
    time.sleep(2)

if __name__ == '__main__':
    # Test the function with rate limit handling
    test_prompt = "What is the capital of France?"
    history = [
        {"role": "user", "content": "Hello, I need some help."},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
    ]
    
    print("=== Testing main function with fallback models ===")
    response = get_llm_response(test_prompt, conversation_history=history)
    print(f"User: {test_prompt}")
    print(f"LLM: {response}")
    print()

    add_rate_limit_delay()

    print("=== Testing without history ===")
    response_no_history = get_llm_response("Explain quantum entanglement in simple terms.")
    print(f"User: Explain quantum entanglement in simple terms.")
    print(f"LLM: {response_no_history}")
    print()
    
    add_rate_limit_delay()
    
    print("=== Testing chat completions API (v2) ===")
    response_v2 = get_llm_response_v2("What is 2+2?")
    print(f"User: What is 2+2?")
    print(f"LLM: {response_v2}")
    
    print("\n=== Rate Limit Info ===")
    print("The Llama-4-Maverick model has very low rate limits (0.6 queries/minute).")
    print("Consider using alternative models like Meta-Llama-3.1-8B-Instruct-Turbo for development.")
    print("You can change MODEL_NAME at the top of the file to use a different default model.")