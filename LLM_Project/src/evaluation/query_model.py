import json
import urllib.request


def query_model(
    prompt,
    model="llama3",
    url="http://localhost:11434/api/chat",
    seed=123,
    temperature=0.7,
    max_tokens=256,
    context_size=2048
):
    """
    Query a deployed language model and get a response.

    Args:
        prompt (str): Input prompt to the language model.
        model (str): Name of the model to query (default: "llama3").
        url (str): URL of the model's REST API endpoint.
        seed (int): Seed for deterministic behavior.
        temperature (float): Sampling temperature for generation.
        max_tokens (int): Maximum number of tokens to generate.
        context_size (int): Context window size.

    Returns:
        str: The model's response.
    """
    # Create the data payload for the API request
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "num_ctx": context_size
        }
    }

    # Convert the data dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object with necessary headers
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # Send the request and handle the response
    try:
        with urllib.request.urlopen(request) as response:
            response_data = response.read().decode("utf-8")
            response_json = json.loads(response_data)
            return response_json["message"]["content"]
    except Exception as e:
        print(f"Error querying the model: {e}")
        return None


if __name__ == "__main__":
    # Example usage of the query_model function
    prompt = "What is the capital of France?"
    model_name = "llama3"
    api_url = "http://localhost:11434/api/chat"

    response = query_model(prompt, model=model_name, url=api_url)
    if response:
        print(f"Model response:\n{response}")
    else:
        print("Failed to get a response from the model.")
