# We are import the require library
import redis

class EphemeralStore:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.StrictRedis(host=host, port=port, db=db)
        
    def save_prompt_response(self, prompt, response, expiration_seconds=300):
        """
        Save a prompt-response pair in the ephemeral store with an expiration time.
        """
        self.redis_client.setex(prompt, expiration_seconds, response)
        
    def get_response_for_prompt(self, prompt):
        """
        Retrieve the response for a given prompt from the ephemeral store.
        """
        response = self.redis_client(prompt)
        if response:
            return response.decode('utf-8')
        else:
            return None
    
    def delete_prompt_response(self, prompt):
        """
        Delete a prompt-response pair from the ephemeral store.
        """
        self.redis_client.delete(prompt)
        
# We are define the entrypoint of the fucntion
if __name__ == "__main__":
    # We are define the example usage
    store = EphemeralStore()
    
    # We are save a promt-response pair with a 5 minute expiration time
    prompt = "What is the meaning of life?"
    response = "The meaning of life is 42."
    store.save_prompt_response(prompt, response, expiration_seconds=300)
    
    # We are Retrieve the response for a prompt
    retrieved_response = store.get_response_for_prompt(prompt)
    print("Retrieved response:", retrieved_response)
    
    # We are delete the prompt-esponse pair
    store.delete_prompt_response(prompt)
    
    # We are check if the response is deleted
    retrieved_response = store.get_response_for_prompt(prompt)
    print("Retrieved response after deletion:", retrieved_response)
