# we are importing the require library
from vllm import LLM, SamplingParams
import concurrent.futures

# We are Initialize the LLM model
model = LLM()
# We are Define a function that represents your samping process
def sample(index):
    # We are Define the sampling parameter
    params = SamplingParams(
        # We are Adjust the temperature parameter for sampling
        temperature =1.0,
        # Adjust the maximum number of tokens to generate
        max_token=50,
        # We are Adjust the top-k parameter for sampling
        top_k = 50
    )
    
    # We are Generate text uing the LLM model
    generated_text = model.generate_text(params)
    
    return generated_text




def main():
    # We are Define the number of samples you want to generate in parallel
    num_samples = 10
    
    # We are Creating a ThreadPoolExecutor with a maximum of 5 worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # We are Submit the sampling tasks to the executor
        futures = [ executor.submit(sample, i) for i in range(num_samples)]
        
        # We are Wait for all tasks to complete and retrieve results
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # We are Print the results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()