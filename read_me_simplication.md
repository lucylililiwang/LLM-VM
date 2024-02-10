![Anarchy Logo](anarchy_logo.svg)

<p align="center">
  <a href="https://anarchy.ai/" target="_blank"><img src="https://img.shields.io/badge/View%20Documentation-Docs-yellow"></a>
  <a href="https://discord.gg/YmNvCAk6W6" target="_blank"><img src="https://img.shields.io/badge/Join%20our%20community-Discord-blue"></a>
  <a href="https://github.com/anarchy-ai/LLM-VM">
      <img src="https://img.shields.io/github/stars/anarchy-ai/LLM-VM" />
  </a>
</p>
<h1 align='center'> 🤖 Anarchy LLM-VM 🤖 </h1>
<p align='center'><em>An Open-Source AGI Server for Open-Source LLMs</em></p>

This is [Anarchy's](https://anarchy.ai) effort to build 🏗️ an open generalized artificial intelligence 🤖 through the LLM-VM: a way to give your LLMs superpowers 🦸 and superspeed 🚄.

You can find detailed instructions to try it live here: [anarchy.ai](https://anarchy.ai)

> This project is in BETA. Expect continuous improvement and development.


# Table of Contents

* [Table of Contents](#table)
* [About](#-about-)
    * [What](#-what-is-the-anarchy-llm-vm)
    * [Why](#-why-use-the-anarchy-llm-vm)
    * [Features and Roadmap](#-features-and-roadmap)
* [Quick Start and Installation](#-quickstart-)
   * [Requirements](#-requirements)
   * [Installation](#-installation)
   * [Generating Completions](#-generating-completions)
   * [Running LLMs Locally](#-running-llms-locally)
   * [Supported Models](#-supported-models)
   * [Picking Different Models](#-picking-different-models)
   * [Tool Usage](#-tool-usage)
* [Contributing](#-contributing-)

## 📚 About 📚

### 💁 What is the Anarchy LLM-VM?

The Anarchy LLM-VM is like a smart and efficient brain for running Large Language Models (LLMs). It's designed to handle everything you need for competing tasks with LLMs, such as using tools, remembering information, adapting to new data, and fine-tuning its performance.

Think of it as a virtual machine or interpreter specically built for understanding and generating human language. It manages the flow of the information between the data  you give it, the models it runs (using the CPU), the instructions you provide ( your prompts or code), and the tools it uses to interact with the outside world.

The special thing about the LLM-VM is that it does all of these tasks in one place, in a very opinonated manner. This means it follows a specific set of rules and pratices that have been carefully chosen to optimize its performance. By handling everything internally, it can efficiently process multiple tasks ar once, which would be very costly if done using separate systems.

Additionally, the LLM-VM doesn't stick to one specific model or architecture. It's designed to adapt to different models and system setups, optimizing its operations based on what works best for the current setup.

In summary, the Anarchy LLM-VM is like a powerful and adaptable brain that handles all the complex tasks of working with large language models, making it easier and more efficient to use them for various purposes.

### 🤌 Why use the Anarchy LLM-VM?

In line with Anarchy's mission, the LLM-VM aims to back open-source models. When you use open-source models and run them on your local system, you gain several advantages:

* **Speed up your AGI development 🚀:** *With AnarchyAI, you only need one interface to engage with the newest LLMs.
  
* **Lower your costs 💸:** *Running models locally can reduce the pay-as-you-go costs of development and testing.*
  
* **Flexibility 🧘‍♀️:** *Anarchy lets you quickly switch between popular models, so you can find the perfect tool for your project.
  
* **Community Vibes 🫂:** *Join our active community of highly motivated developers and engineers working passionately to democratize AGI*
  
* **WYSIWYG 👀:** *with open source, there are no secrets; we prioritize transparency and efficiency, allowing you to concentrate on building*

### 🎁 Features and Roadmap

* **Implicit Agents 🔧🕵️:** *The Anarchy LLM-VM can be set up to use external tools through our agents such as **REBEL** just by supplying tool descriptions!*

* **Inference Optimization 🚄:** *The Anarchy LLM-VM is fine-tuned for optimal performance across various LLM architectures, ensuring you get the best value. With cutting-edge techniques like batching, sparse inference, quantization, distillation, and multi-level colocation, we offere the fastest framework possible.*

* **Task Auto-Optimization 🚅:** *The Anarchy LLM-VM examines your tasks to find repetitive ones where it can use student-teacher distillation. This process process trains a highly efficient small model from larger one without sacrificing accuracy. Additionally, it can utilize daya synthesis techniques to enhance outcomes.*


* **Library Callable 📚:** *We provide a library that can be used from any Python codebase directly.*

* **HTTP Endpoints 🕸️:** *We provide an HTTP standalone server to handle completion requests.*

* **Live Data Augmentation 📊:** (ROADMAP) *You will be able to provide a live updating data set and the Anarchy LLM-VM will **fine-tune** your models or work with a **vector DB** to provide up-to-date information with citations*

* **Web Playground 🛝:** (ROADMAP) *You will be able to run the Anarchy LLM-VM and test its outputs from the browser.*

* **Load-Balancing and Orchestration ⚖️:** (ROADMAP) *If you have multiple LLMs or providers you want to use, you can give them to the Anarchy LLM-VM. It will automatically determine which one to use and when, to optimize your uptime or your costs.*

* **Output Templating 🤵:** (ROADMAP) *You can make sure the LLM only gives data in certain formats and fills in variables  from a template using regular expressions, LMQL, or OpenAI's template language.*

* **Persistent Stateful Memory 📝:** (ROADMAP) *The Anarchy LLM-VM can remember a user's conversation history and react accordingly*

## 🚀 Quickstart 🚀

### 🥹 Requirements

#### Installation Requirements

Python >=3.10 Supported. Older Python versions are supported on a best-effort basis.

To check your Python version, use the command:
 ```bash > python3 --version ```
To upgrade Python:

* Create a new Python environment using:
 ```bash > conda create -n myenv python=3.10 ``` 
* Alternatively, visit https://www.python.org/downloads/ to download the latest version.

if you intend to run the setup steps below, a suitable Python version will be installed for you automatically.


#### System Requirements

Different models have different system requirements. Typically, RAM is the main limiting factor on most systems, but many functions can still operate with as little as 16 GB of RAM. However, it's important to check the specifications of the models your're using, as they have different sizes and requirements for memory and compute resources.

### 👨‍💻 Installation

To install the LLM-VM, just download this repository and uses pip with the following commands:

``` bash
> git clone https://github.com/anarchy-ai/LLM-VM.git
> cd LLM-VM
> ./setup.sh
```

The above bash script `setup.sh` only works for MacOs.

If you are on Linux you should do this:

```bash
> git clone https://github.com/anarchy-ai/LLM-VM.git
> cd LLM-VM
> python -m venv <name>
> source <name>/bin/activate
> python -m pip install -e ."[dev]"
```

#### One Last Step, almost there!
If you're using an OpenAI model,remember to set the `LLM_VM_OPENAI_API_KEY` environment
variable with your API key. 


### ✅ Generating Completions
Our LLM-VM enables you to work with popular LLMs locally in only 3 lines. Once installed (as shown), simply load your model and begin generating!



```python
# import our client
from llm_vm.client import Client

# Select which LLM you want to use, here we have OpenAI 
client = Client(big_model = 'chat_gpt')

# Put in your prompt and go!
response = client.complete(prompt = 'What is Anarchy?', context = '', openai_key = 'ENTER_YOUR_API_KEY')
print(response)
# Anarchy is a political ideology that advocates for the absence of government...
```

### 🏃‍♀ Running LLMs Locally
```python
# import our client
from llm_vm.client import Client

# Select the LlaMA 2 model
client = Client(big_model = 'llama2')

# Put in your prompt and go!
response = client.complete(prompt = 'What is Anarchy?', context = '')
print(response)
# Anarchy is a political philosophy that advocates no government...
```


### 😎 Supported Models
Select from the following models
```python

Supported_Models = ['chat_gpt','gpt','neo','llama2','bloom','opt','pythia']
```




### ☯ Picking Different Models
The default model sizes in LLM-VM aim to make experimenting with LLMs accessible to all. However, large parameter models will perform much better if you have enough memory. For instance, if you intend to use large and small models for your teacher and student, and you have sufficient RAM: 


```python
# import our client
from llm_vm.client import Client

# Select the LlaMA model
client = Client(big_model = 'neo', big_model_config={'model_uri':'EleutherAI/gpt-neox-20b'}, 
                small_model ='neo', small_model_config={'model_uri':'EleutherAI/gpt-neox-125m'})

# Put in your prompt and go!
response = client.complete(prompt = 'What is Anarchy?', context = '')
print(response)
# Anarchy is a political philosophy that advocates no government...
```

To explore different memory usage and parameter counts for each model family, refer to the tables in [model_uri_tables](./model_uri_tables.md)


### 🛠 Tool Usage

There are two agents: FLAT and REBEL. 

To run them separately naviage to `src/llm_vm/agents/<AGENT_FOLDER>` and execute the `agent.py` file. 

Alternatively, to use a  simple interface and select an agent to run from the CLI, run  `src/llm_vm/agents/agent_interface.py` file 
and follow the command prompt instructions. 


## 🩷 Contributing 🩷

We welcome contributors!  To get started is to join our active [discord community](https://discord.anarchy.ai).  Otherwise here are some ways to contribute and get paid:

### Jobs

- We're seeking skilled hackers. Show your ability to tackle tough problems and reach out! The best way to join us is by submitting pull requests for open issues. 
- Then, you can apply directly here https://forms.gle/bUWDKW3cwZ8n6qsU8

### Bounty

We offer bounties for closing specific tickets! Look at the ticket labels to see how much the bounty is.  To get started, [join the discord and read the guide](https://discord.com/channels/1075227138766147656/1147542772824408074)

## 🙏 Acknowledgements 🙏

- **Matthew Mirman** - CEO
  - GitHub: [@mmirman](https://github.com/mmirman)
  - LinkedIn: [@matthewmirman](https://www.linkedin.com/in/matthewmirman/)
  - Twitter: [@mmirman](https://twitter.com/mmirman)
  - Website: [mirman.com](https://www.mirman.com)

- **Victor Odede** - Undoomer
  - GitHub: [@VictorOdede](https://github.com/VictorOdede)
  - LinkedIn: [@victor-odede](https://www.linkedin.com/in/victor-odede-aaa907114/)

- **Abhigya Sodani** - Research Intern
  - GitHub: [@abhigya-sodani](https://github.com/abhigya-sodani)
  - LinkedIn: [@abhigya-sodani](https://www.linkedin.com/in/abhigya-sodani-405918160/)

- **Carter Schonwald** - Fearless Contributor
  - GitHub: [@cartazio](https://github.com/cartazio)
  - LinkedIn: [@carter-schonwald](https://www.linkedin.com/in/carter-schonwald-aa178132/)
 
- **Kyle Wild** - Fearless Contributor
  - GitHub: [@dorkitude](https://github.com/dorkitude)
  - LinkedIn: [@kylewild](https://www.linkedin.com/in/kylewild/)

- **Aarushi Banerjee** - Fearless Contributor
  - GitHub: [@AB3000](https://github.com/AB3000)
  - LinkedIn: [@ab99](https://www.linkedin.com/in/ab99/)

- **Andrew Nelson** - Fearless Contributor
  - GitHub: [@ajn2004](https://github.com/ajn2004)
  - LinkedIn: [@ajnelsnyc](https://www.linkedin.com/in/ajnelsnyc/)
    
## License

[MIT License](LICENSE)
