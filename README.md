# language-detector

## Application

App deployed on HuggingFace Spaces: https://huggingface.co/spaces/saadkh225/language_detector

## Task
Given an input text, output it's language in Latin script.

## Methodology

There are a couple of approaches possible:

1. **Traditional ML model building**

    Find a good dataset and create a model.

   *Pros* : should perform the best amongst all other options given the right dataset
   
   *Cons*: model training, maintenance

3. **LLM based**

    There are many LLM options, most trained in the English language but multilingual options are available.
    
    LLMs pre-trained on a few languages can display advanced multilingual capabilities.
    > Despite the fact that most LLMs are trained predominantly on English, multiple studies have demonstrated their capabilities in a variety of languages LLM options tested.
    > Source: [Don’t Trust ChatGPT when your Question is not in English: A Study of Multilingual Abilities and Types of LLMs](https://aclanthology.org/2023.emnlp-main.491.pdf)
    
    And indeed in my tests *the Mistral models did demonstrate this*. 
    
    **LLMs testing methodology**:
    
    * Proprietary LLM service (GPT-4o)
        * Zero shot prompting
        * Few shot prompting
    * Open-source LLM based (Mixtral, Llama)
        * Straightforward querying
        * RAG + Few shot prompting

4. **Managed solution like Google AI Edge Language detection (MediaPipe Studio)**

    Supports 110 languages.
   
    https://mediapipe-studio.webapps.google.com/demo/language_detector 


## Data

These are the most relevant datasets I found:

*language-identification*: 20 languages (https://huggingface.co/datasets/papluca/language-identification)

*open-lid-dataset* : 201 languages (https://huggingface.co/datasets/laurievb/open-lid-dataset)

## Prompts

Initial zero shot prompt:

```
Act as a language detector.
Which language is this: {text}?
Answer only with one word. Do not exceed one word.
```

Improved zero shot prompt:

```
You are an AI language detection assistant. You will be given a piece of text.
Identify the language of the given text. Answer ONLY with one word. Answer ONLY in the Latin script.
If you do not know the language, just say "I'm not sure". Don't try to make up an answer.

Text:

{text}

Language identified:
```

RAG + few shot prompt:

```
You are an AI language detection assistant.
Here are some examples of Text and its language in one word:

{context}

Provide the language of the given text. Answer ONLY with one word. Answer ONLY in the Latin script.
If you do not know the language, just say "I'm not sure". Don't try to make up an answer.
Text:

{input}

Language Identified:
```

## Results and Observations

* **Proprietary LLM service (GPT-4o)**
1. Manages to correctly identify many obscure languages with zero shot prompting but does not get all of them.
2. Few shot prompting will definitely rectify this.
3. My tests: https://chatgpt.com/share/9227f578-e35f-4559-99ce-2512b5323238

* **Zero shot prompting Open-source LLMs**
  
  > The following testing has been done on 100 samples randomly drawn from the dataset.

  * *language-identification dataset (20 languages)*
    * `Mistral-7B-Instruct-v0.3`
      1. Accuracy: 69%
      2. A lot of the results are not in Latin script -> prompt modification will solve this -> good increase in accuracy
      
    * `Meta-Llama-3-8B-Instruct`
      1. Accuracy: 97%
      2. This is somewhat unexpected given that Llama has been trained on English only -> Instruction finetuning + some inherent multilingual ability works for popular languages
      3. Few errors occured because of the model predicting a short code instead of the full word -> Add to instructions in the prompt
      
    * `Mixtral-8x7B-Instruct-v0.1`
      1. Accuracy: 77% (Much lower than expected)
      2. Errors are mainly due to result formatting issues, which occur due to the prompt structure -> Improve prompt and result formatting
      3. With these fixes we can expect the same performace as the Llama 3 model

   * *open-lid-dataset (200 languages)*
      * `Mixtral-8x7B-Instruct-v0.1`
        1. Accuracy: 44% (Expected)
        2. In some cases the output is close but didn't exactly match the labels :
            * Guaran**í** vs Guarani 
            * Eastern Panjabi vs Punjabi (Incorrect spelling in the dataset labels)
            * Slovenian vs Slovene
            * Southern Pasto vs Pashto (Incorrect spelling in the dataset labels)
        
        3. Modifying the dataset labels or perhaps creating a mapping for such cases could solve the problem.
        4. In other cases, the predicted language seems to be a close family language -> Few shot prompting should help the model to narrow down on the right language.
            
      * `Meta-Llama-3-8B-Instruct`
        1. Accuracy: 0
        2. Model is just unable to predict the language.
        3. This is expected and the Llama models are trained only on English, so knowledge of obscure languages is definitely not expected
        4. Few shot prompting should help

* **RAG + Few shot prompting Open-source LLMs**

  > VectorDB used: Pinecone

  * `Mixtral-8x7B-Instruct-v0.1` with 4 documents (samples)
      1. Accuracy: 56% (An improvement from 44% without RAG)
      2. In some cases the output is close but didn't match the labels:
          * Eastern Panjabi vs Punjabi (Incorrect spelling in the dataset labels)
          * Southern Pasto vs Pashto (Incorrect spelling in the dataset labels)
      3. In other cases, the predicted language seems to be a close family language -> more documents as context can help

  * `Meta-Llama-3-8B-Instruct` with 4 documents (samples)
      1. Accuracy: 51% (Huge improvement from 0% without RAG)
      2. Worse than Mixtral in this use case

  * `Mixtral-8x7B-Instruct-v0.1` with 6 documents (samples)
      1. Accuracy: 61%
      2. A 10% increase just by providing 2 additional examples per prompt

  * `Mixtral-8x7B-Instruct-v0.1` with 10 documents (samples)
      1. Accuracy: 63.5%
      2. The improvement (2.5%) is low, meaning providing more examples will not help


## Conclusion

| Approach      | Description | Results | Pros | Cons |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| Traditional ML model building      | Creating a model from scratch (preferably using BERT embeddings)       | - | Could be one of the best amongst all other options given the right dataset | Model training, maintenance, good data |
| Proprietary LLM service (GPT-4o)   | Using OpenAI service        | Manages to correctly identify many obscure languages with zero shot prompting. With few shot prompting, will the best performing option | No development time | Cost |
| Zero shot prompting Open-source LLMs | Using open sourced LLMs |  `Meta-Llama-3-8B-Instruct` works well for popular languages | Free LLM access, fine tuning possible |Development time, prompt tuning, maintenance, hosting |
| | | `Mixtral-8x7B-Instruct-v0.1` gives 44% accuracy on open-lid dataset (200 languages) | | |
| RAG + Few shot prompting Open-source LLMs | Using RAG to give the LLM familiar examples in the context | `Mixtral-8x7B-Instruct-v0.1` with 10 document retreival gives 63.5% accuracy on open-lid dataset (200 languages) | RAG can significantly improve performace with the right data | Additional maintenace of a vector database |
