import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

def tokenize():
    #Load a pretrained model and a tokenizer using HuggingFace
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # tokenize a sammple sentence
    text = "The cow jumped over the moon"
    print(text)
    inputs = tokenizer(text, return_tensors="pt")

    # print the tokenized input
    print(inputs)

    print(inputs["input_ids"]) # print the token ids


    # decode the tokenized input
    decoded_text = tokenizer.decode(inputs["input_ids"][0])
    print(decoded_text)

    print([(id, tokenizer.decode([id])) for id in inputs["input_ids"][0]])# print the token ids and their corresponding text

    return inputs

if __name__ == "__main__":
    inputs = tokenize()