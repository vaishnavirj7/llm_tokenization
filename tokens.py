import transformers, torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def tokenize(tokenizer):
    #Load a pretrained model and a tokenizer using HuggingFace
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

    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :] #logits of the last token
        # what are logits - logits are the raw, unnormalized scores output by the model before applying softmax
        probabilities = torch.nn.functional.softmax(logits, dim=-1) # apply softmax to get probabilities

    return inputs, probabilities



def show_next_token_choices(probabilities, tokenizer, top_n=10):
    # get the top n token choices
    top_n_probabilities, top_n_indices = torch.topk(probabilities, top_n)
    print("Top {} token choices:".format(top_n))
    for i in range(top_n):
        print("Token ID: {}, Probability: {:.4f}, Text: {}".format(
            top_n_indices[0][i].item(),
            top_n_probabilities[0][i].item(),
            tokenizer.decode([top_n_indices[0][i]]),
        ))
    


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    inputs, probabilities = tokenize(tokenizer)
    show_next_token_choices(probabilities, tokenizer=tokenizer)