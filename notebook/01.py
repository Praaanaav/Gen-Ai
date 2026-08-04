import tiktoken

tokenizer = tiktoken.encoding_for_model("gpt-4");
# print(tokenizer)

text = "Hey How Are You"
tokenId = tokenizer.encode(text)
print(tokenId)

result = tokenizer.decode( [19182, 2650, 8886, 1472])
print(result)
