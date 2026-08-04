import tiktoken

tokenizer = tiktoken.encoding_for_model("gpt-4");
# print(tokenizer)

text = "Hey How Are You"
print (text)
tokenId = tokenizer.encode(text)
print(tokenId)

# result_text = "I'm good what about you"
result = tokenizer.decode( [40, 2846, 1695, 1148, 922, 499])
print(result)
