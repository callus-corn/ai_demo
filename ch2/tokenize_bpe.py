from tokenizers import Tokenizer


tokenizer = Tokenizer.from_file("bpe_1000.json")
text = "We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness."
encoding = tokenizer.encode(text)
print("Tokens:", encoding.tokens)
print("IDs:", encoding.ids)
print("len(Tokens)", len(encoding.tokens))
print("len(IDs)", len(encoding.ids))
