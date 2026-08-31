from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


# 空のBPE tokenizerを作成
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 空白で事前分割
tokenizer.pre_tokenizer = Whitespace()

# 学習設定
trainer = BpeTrainer(
    vocab_size=1000,
    special_tokens=["[UNK]", "[PAD]"]
)

# 学習
files = ["Declaration_of_Independence.txt"]
tokenizer.train(files, trainer)

# 保存
tokenizer.save("bpe_1000.json")
