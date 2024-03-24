from datasets import load_dataset

# Carregar o dataset do Hugging Face Hub
dataset = load_dataset(
    "portuguese-benchmark-datasets/proverbs",
    name="proverbs_ptbr",
    token='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'
)

print(dataset)

for data in dataset["train"]:
    print(data)
    break