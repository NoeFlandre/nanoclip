from transformers import AutoTokenizer

class CLIPTokenizer:
    def __init__(self):
        # We load the exact same tokenizer used by OpenAI to avoid headaches
        # fast=True is using the Rust implementation which is much faster
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", fast=True)

    def __call__(self, text, max_length=77):
        """
        Takes a string (text) as input and converts it to a tensor of integers
        """

        output = self.tokenizer(
            text,
            padding="max_length", # pad short sentences with zeroes to reach max_length
            truncation=True, # cut off long sentences at max_length
            max_length=max_length, 
            return_tensors="pt"

        )

        # the tokenizer returns a dictionary : {'input_ids : ... , 'attention_mask': ...}
        # we just want the input_ids
        return output["input_ids"][0]