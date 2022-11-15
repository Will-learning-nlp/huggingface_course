from transformers import pipeline

camebert_fill_mask = pipeline("fill-mask", model="camembert-base")
results = camebert_fill_mask("Le camembert est <mask> :)")
print(results)

# [{'score': 0.49091100692749023, 'token': 7200, 'token_str': 'délicieux', 'sequence': 'Le camembert est délicieux :)'},
# {'score': 0.10556945204734802, 'token': 2183, 'token_str': 'excellent', 'sequence': 'Le camembert est excellent :)'},
# {'score': 0.03453319892287254, 'token': 26202, 'token_str': 'succulent', 'sequence': 'Le camembert est succulent :)'},
# {'score': 0.03303125500679016, 'token': 528, 'token_str': 'meilleur', 'sequence': 'Le camembert est meilleur :)'},
# {'score': 0.03007635846734047, 'token': 1654, 'token_str': 'parfait', 'sequence': 'Le camembert est parfait :)'}]
'''
from transformers import CamembertTokenizer, CamembertForMaskedLM
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForMaskedLM.from_pretrained("camembert-base")
'''

# 更好的方式，使用 Auto*classes

from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")