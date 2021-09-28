import torch
import numpy as np
from joblib import Parallel, delayed

# tokenized DSVA df
def multi_prcs(input_ids, token_types, tokenizer):
    valued_token_type_ids = []
 
    for idcs, token_type in zip(input_ids, token_types):
        def decode_transform(idx, n_digits, is_decimal):
            try:
                victim = tokenizer.decode(idcs[idx])
            except IndexError:
                if is_decimal:
                    return 0
                else:
                    return 2

            if victim.isdigit():
                if is_decimal:
                    digit = n_digits
                    decode_transform(idx + 1, n_digits + 1, is_decimal)
                else:
                    digit = decode_transform(idx + 1, n_digits + 1, is_decimal)
            elif victim == '.':
                if is_decimal:
                    decode_transform(idx + 1, n_digits = 0, is_decimal = False)
                    return 0
                else:
                    decode_transform(idx + 1, n_digits = 8, is_decimal = True)
                    return 2
            else:
                if is_decimal:
                    decode_transform(idx + 1, n_digits = 0, is_decimal = False)
                    return 0
                else:
                    decode_transform(idx + 1, n_digits = 0, is_decimal = False)
                    return 2

            try:
                token_type[idx] = torch.LongTensor([digit])
            except:
                breakpoint()
            return (digit + 1)

        decode_transform(idx = 0, n_digits = 0, is_decimal = False)

        valued_token_type_ids.append(token_type.tolist())

        #pbar.update(1)
    return valued_token_type_ids

def digit_place_embed(data, tokenizer):
    Queue=zip(data['input_ids'], data['token_type_ids'])
    print("[Processing] transform token_type_ids with value encoding")
    valued_token_type_ids_list = Parallel(n_jobs=-1, verbose=10)(delayed(multi_prcs)(input_ids, token_types, tokenizer) for input_ids, token_types in Queue) 
    print("[End] transform np.array into DataFrame")
    array_token_type = np.array(valued_token_type_ids_list)
    for i in range(len(data)):
        data['token_type_ids'][i] = array_token_type[i]
    
    return data