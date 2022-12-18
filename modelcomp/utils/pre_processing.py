import numpy as np
import copy


def transform_data(user, desired_design, num_arms=3):    
    
    # copy user over
    user_dict = copy.deepcopy(user)

    assert len(user_dict['design']) == len(desired_design), "Designs have a different number of blocks."
    num_blocks = len(desired_design)
        
    # Obtain indices to swap blocks, ignoring orderings within blocks
    idxs_swap = list()
    for d in range(num_blocks):  # desired
        for c in range(num_blocks):  # current
            # if sorted blocks match, select block and r
            if sorted(user_dict['design'][c]) == sorted(desired_design[d]):  
                if c not in idxs_swap:
                    idxs_swap.append(c)
                    break

    # swap blocks of designs, choices and rewards
    user_dict['design'] = np.array(user_dict['design'])[np.array(idxs_swap)].tolist()
    user_dict['choices'] = np.array(user_dict['choices'])[np.array(idxs_swap)].tolist()
    user_dict['rewards'] = np.array(user_dict['rewards'])[np.array(idxs_swap)].tolist()
    
    
    # note down whether or not we swapped blocks
    if idxs_swap[::-1] == list(range(len(user_dict['design']))):
        user_dict['swapped_blocks'] = True
    else:
        user_dict['swapped_blocks'] = False
        
    # Obtain the desired order of arms within each block
    idxs_swap_arms = list()
    for k in range(len(user_dict['design'])):
    
        idxs_tmp = list()
        for i in range(len(user_dict['design'][k])):
            for j in range(len(desired_design[k])):
                if user_dict['design'][k][i] == desired_design[k][j]:
                    if j not in idxs_tmp:
                        idxs_tmp.append(j)
                        break
        # reassign. // slicing produced wrong results
        arms_new = [0 for _ in range(len(user_dict['design'][k]))]
        for a in range(len(user_dict['design'][k])):
            arms_new[idxs_tmp[a]] = user_dict['design'][k][a]
        user_dict['design'][k] = arms_new
        idxs_swap_arms.append(idxs_tmp)

    # Swap order of arms withing each block of choices
    choices_tmp = np.array(copy.deepcopy(user_dict['choices']))
    for k in range(len(user_dict['design'])):
    
        list_actions = list()
        for i in range(len(user_dict['design'][k])):
            idxs_action = np.argwhere(
                np.array(user_dict['choices'][k]) == i).reshape(-1)
            list_actions.append(idxs_action)

        for i in range(len(user_dict['design'][k])):
            if len(list_actions[i]) == 0:
                continue
            elif len(list_actions[i]) == 1:
                choices_tmp[k][list_actions[i][0]] = idxs_swap_arms[k][i]
            else:
                choices_tmp[k][list_actions[i]] = idxs_swap_arms[k][i]
    user_dict['choices'] = choices_tmp.tolist()
    
    return user_dict
