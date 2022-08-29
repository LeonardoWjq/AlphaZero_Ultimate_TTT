import pickle as pkl
import numpy as np
PROVEN_WIN = 1
AT_LEAST_DRAW = 0.5
PROVEN_DRAW = 0
AT_MOST_DRAW = -0.5
PROVEN_LOSS = -1


def make_tt():
    # Max key can be bounded by 6500
    table = [None]*6500
    with open('tt.pickle','wb') as fp:
        pkl.dump(table,fp)


def hash_func(state:dict):
    # X -> 2
    # O -> 0
    # Empty -> 1
    trans_board = state['inner'] + 1
    flattened = trans_board.flatten()
    key = 0
    for index, val in enumerate(flattened):
        key += index*val
    
    return int(key)

def equal_state(first:dict, second:dict):
        if  (first['inner'] != second['inner']).any():
            return False
        
        if first['current'] != second['current']:
            return False

        if first['winner'] != second['winner']:
            return False
        
        if first['previous'] != second['previous']:
            return False
        
        return True

def lookup(table, key:int, state:dict):
    record_list = table[key]
    if record_list is None:
        return None
    

    for record in record_list:
        stored_state = record[0]
        if equal_state(state, stored_state):
            # outcome is in record[1]
            return record[1]
    
    # did not find the state
    return None

def store(key:int, table, state:dict, outcome:int):
    entry = table[key]
    if entry is None:
        # declare a list of records
        table[key] = [[state, outcome]]
        return
    # entry is not None
    for record in entry:
        # found duplicate state
        if equal_state(record[0], state):
            # update the outcome if it is not proven yet
            if record[1] == AT_LEAST_DRAW:
                # at least draw and proven loss are
                # mutually exclusive
                assert outcome != PROVEN_LOSS
                if outcome == AT_MOST_DRAW:
                    # must be draw
                    record[1] = PROVEN_DRAW
                else:
                    # could be proven win or stay the same
                    record[1] = outcome
            elif record[1] == AT_MOST_DRAW:
                # at most draw and proven win are mutually
                # exclusive
                assert outcome != PROVEN_WIN
                if outcome == AT_LEAST_DRAW:
                    # must be draw
                    record[1] = PROVEN_DRAW
                else:
                    # could be proven loss or stay the same
                    record[1] = outcome
            return
    
    # the state is not present in the entry
    # add it to the list
    entry.append([state, outcome])
    return

def load():
    with open('tt.pickle','rb') as fp:
        return pkl.load(fp)

def save(table):
    with open('tt.pickle','wb') as fp:
        pkl.dump(table, fp)

def stats(table):
    statistics = {'total':0, 'proven win':0, 'at least draw':0, 'proven draw':0,
                  'at most draw':0, 'proven loss':0, 'depth distribution':None}
    entry_lengths = []
    depth_distribution = np.zeros(82)
    for entry in table:
        if entry is not None:
            entry_lengths.append(len(entry))
            for state, outcome in entry:
                depth = get_depth(state)

                depth_distribution[depth] += 1
                if outcome == PROVEN_WIN:
                    statistics['proven win'] += 1
                elif outcome == AT_LEAST_DRAW:
                    statistics['at least draw'] += 1
                elif outcome == PROVEN_DRAW:
                    statistics['proven draw'] += 1
                elif outcome == AT_MOST_DRAW:
                    statistics['at most draw'] += 1
                else:
                    statistics['proven loss'] += 1

                statistics['total'] += 1
        else:
            entry_lengths.append(0)

    mean_entry_length = np.mean(entry_lengths)
    std_entry_length = np.std(entry_lengths)
    statistics['mean entry length'] = mean_entry_length
    statistics['entry length std'] = std_entry_length
    statistics['depth distribution'] = depth_distribution
    
    
    return statistics

def get_depth(state:dict):
    inner_board = state['inner']
    total = 0
    for row in inner_board:
        for entry in row:
            if entry != 0:
                total+=1
    return total


def to_list(table, proof_only=False):
    out = []
    for record in table:
        if record:
            for entry in record:
                if proof_only:
                    if entry[1] in (PROVEN_WIN,PROVEN_DRAW,PROVEN_LOSS):
                        out.append(entry)
                else:
                    out.append(entry)
    
    return out

