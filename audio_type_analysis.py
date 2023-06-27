import json
import os

AMT_testset = json.load(open('data/AMT_test_set.json', 'r'))
AMT_trainset = json.load(open('data/greatesthit_train_2.00.json', 'r'))
AMT_validset = json.load(open('data/greatesthit_val_2.00.json', 'r'))
greatest_testset = json.load(open('data/greatesthit_test_2.00.json', 'r'))
record_dir = 'data/greatesthit/greatesthit_processed'
match_dict = {}
type_dict = {}
overall_cnt = {True: 0, False: 0}

def process_action(name, st):
    record_path = os.path.join(record_dir, name, 'hit_record.json')
    record = json.load(open(record_path, 'r'))
    action_cnt = {}
    for (t, action) in record:
        if t >= st and t <= st + 2:
            _, act = action.split(' ')
            if act not in action_cnt.keys():
                action_cnt[act] = 0
            action_cnt[act] += 1
    return action_cnt


def process_type(name, st, drop_none=False):
    record_path = os.path.join(record_dir, name, 'hit_record.json')
    record = json.load(open(record_path, 'r'))
    action_cnt = {}
    for (t, action) in record:
        if 'None' in action:
            continue
        if t >= st and t <= st + 2:
            if action not in action_cnt.keys():
                action_cnt[action] = 0
            action_cnt[action] += 1
    return action_cnt


def check_match(cnt1, cnt2):
    type1 = list(cnt1.keys())
    if 'None' in type1:
        type1.remove('None')
    type2 = list(cnt2.keys())
    if 'None' in type2:
        type2.remove('None')
    for t in type1:
        if t not in type2:
            return False
    for t in type2:
        if t not in type1:
            return False
    return True


if __name__ == '__main__':
    if not os.path.exists('data/AMT_test_set_match_dict.json'):
        for target in AMT_testset:
            target_name, start_time = target.split('_')
            target_action_cnt = process_action(target_name, float(start_time))

            match_dict[target] = {}
            type_dict[target] = target_action_cnt

            for condition in AMT_testset[target]:
                cond_name, start_time = condition.split('_')
                cond_action_cnt = process_action(cond_name, float(start_time))

                match_dict[target][condition] = check_match(target_action_cnt, cond_action_cnt)
                overall_cnt[match_dict[target][condition]] += 1
                type_dict[condition] = cond_action_cnt
        json.dump(match_dict, open('data/AMT_test_set_match_dict.json', 'w'))

    print(len(greatest_testset))
    for video_idx in greatest_testset:
        name, idx = video_idx.split('_')
        time = float(idx) / 22050
        action_cnt = process_type(name, time)
        if len(action_cnt.keys()) == 1:
            type_dict[video_idx] = list(action_cnt.keys())[0]

    print(len(type_dict))
    json.dump(type_dict, open('data/greatesthit_test_2.00_single_type_only.json', 'w'))



