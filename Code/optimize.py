
import torch


def calcF1(acc, cnt, tot, beta=1.0):
    if cnt == 0 or tot == 0:
        return 0
    precision = float(acc) / float(cnt)
    recall = float(acc) / float(tot)
    if precision + recall < 1e-5:
        return 0
    return (1+beta*beta) * precision * recall / (beta*beta*precision + recall)


def calc_acc(actions, tags, gold_labels, mode):
    acc, cnt, tot = 0, 0, len(gold_labels)
    for label in gold_labels:
        typ, tag = label['type'], label['tags']
        j, ok = 0, 0
        for i in range(len(actions)):
            if actions[i] == typ and typ > 0 and ok == 0:
                match = 1
                if "AD" in mode:
                    for k in range(len(tags[j])):
                        if tag[k] == 1 and tags[j][k] != 1:
                            match = 0
                        if tag[k] != 1 and tags[j][k] == 1:
                            match = 0
                        if tag[k] == 2 and tags[j][k] != 2:
                            match = 0
                        if tag[k] != 2 and tags[j][k] == 2:
                            match = 0
                if match == 1:
                    ok = 1
            if actions[i] > 0:
                j += 1
                cnt += 1
        acc += ok
    cnt //= tot
    return acc, tot, cnt


def calcReward(actions, tags, gold_labels):
    length = len(actions)
    r = [0. for i in range(length)]
    eve_set = set()
    for item in gold_labels:
        eve_set.add(item["type"])
    for i in range(length):
        if actions[i] > 0:
            base = 1 if actions[i] in eve_set else -1
            for label in gold_labels:
                if label["type"] == actions[i]:
                    for t in range(length):
                        if label["tags"][t] == tags[i][t]:
                            r[i] += 1
            r[i] *= base
    return r


def calcGrad(actions, top_actprob, rewards):
    length = len(actions)
    decay_r = 0.
    avg = 0.
    grads = torch.tensor([0.], requires_grad=True)
    for i in range(length):
        avg = (avg + rewards[i]) / (i + 1)
        decay_r = decay_r * 0.95 + rewards[i]
        to_grad = -torch.log(top_actprob[i])
        to_grad *= torch.tensor([decay_r - avg], requires_grad=True)
        grads = grads + to_grad
    return grads


def rule_labels(sent, gold_labels):
    length = len(gold_labels[0]['tags'])
    actions = [0 for i in range(length)]
    tags = [[] for i in range(length)]
    for label in gold_labels:
        trig, typ, tag = label['trigger'], label['type'], label['tags']
        trig_idx = sent.split(' ').index(trig)
        actions[trig_idx] = typ
        tags[trig_idx] = tag
    return actions, tags


def optimize(model, actions, top_actprob, tags, gold_labels):
    reward = calcReward(actions, tags, gold_labels)
    grads = torch.tensor([0.], requires_grad=True).cuda()
    grads += calcGrad(actions, top_actprob, reward)
    loss = grads.cpu().data[0]
    grads.backward()
    return loss


def optimize_round(model, actions, top_actprob, tags, gold_labels, mode):
    sample_round = len(actions)
    loss = .0
    for i in range(sample_round):
        loss += optimize(model, actions, top_actprob, tags, gold_labels)
    return loss / sample_round

