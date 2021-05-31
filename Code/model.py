
import torch
import torch.nn as nn
import torch.nn.functional as F


class MainTaskModule(nn.Module):
    def __init__(self, dim, state_dim, evt_count):
        super(MainTaskModule, self).__init__()
        self.dim = dim * 3
        self.hid2state = nn.Linear((dim + state_dim) * 2, state_dim)
        self.state2prob = nn.Linear(state_dim, evt_count + 1)

    def forward(self, word_vec, evt_vec, latest_info, memory):
        inp = torch.cat([word_vec, evt_vec, latest_info, memory], 1)
        state = torch.tanh(self.hid2state(inp))
        prob = F.softmax(self.state2prob(state), dim=1)
        return state, prob


class SubTaskModule(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(SubTaskModule, self).__init__()
        self.input_dim = dim * 3
        self.action_dim = dim
        self.envinfo_dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = 2
        self.num_directions = 2
        self.tagset_size = 3
        self.hidden = self.init_hidden()

        self.biLSTM_layer_in = nn.LSTM(input_size=self.input_dim * 2 + self.action_dim, hidden_size=hidden_dim,
                                       num_layers=self.num_layers, bidirectional=True)
        self.biLSTM_layer_out = nn.LSTM(input_size=self.tagset_size, hidden_size=self.tagset_size * 2,
                                        num_layers=self.num_layers, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * self.num_layers, self.tagset_size)
        self.tag2env_info = nn.Linear(self.tagset_size * 2, self.envinfo_dim)

    def init_hidden(self):
        return (torch.zeros(self.num_layers * self.num_directions, 1, self.hidden_dim),
                torch.zeros(self.num_layers * self.num_directions, 1, self.hidden_dim))

    def forward(self, sentence, word_t, act_t):
        new_sentence = torch.rand(len(sentence), 1, self.input_dim * 2 + self.action_dim)
        for i in range(len(sentence)):
            new_word = torch.cat([sentence[i], word_t, act_t], 1)
            new_sentence[i].copy_(new_word)
        lstm_out, self.hidden = self.biLSTM_layer_in(new_sentence, self.hidden)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.softmax(tag_space, dim=2)
        _, (h_0, _) = self.biLSTM_layer_out(tag_scores)
        env_info = self.tag2env_info(h_0[0][0])
        return tag_space, tag_scores, env_info.unsqueeze(0)


class Model(nn.Module):
    def __init__(self, lr, dim, state_dim, hidden_dim, evt_count, wv):
        super(Model, self).__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.evt_count = evt_count
        self.MainTaskModule = MainTaskModule(dim, state_dim, evt_count)
        self.SubTaskModule = SubTaskModule(dim, hidden_dim)
        wvTensor = torch.FloatTensor(wv)
        self.wordvector = nn.Embedding(wvTensor.size(0), wvTensor.size(1)).cuda()
        self.wordvector.weight = nn.Parameter(wvTensor)
        self.event_vector = nn.Embedding(evt_count + 1, 100).cuda()
        self.argument_vector = nn.Embedding(3, 100).cuda()

    def sample(self, prob, training, preactions, position):
        if not training:
            return torch.max(prob, 1).indices
        elif preactions is not None:
            return torch.tensor(preactions[position], requires_grad=True)
        else:
            return torch.multinomial(prob, 1).view(1)

    def forward(self, mode, text, preactions=None):
        textin = torch.cuda.LongTensor(text)
        wvs = self.wordvector(textin)
        top_action, top_actprob, tag_scores_set, latest_info_set = [], [], [], []
        # main task ---------------------------------------------------------
        training = True if "test" not in mode else False
        mem = torch.randn(self.state_dim).cuda()
        rel_action = torch.tensor([0]).cuda()
        latest_info = torch.zeros(1, 100)
        for word_t in range(len(text)):
            mem, prob = self.MainTaskModule(wvs[word_t], self.event_vector(rel_action)[0].unsqueeze(0), latest_info, mem)
            action = self.sample(prob, training, preactions, word_t)
            if action.data[0] != 0:
                rel_action = action
            actprob = prob[0][action]
            top_action.append(action.cpu().data[0])
            if not training:
                top_actprob.append(actprob.cpu().data[0])
            else:
                top_actprob.append(actprob)
            # sub task -----------------------------------------------------
            if "AD" in mode and action.data[0] > 0:
                act_t = self.event_vector(action.data[0]).unsqueeze(0)
                wd_t = wvs[word_t]
                tag_space, tag_scores, latest_info = self.SubTaskModule(wvs, wd_t, act_t)
                latest_info_set.append(latest_info)
                tag_scores_set.append(tag_scores)
        actions = [top_action[i].data.numpy() for i in range(len(top_action))]
        tags = []
        for i in range(len(tag_scores_set)):
            tag = torch.max(tag_scores_set[i].squeeze(), 1).indices
            tags.append(tag.data.numpy())
        return actions, top_actprob, tags

