# Created by Andrew Silva on 2/21/19
import torch.nn as nn
import torch
import numpy as np


# embedding module
class EmbeddingModule(nn.Module):
    """
    embedding class (allows us to access parameters directly)
    """

    def __init__(self, bayesian_embedding_dim):
        super(EmbeddingModule, self).__init__()
        self.embedding = nn.Parameter(torch.randn(bayesian_embedding_dim))

    def forward(self):
        """
        doesn't do anything
        :return:
        """
        return


class ProLoNet(nn.Module):
    def __init__(self,
                 input_dim,
                 weights,
                 comparators,
                 leaves,
                 selectors=None,
                 output_dim=None,
                 bayesian_embedding_dim=None,
                 alpha=1.0,
                 freeze_alpha=False,
                 is_value=False,
                 use_gpu=False,
                 vectorized=True,
                 attention=False):
        super(ProLoNet, self).__init__()
        """
        Initialize the ProLoNet, taking in premade weights for inputs to comparators and sigmoids
        Alternatively, pass in None to everything except for input_dim and output_dim, and you will get a randomly
        initialized tree. If you pass an int to leaves, it must be 2**N so that we can build a balanced tree
        :param input_dim: int. always required for input dimensionality
        :param weights: None or a list of lists, where each sub-list is a weight vector for each node
        :param comparators: None or a list of lists, where each sub-list is a comparator vector for each node
        :param leaves: None, int, or truple of [[left turn indices], [right turn indices], [final_probs]]. If int, must be 2**N
        :param output_dim: None or int, must be an int if weights and comparators are None
        :param alpha: int. Strictness of the tree, default 1
        :param dphi: for linear sig, determines width of middle part. defaults to correct value of 1.31696
        :param epsi: for linear sig, determines height of middle part. defaults to correct value of 0.2
        :param tail_slope: for linear sig, determines slope of outside parts. defaults to 0.1, which is close-ish
        :param is_value: if False, outputs are passed through a Softmax final layer. Default: False
        :param use_gpu: is this a GPU-enabled network? Default: False
        :param vectorized: Use a vectorized comparator? Default: True
        :param attention: Use softmax attention instead of the selector parameter? Default: False
        """
        self.use_gpu = use_gpu
        self.vectorized = vectorized
        self.leaf_init_information = leaves
        self.bayesian_embedding_dim = bayesian_embedding_dim
        self.freeze_alpha = freeze_alpha

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = None
        self.comparators = None
        self.bayesian_embedding = None
        self.selector = None
        self.attention = None

        self.init_bayesian_embedding()
        self.init_comparators(comparators)
        self.init_weights(weights)
        self.init_alpha(alpha)
        if self.vectorized:
            if not attention:
                self.init_selector(selectors, weights)
            else:
                self.init_attention()
        self.init_paths()
        self.init_leaves()
        self.added_levels = nn.Sequential()

        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.is_value = is_value

    def init_bayesian_embedding(self):
        if self.bayesian_embedding_dim is not None:
            self.input_dim += self.bayesian_embedding_dim
            self.bayesian_embedding = EmbeddingModule(self.bayesian_embedding_dim)

    def init_comparators(self, comparators):
        if comparators is None:
            comparators = []
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            else:
                depth = 4
            for level in range(depth):
                for node in range(2**level):
                    if self.vectorized:
                        comparators.append(np.ones(self.input_dim)*(1.0/self.input_dim))
                    else:
                        comparators.append(np.array([1.0/self.input_dim]))
        new_comps = torch.Tensor(comparators)
        new_comps.requires_grad = True
        if self.use_gpu:
            new_comps = new_comps.cuda()
        self.comparators = nn.Parameter(new_comps)

    def init_weights(self, weights):
        if weights is None:
            weights = []
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            else:
                depth = 4
            for level in range(depth):
                for node in range(2**level):
                    weights.append(np.random.rand(self.input_dim))  # *(1.0/self.input_dim))

        new_weights = torch.Tensor(weights)
        new_weights.requires_grad = True
        if self.use_gpu:
            new_weights = new_weights.cuda()
        self.layers = nn.Parameter(new_weights)

    def init_alpha(self, alpha):
        self.alpha = torch.Tensor([alpha])
        if self.use_gpu:
            self.alpha = self.alpha.cuda()
        if not self.freeze_alpha:
            self.alpha.requires_grad = True
            self.alpha = nn.Parameter(self.alpha)

    def init_selector(self, selector, weights):
        if selector is None:
            if weights is None:
                selector = np.ones(self.layers.size())*(1.0/self.input_dim)
            else:
                selector = []
                for layer in self.layers:
                    new_sel = np.zeros(layer.size())
                    max_ind = torch.argmax(torch.abs(layer)).item()
                    new_sel[max_ind] = 1
                    selector.append(new_sel)
        selector = torch.Tensor(selector)
        selector.requires_grad = True
        if self.use_gpu:
            selector = selector.cuda()
        self.selector = nn.Parameter(selector)

    def init_attention(self):
        attn = torch.Tensor(np.random.rand(*self.layers.size()))
        if self.use_gpu:
            attn = attn.cuda()
        attn.requires_grad = True
        self.attention = nn.Parameter(attn)

    def init_paths(self):
        if type(self.leaf_init_information) is list:
            left_branches = torch.zeros((len(self.layers), len(self.leaf_init_information)))
            right_branches = torch.zeros((len(self.layers), len(self.leaf_init_information)))
            for n in range(0, len(self.leaf_init_information)):
                for i in self.leaf_init_information[n][0]:
                    left_branches[i][n] = 1.0
                for j in self.leaf_init_information[n][1]:
                    right_branches[j][n] = 1.0
        else:
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            elif self.leaf_init_information is None:
                depth = 4
            left_branches = torch.zeros((2 ** depth - 1, 2 ** depth))
            for n in range(0, depth):
                row = 2 ** n - 1
                for i in range(0, 2 ** depth):
                    col = 2 ** (depth - n) * i
                    end_col = col + 2 ** (depth - 1 - n)
                    if row + i >= len(left_branches) or end_col >= len(left_branches[row]):
                        break
                    left_branches[row + i, col:end_col] = 1.0
            right_branches = torch.zeros((2 ** depth - 1, 2 ** depth))
            left_turns = np.where(left_branches == 1)
            for row in np.unique(left_turns[0]):
                cols = left_turns[1][left_turns[0] == row]
                start_pos = cols[-1] + 1
                end_pos = start_pos + len(cols)
                right_branches[row, start_pos:end_pos] = 1.0
        left_branches.requires_grad = False
        right_branches.requires_grad = False
        if self.use_gpu:
            left_branches = left_branches.cuda()
            right_branches = right_branches.cuda()
        self.left_path_sigs = left_branches
        self.right_path_sigs = right_branches

    def init_leaves(self):
        if type(self.leaf_init_information) is list:
            new_leaves = [leaf[-1] for leaf in self.leaf_init_information]
        else:
            new_leaves = []
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            else:
                depth = 4

            last_level = np.arange(2**(depth-1)-1, 2**depth-1)
            going_left = True
            leaf_index = 0
            self.leaf_init_information = []
            for level in range(2**depth):
                curr_node = last_level[leaf_index]
                turn_left = going_left
                left_path = []
                right_path = []
                while curr_node >= 0:
                    if turn_left:
                        left_path.append(int(curr_node))
                    else:
                        right_path.append(int(curr_node))
                    prev_node = np.ceil(curr_node / 2) - 1
                    if curr_node // 2 > prev_node:
                        turn_left = False
                    else:
                        turn_left = True
                    curr_node = prev_node
                if going_left:
                    going_left = False
                else:
                    going_left = True
                    leaf_index += 1
                new_probs = np.random.uniform(0, 1, self.output_dim)  # *(1.0/self.output_dim)
                self.leaf_init_information.append([sorted(left_path), sorted(right_path), new_probs])
                new_leaves.append(new_probs)

        labels = torch.Tensor(new_leaves)
        if self.use_gpu:
            labels = labels.cuda()
        labels.requires_grad = True
        self.action_probs = nn.Parameter(labels)

    def forward(self, input_data, embedding_list=None):
        if self.bayesian_embedding is not None:
            if embedding_list is not None:
                input_temp = [torch.cat((input_data[0], self.bayesian_embedding))]
                for e_ind, embedding in enumerate(embedding_list):
                    embedding = torch.Tensor(embedding)
                    if self.use_gpu:
                        embedding = embedding.cuda()
                    input_temp.append(torch.cat((input_data[e_ind+1], embedding)))
                input_data = torch.stack(input_temp)
            else:
                input_data = torch.cat((input_data, self.bayesian_embedding.embedding.expand(input_data.size(0),
                                                                                   *self.bayesian_embedding.embedding.size())), dim=1)

        input_data = input_data.t().expand(self.layers.size(0), *input_data.t().size())

        input_data = input_data.permute(2, 0, 1)
        comp = self.layers.mul(input_data)
        if not self.vectorized:
            comp = comp.sum(dim=2).unsqueeze(-1)
        comp = comp.sub(self.comparators.expand(input_data.size(0), *self.comparators.size()))
        comp = comp.mul(self.alpha)
        sig_vals = self.sig(comp)
        if self.vectorized:
            # sig_vals = self.single_sigmoid_from_vector(sig_vals)  # This was the old attempt at averaging out sigmoids. it isn't correct.
            if self.attention is None:
                s_temp_main = self.selector

                # s_temp_main = s_temp_main.pow(2)
                # sum_div = s_temp_main.sum(dim=0).detach().clone()
                # sum_div[sum_div == 0] = 1
                # s_temp_main = s_temp_main.div(sum_div)

                selector_subber = self.selector.detach().clone()
                selector_divver = self.selector.detach().clone()
                selector_subber[np.arange(0, len(selector_subber)), selector_subber.max(dim=1)[1]] = 0
                selector_divver[selector_divver == 0] = 1
                s_temp_main = s_temp_main.sub(selector_subber)
                s_temp_main = s_temp_main.div(selector_divver)

                s_temp_main = s_temp_main.expand(input_data.size(0), *self.selector.size())

                sig_vals = sig_vals.mul(s_temp_main)
                sig_vals = sig_vals.sum(dim=2)
            else:
                attn = self.softmax(torch.mul(self.attention, input_data))
                sig_vals = sig_vals.mul(attn)
                sig_vals = sig_vals.sum(dim=2)

        sig_vals = sig_vals.view(input_data.size(0), -1)

        if not self.use_gpu:
            one_minus_sig = torch.ones(sig_vals.size()).sub(sig_vals)
        else:
            one_minus_sig = torch.ones(sig_vals.size()).cuda().sub(sig_vals)

        left_path_probs = self.left_path_sigs.t()
        right_path_probs = self.right_path_sigs.t()
        left_path_probs = left_path_probs.expand(input_data.size(0), *left_path_probs.size()) * sig_vals.unsqueeze(1)
        right_path_probs = right_path_probs.expand(input_data.size(0), *right_path_probs.size()) * one_minus_sig.unsqueeze(1)
        left_path_probs = left_path_probs.permute(0, 2, 1)
        right_path_probs = right_path_probs.permute(0, 2, 1)

        # We don't want 0s to ruin leaf probabilities, so replace them with 1s so they don't affect the product
        left_filler = torch.zeros(self.left_path_sigs.size())
        left_filler[self.left_path_sigs == 0] = 1
        right_filler = torch.zeros(self.right_path_sigs.size())
        if self.use_gpu:
            left_filler = left_filler.cuda()
            right_filler = right_filler.cuda()
        right_filler[self.right_path_sigs == 0] = 1

        left_path_probs = left_path_probs.add(left_filler)
        right_path_probs = right_path_probs.add(right_filler)

        probs = torch.cat((left_path_probs, right_path_probs), dim=1)
        probs = probs.prod(dim=1)

        actions = probs.mm(self.action_probs)

        if not self.is_value:
            return self.softmax(actions)
        else:
            return actions

    def set_bayesian_embedding(self, embedding):
        """
        sets embedding into BNN
        :param embedding:
        :return:
        """
        for n, i in enumerate(embedding):
            self.bayesian_embedding.embedding.data[n].fill_(i)

        if self.use_gpu:
            self.bayesian_embedding = self.bayesian_embedding.cuda()
    #
    def get_bayesian_embedding(self):
        """
        gets embedding inside BNN
        :return:
        """
        return self.bayesian_embedding.embedding


    # this is used for DDT
    # def get_bayesian_embedding(self):
    #     if self.bayesian_embedding is not None:
    #         return self.bayesian_embedding.data.cpu().numpy()
    #     else:
    #         return None

    def reset_bayesian_embedding(self):
        self.init_bayesian_embedding()
