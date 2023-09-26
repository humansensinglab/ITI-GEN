import os
import clip
import warnings
import numpy as np
import itertools as it
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.image_dataset import ImgDataset

from utils import split_attr_list, get_dataset_for_attribute, get_category_for_attribute

class FairToken(nn.Module):
    """
    K_m * 3 * 768 parameters
    """
    def __init__(self, num_basis=2, token_length=3, emb_dim=768):
        super().__init__()
        self.para = nn.Parameter(torch.zeros(num_basis, token_length, emb_dim), requires_grad=True)

    def forward(self):
        return self.para

class ITI_GEN(object):
    """
    Construct iti-gen model and training pipeline
    """
    def __init__(self, args):
        self.args = args

        # split attribute name
        self.attr_list, self.attr_num = split_attr_list(self.args.attr_list)
        # get category name
        self.cate_list, self.cate_num_list = get_category_for_attribute(self.attr_list)

        # construct dataset and data loader
        self.init_dataset()
        self.init_dataloader()
        self.total_combination = int(np.prod(np.array(self.cate_num_list)))

        # setup clip models
        self.setup_clip_model()
        self.clip_layers_num = self.clip_model.transformer.layers

        # self.last_word_idx equals 6 in '<start> a headshot of a person <end>'
        self.tokenized_text_queries, self.last_word_idx = self.prompt_tokenization(self.args.prompt)

        self.setup_fairtoken_model()
        self.setup_optimizer()
        self.setup_scheduler()

        # get the original text prompt feature
        self.ori_text_feature_extraction()

        self.setup_index()

    def init_dataset(self):
        """
        init dataset for different attributes
        :return:
        """
        # a list of training dataset. eg, 400, 702, 1800 -> gender, skin_tone, age
        self.trainset = []

        for idx, attr_name in enumerate(self.attr_list):

            dataset_path = os.path.join(self.args.data_path, get_dataset_for_attribute(attr_name))
            attr_path = os.path.join(dataset_path, attr_name)

            # reference image path and labels
            label_type = [i for i in range(self.cate_num_list[idx])]
            refer_img_path = []
            for cate in self.cate_list[idx]:
                refer_img_path.append(os.path.join(attr_path, cate))

            self.trainset.append(ImgDataset(root_dir=refer_img_path, label=label_type, upper_bound=self.args.refer_size_per_category))

    def init_dataloader(self):
        """
        init dataloader for different attributes
        :return:
        """
        # a list of training dataloader for different attributes
        self.trainloader = []
        for i, dataset in enumerate(self.trainset):
            self.trainloader.append(DataLoader(
                dataset=dataset,
                batch_size=len(dataset)//self.args.steps_per_epoch,
                num_workers=1,
                shuffle=True,
                # drop the last one
                drop_last=True,
                pin_memory=True,
            ))

    def setup_clip_model(self):
        self.clip_model, _ = clip.load("ViT-L/14", device="cpu")
        self.clip_model.to(device=self.args.device)
        self.clip_model.eval()

    def prompt_tokenization(self, prompt):
        """
        tokenize the text prompt
        :return:
        """
        text_queries = [prompt] * self.total_combination
        tokenized_text_queries = clip.tokenize(text_queries).to(self.args.device)
        return tokenized_text_queries, tokenized_text_queries[-1].argmax()

    def setup_fairtoken_model(self):
        """
        Initialize the fairtoken models
        :return:
        """
        self.fairtoken_model = []
        for cate_num in self.cate_num_list:
            self.fairtoken_model.append(
                FairToken(num_basis=cate_num, token_length=self.args.token_length).to(device=self.args.device)
            )

    def setup_optimizer(self):
        self.optimizer_list = []
        for i in self.fairtoken_model:
            self.optimizer_list.append(
                optim.Adam(i.parameters(), lr=self.args.lr)
            )

    def setup_scheduler(self):
        self.scheduler_list = []
        for optimizer in self.optimizer_list:
            self.scheduler_list.append(
                optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[10], gamma=0.5, verbose=True)
            )

    def ori_text_feature_extraction(self):
        """
        get the text features for self.args.prompt/self.tokenized_text_queries
        for computing semantic consistency loss
        :return:
        """
        with torch.no_grad():
            xp = self.clip_model.token_embedding(self.tokenized_text_queries).type(self.clip_model.dtype)  # (108, 77, 768)
            self.text_queries_embedding = xp + self.clip_model.positional_embedding.type(self.clip_model.dtype)
            xp = self.text_queries_embedding.permute(1, 0, 2)
            for ll in range(self.clip_layers_num):
                xp = self.clip_model.transformer.resblocks[ll](xp)
            xp = xp.permute(1, 0, 2)
            xp = self.clip_model.ln_final(xp).type(self.clip_model.dtype)  # (108, 77, 768)
            xp_feat = xp[torch.arange(xp.shape[0]), self.tokenized_text_queries.argmax(dim=-1)] @ self.clip_model.text_projection  # 108*768
            self.ori_text_feature = xp_feat / xp_feat.norm(dim=1, keepdim=True) # (108, 768)

    def setup_index(self):
        """
        Eg. for 2*6*9 combinations, self.index should contain three lists (for each attribute).
        Each list contain the index that one attribute's values differ while the other attributes' values are the same.
        Eg. for 2nd list -> Skin_Tone, we should include the index that gender, age are the same, while only skin_tone value is different.
        Total self.tokenized_text_queries contain 108 combinations, as shown below:
        0: + 0 0    54: - 0 0
        1: + 0 1    55: - 0 1
        2: + 0 2    56: - 0 2
        3: + 0 3    57: - 0 3
        4: + 0 4    58: - 0 4
        5: + 0 5    59: - 0 5
        6: + 0 6    60: - 0 6
        7: + 0 7    61: - 0 7
        8: + 0 8    62: - 0 8
        9: + 1 0    63: - 1 0
        ..          ..
        53: + 5 8   107: - 5 8
        the second list should contain 18 lists, as shown below:
        [
            [0, 9, 18, 27, 36, 45],
            [1, 10, 19, 28, 37, 46],
            ...
            [8, 17, 26, 35, 44, 53],
            [54, 63, 72, 81, 90, 99],
            ...
            [62, 71, 80, 89, 98, 107]
        ]
        """
        # 1, 2, 12. The multiplication accumulation before the current attribute.
        pre_multi_accu = 1
        # 108, 54, 9
        nonpre_multi_accu = self.total_combination # 108
        # 54, 9, 1
        later_multi_accu= int(self.total_combination // self.cate_num_list[0])

        self.index = []
        for attr_idx in range(self.attr_num):
            if attr_idx >= 1:
                pre_multi_accu *= self.cate_num_list[attr_idx - 1]
                nonpre_multi_accu = int(nonpre_multi_accu/self.cate_num_list[attr_idx - 1])
                later_multi_accu = int(later_multi_accu/self.cate_num_list[attr_idx])

            each_attr_index = []
            for i1 in range(pre_multi_accu):
                for i2 in range(later_multi_accu):
                    # construct a tensor with the length equals attribute category's number. skin tone: 6, age: 9...
                    sample = []
                    for i3 in range(self.cate_num_list[attr_idx]):
                        sample.append(0 + i1*nonpre_multi_accu + i2*1 + i3*later_multi_accu)
                    each_attr_index.append(torch.LongTensor(sample))
            self.index.append(each_attr_index)

    def construct_fair_text_features(self, text_queries_embedding, last_word_idx):
        """
        insert fair_tokens to the original text queries' embeddings and obtain the
        corresponding text feature
        :return:
        """
        x = text_queries_embedding.detach() # (108, 77, 768)
        # add FairToken to the corresponding place
        # for 1st attr, replace from the last word index; for 2nd and later, add to them
        for i, each_index in enumerate(self.index):
            for index in each_index:
                if i == 0:
                    x[index, last_word_idx:last_word_idx + self.args.token_length, :] = (
                    self.fairtoken_model[i])()
                else:
                    x[index, last_word_idx:last_word_idx + self.args.token_length, :] += (
                    self.fairtoken_model[i])()
        x = x.permute(1, 0, 2)
        for ll in range(self.clip_layers_num):
            x = self.clip_model.transformer.resblocks[ll](x)
        x = x.permute(1, 0, 2)
        return self.clip_model.ln_final(x).type(self.clip_model.dtype)  # (108, 77, 768)

    def cos_loss(self, image_features, text_features, each_index, data_cls):
        """
        compute the cosine similarity loss, 1-cos(image, text)
        :return:
        """
        # cosine similarity logits
        logits_i = image_features @ text_features.t()  # (n_img, n_text) # bs * 108
        # for each image, we only try to maximize the cosine similarity between the image and corresponding text,
        # which should own the same category value for that attribute with image.
        # Eg, for skin tone, image1's label is type3, so for text1, text2, ..., text6. Only text3 should have skin tone type3,
        # so we mask all the other texts.
        temp_mask = torch.zeros(logits_i.size(0), logits_i.size(1) // len(each_index)).to(self.args.device)  # bs * 108/18 (6)
        range_index = torch.arange(logits_i.size(0)).long()
        temp_mask[range_index, data_cls[range_index]] = 1  # only 0 or 1
        # extend to the whole text
        mask = torch.zeros_like(logits_i)  # bs*108
        for index in each_index:
            mask[:, index] = temp_mask
        mask = mask.to(self.args.device)
        # conduct mask operation
        logits_i = mask * logits_i
        # make summation across all images
        logits_i = logits_i.sum(dim=0)  # 108
        mask = mask.sum(dim=0)  # 108
        for i in range(self.total_combination):
            if mask[i] != 0:
                logits_i[i] = 1 - logits_i[i] / mask[i]
            else:
                logits_i[i] = 0
        # make summation for the cosine similarity loss and divide by the index length
        return logits_i.sum() / len(each_index)

    def iti_gen_loss(self, image_features, text_features, cate_num, each_index, data_cls):
        """
        compute two iti_gen losses
        :return:
        """
        # select all the combinations, eg. skin tone. We take C_6^2
        all_comb = it.combinations(range(cate_num), 2)  # 15 combinations
        comb_len = int(cate_num * (cate_num - 1) / 2)  # 15
        temp_loss_direction = torch.zeros(comb_len)
        temp_loss_con = torch.zeros(comb_len)

        # loop through all combinations
        for comb_idx, (first_class, second_class) in enumerate(all_comb):

            loss_direction, loss_con = None, None
            first_class_img_mean = image_features[data_cls == first_class].mean(dim=0) # 768
            second_class_img_mean = image_features[data_cls == second_class].mean(dim=0) # 768
            delta_img = first_class_img_mean - second_class_img_mean  # (+) - (-)
            delta_img = delta_img / delta_img.norm(dim=0, keepdim=True) # 768

            selected_class= torch.LongTensor([first_class, second_class])
            logits_con = self.ori_text_feature @ text_features.t()  # 108*108

            # for each index, select the corresponding text features and compute loss
            for i, index in enumerate(each_index):
                delta_txt = text_features[index[first_class]] - text_features[index[second_class]]
                delta_txt = delta_txt / delta_txt.norm(dim=0, keepdim=True) # 768
                logits_direction = delta_img @ delta_txt.t() # 1
                if i == 0:
                    loss_direction = (1 - logits_direction.t())  # 1
                    # logits_con[0,:] is the similarity between original text feature and all new text features.
                    # since we select two classes from cate_num of classes, we only compute their similarity
                    # with self.ori_text_feature in this index and compute the mean
                    loss_con = F.relu(self.args.lam - logits_con[0, index[selected_class]].mean())
                else:
                    loss_direction = loss_direction + (1 - logits_direction.t())
                    loss_con = loss_con + F.relu(self.args.lam - logits_con[0, index[selected_class]].mean())
            loss_direction = loss_direction / len(each_index)
            loss_con = loss_con / len(each_index)

            temp_loss_direction[comb_idx] = loss_direction
            temp_loss_con[comb_idx] = loss_con

        # when the category number is 2 for attributes, such as Male (man and woman), we have one combination.
        # when the category number is larger then 2, we have C_N^2 combinations, which N is the cate_num.
        # in order to keep the updating speed for different attributes (even a specific category in those attributes) roughly the same,
        # we keep the combination number/total loss accumulation as N/2.

        divide_num = 2 * comb_len / cate_num
        loss_direction = temp_loss_direction.sum() / divide_num
        loss_con = temp_loss_con.sum() / divide_num
        return loss_direction, loss_con

    def prompt_prepend(self, prepended_prompt):
        """
        prepend the learnt FairToken after args.prompt_gen
        for Train-once-for-all generation
        Returns:
        """
        pre_tokenized_text_queries, self.last_word_idx_gen = self.prompt_tokenization(prepended_prompt)

        xp = self.clip_model.token_embedding(pre_tokenized_text_queries).type(self.clip_model.dtype)  # (108, 77, 768)
        self.text_queries_embedding_gen = xp + self.clip_model.positional_embedding.type(self.clip_model.dtype)

        return self.construct_fair_text_features(self.text_queries_embedding_gen,
                                                 self.last_word_idx_gen)  # (108, 77, 768)

    def save_model(self, ep, folder_path):
        """
        save the latest FairTokens & prompt embeddings & prepended prompt embeddings
        Returns:
        """
        # Save each FairToken
        for i, fair_model in enumerate(self.fairtoken_model):
            torch.save(fair_model.state_dict(), os.path.join(folder_path, 'basis_perturbation_embed_' + str(ep) + '_' + self.attr_list[i] + '.pth'))
        print("Successfully save FairTokens!")

        # Save prompt embeddings
        latest_prompt_embeddings = self.construct_fair_text_features(self.text_queries_embedding,
                                                                 self.last_word_idx)  # (108, 77, 768)
        basis_np = latest_prompt_embeddings.clone().detach().cpu()
        path = os.path.join(folder_path, 'original_prompt_embedding')
        os.makedirs(path, exist_ok=True)
        torch.save(basis_np, os.path.join(path, 'basis_final_embed_' + str(ep) + '.pt'))  # (108, 77, 768)
        print("Successfully save Models!")

    def train(self, ep, epoch_saving_list, folder_path):
        """
        training pipeline for one epoch
        :return:
        """
        iter_list = []
        if self.attr_num > 1:
            for i in range(1, self.attr_num):
                iter_list.append(iter(self.trainloader[i]))

        for batch_i, data in enumerate(self.trainloader[0]):
            # iteratively update for different attributes
            for attr_idx in range(self.attr_num):

                # get the data when it isn't the first attribute
                if attr_idx > 0:
                    data = next(iter_list[attr_idx-1])

                for i1 in range(self.attr_num):
                    if i1 == attr_idx:
                        self.fairtoken_model[i1].train()
                    else:
                        self.fairtoken_model[i1].eval()

                # get img data and corresponding label
                img = data['img'].to(self.args.device)  # (bs, 3, H, W)
                data_cls = data['label'].to(self.args.device)  # 0 to cate_num-1

                # get image feature
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(img)  # (bs, 768)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)  # (bs, 768)

                # get text feature
                text_features = self.construct_fair_text_features(self.text_queries_embedding, self.last_word_idx) # (108, 77, 768)
                text_features = text_features[torch.arange(text_features.shape[0]), self.tokenized_text_queries.argmax(
                    dim=-1) + self.args.token_length] @ self.clip_model.text_projection  # (108, 768)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)  # (108, 768)

                # compute the cosine loss, use the cosine similarity loss only when direction loss is nan
                loss_cos = self.cos_loss(image_features, text_features, self.index[attr_idx], data_cls)

                # compute direction and semantic consistency loss
                loss_direction, loss_con = self.iti_gen_loss(image_features, text_features, self.cate_num_list[attr_idx], self.index[attr_idx], data_cls)

                # compute the overall loss
                if torch.isnan(loss_direction) == False:
                    loss = loss_direction + loss_con
                else:
                    loss = loss_cos + loss_con


                print(
                    'Epoch {}, step {} / {}. Update No.{} attribute: {:<15}'.format(ep+1, batch_i, self.args.steps_per_epoch, attr_idx+1, self.attr_list[attr_idx]),
                    'Total loss: {:<10.5f}'.format(loss.item()),
                    'Direction loss: {:<10.5f}'.format(loss_direction.item()),
                    'Consistency loss: {:<10.5f}'.format(loss_con.item()),
                    'Sim loss: {:<10.5f}'.format(loss_cos.item()),
                )

                # optimizers step
                self.optimizer_list[attr_idx].zero_grad()
                loss.backward()
                self.optimizer_list[attr_idx].step()

        # scheduler step
        for scheduler in self.scheduler_list:
            scheduler.step()

        # save model
        with torch.no_grad():
            if (ep + 1) in epoch_saving_list:
                self.save_model(ep, folder_path)