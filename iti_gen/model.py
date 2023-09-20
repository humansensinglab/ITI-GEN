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

from utils import split_attribute_list, make_attr2dataset_dictionary, make_attr2category_list

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
        self.attribute_list, self.attribute_num = split_attribute_list(self.args.attribute_list)
        # make a dict that change attribute name to dataset name
        self.attr2dataset_dict = make_attr2dataset_dictionary()
        # change the attribute name to category name
        self.categories_list, self.category_number_list = make_attr2category_list(self.attribute_list, self.attr2dataset_dict)

        # construct dataset and data loader
        self._init_dataset()
        self._init_dataloader()
        self.token_combination_number = int(np.prod(np.array(self.category_number_list)))

        # make models
        self.setup_clip_model()
        self.clip_layers_number = self.clip_model.transformer.layers

        # self.last_word_index equals 6 in '<start> a headshot of a person <end>'
        self.tokenized_text_queries, self.last_word_index = self.prompt_tokenization(self.args.prompt)

        self.setup_fairtoken_model()
        self.setup_optimizer()
        self.setup_scheduler()

        # get the original text prompt feature
        self.ori_clip_text_feature_extraction()

        self.setup_index()

    def _init_dataset(self):
        """
        init dataset for different attributes
        :return:
        """
        # a list of training dataset. eg, 400, 702, 1800 -> gender, skin_tone, age
        self.train_dataset_list = []

        for idx, attribute_name in enumerate(self.attribute_list):

            dataset_root_folder = os.path.join(self.args.data_path, self.attr2dataset_dict[attribute_name])
            attribute_root_folder = os.path.join(dataset_root_folder, attribute_name)

            # reference image path list and labels
            label_type = [i for i in range(self.category_number_list[idx])]
            refer_img_path_list = []
            for category in self.categories_list[idx]:
                refer_img_path_list.append(os.path.join(attribute_root_folder, category))

            self.train_dataset_list.append(ImgDataset(root_dir=refer_img_path_list, label=label_type, upper_bound=self.args.refer_size_per_category))

    def _init_dataloader(self):
        """
        init dataloader for different attributes
        :return:
        """
        # a list of training dataloader for different attributes
        self.train_dataloader_list = []
        # a list of batch size for different attributes
        self.batch_size_list = []
        for i, dataset in enumerate(self.train_dataset_list):
            batch_size = len(dataset)//self.args.steps_per_epoch
            self.train_dataloader_list.append(DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=1,
                shuffle=True,
                # drop the last one
                drop_last=True,
                pin_memory=True,
            ))
            self.batch_size_list.append(batch_size)

    def setup_clip_model(self):
        self.clip_model, _ = clip.load("ViT-L/14", device="cpu")
        self.clip_model.to(device=self.args.device)
        self.clip_model.eval()

    def prompt_tokenization(self, prompt):
        """
        tokenize the text prompt
        :return:
        """
        text_queries = [prompt] * self.token_combination_number  # composite basis, eg,108
        tokenized_text_queries = clip.tokenize(text_queries).to(self.args.device)
        last_word_index = tokenized_text_queries[-1].argmax()
        return tokenized_text_queries, last_word_index

    def setup_fairtoken_model(self):
        """
        Initialize the fairtoken models
        :return:
        """
        self.fairtoken_model_list = []
        for category_number in self.category_number_list:
            self.fairtoken_model_list.append(
                FairToken(num_basis=category_number, token_length=self.args.token_length).to(device=self.args.device)
            )

    def setup_optimizer(self):
        self.optimizer_list = []
        for fairtoken_model in self.fairtoken_model_list:
            self.optimizer_list.append(
                optim.Adam(fairtoken_model.parameters(), lr=self.args.lr)
            )

    def setup_scheduler(self):
        self.scheduler_list = []
        for optimizer in self.optimizer_list:
            self.scheduler_list.append(
                optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[10], gamma=0.5, verbose=True)
            )

    def ori_clip_text_feature_extraction(self):
        """
        get the text features for self.args.prompt/self.tokenized_text_queries
        for computing semantic consistency loss
        :return:
        """
        with torch.no_grad():
            xp = self.clip_model.token_embedding(self.tokenized_text_queries).type(self.clip_model.dtype)  # (108, 77, 768)
            self.text_queries_embedding = xp + self.clip_model.positional_embedding.type(self.clip_model.dtype)
            xp = self.text_queries_embedding.permute(1, 0, 2)
            for ll in range(self.clip_layers_number):
                xp = self.clip_model.transformer.resblocks[ll](xp)
            xp = xp.permute(1, 0, 2)
            xp = self.clip_model.ln_final(xp).type(self.clip_model.dtype)  # (108, 77, 768)
            xp_feat = xp[torch.arange(xp.shape[0]), self.tokenized_text_queries.argmax(dim=-1)] @ self.clip_model.text_projection  # 108*768
            self.ori_clip_text_feature = xp_feat / xp_feat.norm(dim=1, keepdim=True) # (108, 768)

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
        :return:
        """
        # 1, 2, 12
        pre_multiplication_accumulation = 1
        # 108, 54, 9
        non_pre_multiplication_accumulation = self.token_combination_number # 108
        # 54, 9, 1
        later_multiplication_accumulation = int(self.token_combination_number // self.category_number_list[0])

        # a list containing self.attribute_num's number of list
        self.index = []
        for attribute_index in range(self.attribute_num):
            each_index = []
            # if attribute_index == 0, use the default value
            if attribute_index - 1 >= 0:
                pre_multiplication_accumulation *= self.category_number_list[attribute_index - 1]
                non_pre_multiplication_accumulation = int(non_pre_multiplication_accumulation/self.category_number_list[attribute_index - 1])
                later_multiplication_accumulation = int(later_multiplication_accumulation/self.category_number_list[attribute_index])

            for i1 in range(pre_multiplication_accumulation):
                for i2 in range(later_multiplication_accumulation):
                    # construct a tensor with the length equals attribute category's number.
                    # skin tone: 6, age: 9...
                    tensor_list = []
                    for i3 in range(self.category_number_list[attribute_index]):
                        tensor_list.append(0+non_pre_multiplication_accumulation*i1+1*i2 + i3*later_multiplication_accumulation)
                    each_index.append(torch.LongTensor(tensor_list))
            self.index.append(each_index)

    def construct_fair_text_features(self, text_queries_embedding, last_word_index):
        """
        insert fair_tokens to the original text queries' embeddings and obtain the
        corresponding text feature
        :return:
        """
        x = text_queries_embedding.detach() # (108, 77, 768)
        # add FairToken to the corresponding place
        # for 1st attr, replace from the last word index; for 2nd and later, add to them
        for i2, each_index in enumerate(self.index):
            for index in each_index:
                if i2 == 0:
                    x[index, last_word_index:last_word_index + self.args.token_length, :] = (
                    self.fairtoken_model_list[i2])()
                else:
                    x[index, last_word_index:last_word_index + self.args.token_length, :] += (
                    self.fairtoken_model_list[i2])()
        x = x.permute(1, 0, 2)
        for ll in range(self.clip_layers_number):
            x = self.clip_model.transformer.resblocks[ll](x)
        x = x.permute(1, 0, 2)
        return self.clip_model.ln_final(x).type(self.clip_model.dtype)  # (108, 77, 768)

    def cos_loss(self, image_features, text_features, each_index, data_cls):
        """
        compute the cosine similarity loss, 1-cos(image, text)
        :return:
        """
        # get the cosine similarity logits between text and image
        logits_i = image_features @ text_features.t()  # (n_img, n_text) # bs * 108
        # for each image, we only try to maximize the cosine similarity between the image and corresponding text, which should own the same
        # category value for that attribute with image.
        # Eg, for skin tone, image1's label is type3, so for text1, text2, ..., text6. Only text3 should have skin tone type3, so we mask
        # all the other texts.
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
        for i in range(self.token_combination_number):
            if mask[i] != 0:
                logits_i[i] = 1 - logits_i[i] / mask[i]
            else:
                logits_i[i] = 0
        # make summation for the cosine similarity loss and divide by the index length
        return logits_i.sum() / len(each_index)

    def iti_gen_loss(self, image_features, text_features, category_number, each_index, data_cls):
        """
        compute two iti_gen losses
        :return:
        """
        # select all the combinations, eg. skin tone. We take C_6^2
        all_combinations = it.combinations(range(category_number), 2)  # 15 combinations
        combination_length = int(category_number * (category_number - 1) / 2)  # 15
        temp_loss_direction = torch.zeros(combination_length)
        temp_loss_con = torch.zeros(combination_length)

        # loop through all combinations
        for combination_index, (first_class, second_class) in enumerate(all_combinations):

            loss_direction, loss_con = None, None
            first_class_img_mean = image_features[data_cls == first_class].mean(dim=0) # 768
            second_class_img_mean = image_features[data_cls == second_class].mean(dim=0) # 768
            delta_img = first_class_img_mean - second_class_img_mean  # (+) - (-)
            delta_img = delta_img / delta_img.norm(dim=0, keepdim=True) # 768

            selected_class= torch.LongTensor([first_class, second_class])
            logits_con = self.ori_clip_text_feature @ text_features.t()  # 108*108

            # for each index, select the corresponding text features and compute loss
            for i, index in enumerate(each_index):
                delta_txt = text_features[index[first_class]] - text_features[index[second_class]]
                delta_txt = delta_txt / delta_txt.norm(dim=0, keepdim=True) # 768
                logits_direction = delta_img @ delta_txt.t() # 1
                if i == 0:
                    loss_direction = (1 - logits_direction.t())  # 1
                    # logits_con[0,:] is the similarity between original text feature and all new text features.
                    # since we select two classes from category_number of classes, we only compute their similarity
                    # with self.ori_clip_text_feature in this index and compute the mean
                    loss_con = F.relu(self.args.lam - logits_con[0, index[selected_class]].mean())
                else:
                    loss_direction = loss_direction + (1 - logits_direction.t())
                    loss_con = loss_con + F.relu(self.args.lam - logits_con[0, index[selected_class]].mean())
            loss_direction = loss_direction / len(each_index)
            loss_con = loss_con / len(each_index)

            temp_loss_direction[combination_index] = loss_direction
            temp_loss_con[combination_index] = loss_con

        # when the category number is 2 for attributes, such as Male (man and woman), we have one combination.
        # when the category number is larger then 2, we have C_N^2 combinations, which N is the category_number.
        # in order to keep the updating speed for different attributes (even a specific category in those attributes) roughly the same,
        # we keep the combination number/total loss accumulation as N/2.

        divide_num = 2 * combination_length / category_number
        loss_direction = temp_loss_direction.sum() / divide_num
        loss_con = temp_loss_con.sum() / divide_num
        return loss_direction, loss_con

    def prompt_prepend(self, prepended_prompt):
        """
        prepend the learnt FairToken after args.prompt_gen
        for Train-once-for-all generation
        Returns:
        """
        self.tokenized_text_queries_gen, self.last_word_index_gen = self.prompt_tokenization(prepended_prompt)

        xp = self.clip_model.token_embedding(self.tokenized_text_queries_gen).type(self.clip_model.dtype)  # (108, 77, 768)
        self.text_queries_embedding_gen = xp + self.clip_model.positional_embedding.type(self.clip_model.dtype)

        return self.construct_fair_text_features(self.text_queries_embedding_gen,
                                                 self.last_word_index_gen)  # (108, 77, 768)

    def save_model(self, ep, folder_path):
        """
        save the latest FairTokens & prompt embeddings & prepended prompt embeddings
        Returns:
        """
        # Save each FairToken
        for i, fairtoken_model in enumerate(self.fairtoken_model_list):
            torch.save(fairtoken_model.state_dict(), os.path.join(folder_path, 'basis_perturbation_embed_' + str(ep) + '_' + self.attribute_list[i] + '.pth'))
        print("Successfully save FairTokens!")

        # Save prompt embeddings
        latest_prompt_embeddings = self.construct_fair_text_features(self.text_queries_embedding,
                                                                 self.last_word_index)  # (108, 77, 768)
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
        # use first dataloader to enumerate; for the remaining dataloader, we use iter.
        iter_list = []
        if self.attribute_num > 1:
            for i in range(1, self.attribute_num):
                iter_list.append(iter(self.train_dataloader_list[i]))
        for batch_i, data in enumerate(self.train_dataloader_list[0]):
            # iteratively update for different attributes
            for attr_idx in range(self.attribute_num):

                # get the data when it is not the first train loader
                if attr_idx > 0:
                    data = next(iter_list[attr_idx-1])

                # set models train and eval
                for i1 in range(self.attribute_num):
                    if i1 == attr_idx:
                        self.fairtoken_model_list[i1].train()
                    else:
                        self.fairtoken_model_list[i1].eval()

                # get img data and corresponding label
                img = data['img'].to(self.args.device)  # (bs, 3, H, W)
                data_cls = data['label'].to(self.args.device)  # 0 to category_number-1

                # get image feature
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(img)  # (bs, 768)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)  # (bs, 768)

                # get text feature
                text_features = self.construct_fair_text_features(self.text_queries_embedding, self.last_word_index) # (108, 77, 768)
                text_features = text_features[torch.arange(text_features.shape[0]), self.tokenized_text_queries.argmax(
                    dim=-1) + self.args.token_length] @ self.clip_model.text_projection  # (108, 768)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)  # (108, 768)

                # compute the cosine loss, use the cosine similarity loss only when direction loss is nan
                loss_cos = self.cos_loss(image_features, text_features, self.index[attr_idx], data_cls)

                # compute direction and semantic consistency loss
                loss_direction, loss_con = self.iti_gen_loss(image_features, text_features, self.category_number_list[attr_idx], self.index[attr_idx], data_cls)

                # compute the overall loss
                if torch.isnan(loss_direction) == False:
                    loss = loss_direction + loss_con
                else:
                    loss = loss_cos + loss_con


                print(
                    'Epoch {}, step {} / {}. Update No.{} attribute: {:<15}'.format(ep+1, batch_i, self.args.steps_per_epoch, attr_idx+1, self.attribute_list[attr_idx]),
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