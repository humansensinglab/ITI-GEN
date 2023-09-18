global CelebA_attribute_list, FairFace_attribute_list, FAIR_attribute_list, LHQ_attribute_list
global reference_image_category

CelebA_attribute_list = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
                         'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
                         'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                         'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                          'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                         'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
                          'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                         'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
FairFace_attribute_list = ['Age'] # may add Gender, Race if you want
FAIR_attribute_list = ['Skin_tone']
LHQ_attribute_list = ['Bright', 'Colorful', 'Complex', 'Contrast', 'Good', 'Happy', 'Natural', 'New', 'Noisy', 'Scary', 'Sharp']

# get the category value for different dataset
reference_image_category = {"celeba":['positive', 'negative'],
                            "fairface":['0_2','3_9','10_19','20_29','30_39','40_49','50_59','60_69','more_than_70'],
                            "FAIR_benchmark":['1','2','3','4','5','6'],
                            "lhq":['0','1','2','3','4']}
## add your custom dataset/attribute/reference images here

def split_attribute_list(attribute_list='Male'):
    """
    :param attribute_list:
        attribute names separated by commas
    :return:
        attribute name list &
        attribute number
    """
    if ',' not in attribute_list:
        if len(attribute_list) == 0:
            raise Warning("No input attribute")
        else:
            attribute_list_ = [attribute_list]
    else:
        attribute_list_ = attribute_list.split(',')
        if len(attribute_list_) >= 4:
            warnings.warn("Too many attributes. May get bad results")
    return attribute_list_, len(attribute_list_)

def make_attr2dataset_dictionary():
    """
    construct the dictionary that from attribute name to dataset name
    :return:
        the dict from attribute name to dataset name
    """
    attr2dataset_dict = {}
    for celeba_attribute in CelebA_attribute_list:
        attr2dataset_dict[celeba_attribute] = "celeba"
    for fairface_attribute in FairFace_attribute_list:
        attr2dataset_dict[fairface_attribute] = "fairface"
    for fair_attribute in FAIR_attribute_list:
        attr2dataset_dict[fair_attribute] = "FAIR_benchmark"
    for lhq_attribute in LHQ_attribute_list:
        attr2dataset_dict[lhq_attribute] = "lhq"
    ### add your custom dataset here ###

    return attr2dataset_dict

def make_attr2category_list(attribute_list, attr2dataset_dict):
    """
    from attribute name to dataset name to category
    Returns:
        the category correspond to each attribute &
        the category number correspond to each attribute
    """
    categories_list = []
    category_number_list = []

    for attribute_name in attribute_list:
        if attribute_name in attr2dataset_dict:

            category = reference_image_category[attr2dataset_dict[attribute_name]]
            categories_list.append(category)
            category_number_list.append(len(category))
        else:
            raise Warning("Not implemented yet, please add your custom attribute and reference image")
    # [['positive', 'negative'], ['1','2','3','4','5','6'], []]
    # [2, 6, 9]
    return categories_list, category_number_list


def get_dataset_for_attribute(attribute):
    """
    Determine the dataset for each attribute

    """
    if attribute in CelebA_attribute_list:
        return "celeba"
    elif attribute in FairFace_attribute_list:
        return "fairface"
    elif attribute in FAIR_attribute_list:
        return "FAIR_benchmark"
    elif attribute in LHQ_attribute_list:
        return "lhq"
    else:
        raise ValueError(f"Unknown attribute: {attribute}")


def get_combinations(attributes):
    """
    Get combinations
    
    """
    datasets = [get_dataset_for_attribute(attr) for attr in attributes]
    category_lists = [reference_image_category[dataset] for dataset in datasets]

    # Using itertools product to get combinations
    from itertools import product
    return list(product(*category_lists))


def get_folder_names_and_indexes(attributes):
    """
    Generate folder names and indexes
    
    """
    combinations = get_combinations(attributes)
    folder_names = []

    for combo in combinations:
        name_parts = []
        for attr, cat in zip(attributes, combo):
            name_parts.append(f"{attr}_{cat}")
        folder_name = "_".join(name_parts)
        folder_names.append(folder_name)

    # Pair each folder name with an index
    folder_with_indexes = {folder_name: idx for idx, folder_name in enumerate(folder_names)}

    return folder_with_indexes
    