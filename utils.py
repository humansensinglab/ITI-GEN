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

def split_attr_list(attribute_list='Male'):
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
            return [attribute_list], 1
    else:
        tmp_list = attribute_list.split(',')
        if len(tmp_list) >= 4:
            warnings.warn("Too many attributes. May get bad results")
        return tmp_list, len(tmp_list)

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
        # Not implemented yet, add your own set and attribute

def get_category_for_attribute(attribute_list):
    """
    from attribute name to dataset name to category
    Returns:
        the category correspond to each attribute &
        the category number correspond to each attribute
    """
    categories_list = []
    categories_number_list = []
    for attribute_name in attribute_list:
        category = reference_image_category[get_dataset_for_attribute(attribute_name)]
        categories_list.append(category)
        categories_number_list.append(len(category))
    return categories_list, categories_number_list

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