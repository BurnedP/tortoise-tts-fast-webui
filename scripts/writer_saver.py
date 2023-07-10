import yaml


def write(cat = "", subcat = "", setting = "", value = ""):

    if (cat == "") or (subcat == "") or (setting == ""):
        raise Exception("Improperly Defined Write")
    else:
        with open("./scripts/config.yaml", "r") as cr:
            config_w = yaml.safe_load(cr)

        config_w[cat][subcat][setting] = value

        with open("./scripts/config.yaml", "w") as cw:
            yaml.safe_dump(config_w, cw, default_flow_style=False)
        

def read(cat = "", subcat = "", setting = ""):

    pointer = [cat, subcat, setting]

    with open("./scripts/config.yaml", "r") as cr:
        config_r = yaml.safe_load(cr)

    if pointer == ["", "", ""]:
        raise Exception("Improperly Defined Read")
    elif cat != "" and subcat == "" and setting == "":
        return config_r[cat]
    elif cat != "" and subcat != "" and setting == "":
        return config_r[cat][subcat]
    elif cat != "" and subcat != "" and setting != "":
        return config_r[cat][subcat][setting]
    else:
        raise Exception("Unknown Error on Read")

    

'''
with open("./scripts/config.yaml", "r") as cr:
    config_r = yaml.safe_load(cr)

if cat != "*":
    raise Exception("Unknown Error on Read")
elif (cat == "*"subcat != "*") and (subcat)
    '''
    

'''
match cat, subcat, setting:
    case ("", "", ""):
        raise Exception("Improperly Defined Read")
    case ("*", "", ""):
        return config_w[cat]
    case ("*", "*", ""):
        return config_w[cat][subcat]
    case ("*", "*", "*"):
        return config_w[cat][subcat][setting]
    case _:
        raise Exception("Unknown Error on Read")
'''