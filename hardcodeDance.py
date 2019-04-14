def hardcodeDance(human):
    parts_dict = human.body_parts
    for k,v in part_dict.items():
        if "Wrist" in str(v.get_part_name):
            print("ay a wrist")
        else:
            print("not a writs")
