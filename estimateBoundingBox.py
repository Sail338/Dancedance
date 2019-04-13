import math
#given a list of bodyParts get a bounding box for the user
def getUserBoundingBox(human):
    parts_dict = human.body_parts
    parts = [human.body_parts[part] for part in human.body_parts]
    wingSpanParts = {}
    for part in parts:
        part_name = part.get_part_name()
        if("Wrist" in part_name or "Elbow" in part_name or "Shoulder" in part_name or "Ankle" in part_name):
            wingSpanParts[part_name] = part
    if(len(wingSpanParts) < 7):
        return None
    curr_distance = 0
    height = 0
    #doing left
    ankle = None
    if("LAnkle" in wingSpanParts):
        ankle = wingSpanParts["LAnkle"]
    else:
        ankle = wingSpanParts["RAnkle"]
    height -= ankle.y
    l_wrist = wingSpanParts["CocaPart.LWrist"]
    l_shoulder = wingSpanParts['CocaPart.LShoulder']
    height += l_shoulder.y
    l_elbow = wingSpanParts['CocaPart.LElbow']
    curr_distance += distanceFormula(l_wrist.x,l_wrist.y,l_elbow.x,l_elbow.y)
    curr_distance += distanceFormula(l_elbow.x,l_elbow.y,l_shoulder.x,l_shoulder.y)
    #This could be flipped
    l_x  = l_shoulder.x - curr_distance
    height += curr_distance
    
    #distance b/w l shoulder and right shoulder
    r_shoulder = wingSpanParts['CocaPart.RShoulder']
    curr_distance += distanceFormula(l_shoulder.x,l_shoulder.y,r_shoulder.x,r_shoulder.y)
    #handle right
    r_wrist = wingSpanParts["CocaPart.RWrist"]
    r_shoulder = wingSpanParts['CocaPart.RShoulder']
    r_elbow = wingSpanParts['CocaPart.RElbow']
    right_span = 0
    right_span += distanceFormula(r_wrist.x,r_wrist.y,r_elbow.x,r_elbow.y)
    right_span += distanceFormula(r_elbow.x,r_elbow.y,r_shoulder.x,r_shoulder.y)
    r_x = r_shoulder.x + right_span 
    curr_distance+=right_span

    low_y = ankle.y
    high_y = height

    mid_x = (l_x + r_x)/2
    mid_y = (low_y + high_y)/2
    
    return {
        "width":x,
        "height":y,
        "midx":mid_x
        "midy":mid_y
    }





def distanceFormula(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)



    



