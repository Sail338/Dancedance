import math

#given a list of bodyParts get a bounding box for the user
def getUserBoundingBox(human):
    parts = [human.body_parts[part] for part in human.body_parts]
    wingSpanParts = {}
    for part in parts:
        part_name = part.get_part_name()

        if("Wrist" in str(part_name) or "Elbow" in str(part_name) or "Shoulder" in str(part_name) or "Ankle" in str(part_name)):
            wingSpanParts[str(part_name)] = part

    if(len(wingSpanParts) < 7):
        return None

    curr_distance = 0
    height = 0
    #doing left
    ankle = None
    print(wingSpanParts.keys())
    if("CocoPart.LAnkle" in wingSpanParts):
        ankle = wingSpanParts["CocoPart.LAnkle"]
    else:
        ankle = wingSpanParts["CocoPart.RAnkle"]

    l_wrist = wingSpanParts["CocoPart.LWrist"]
    l_shoulder = wingSpanParts['CocoPart.LShoulder']
    height += distanceFormula(l_shoulder.x,l_shoulder.y,l_shoulder.x,ankle.y)
    l_elbow = wingSpanParts['CocoPart.LElbow']
    curr_distance += distanceFormula(l_wrist.x,l_wrist.y,l_elbow.x,l_elbow.y)
    curr_distance += distanceFormula(l_elbow.x,l_elbow.y,l_shoulder.x,l_shoulder.y)
    #This could be flipped
    l_x  = l_shoulder.x - curr_distance
    print(l_shoulder.x)
    high_y = l_shoulder.y + curr_distance
    height += curr_distance
    
    #distance b/w l shoulder and right shoulder
    r_shoulder = wingSpanParts['CocoPart.RShoulder']
    curr_distance += distanceFormula(l_shoulder.x,l_shoulder.y,r_shoulder.x,r_shoulder.y)
    #handle right
    r_wrist = wingSpanParts["CocoPart.RWrist"]
    r_shoulder = wingSpanParts['CocoPart.RShoulder']
    r_elbow = wingSpanParts['CocoPart.RElbow']
    right_span = 0
    right_span += distanceFormula(r_wrist.x,r_wrist.y,r_elbow.x,r_elbow.y)
    right_span += distanceFormula(r_elbow.x,r_elbow.y,r_shoulder.x,r_shoulder.y)
    r_x = r_shoulder.x - right_span 
    curr_distance+=right_span

    low_y = ankle.y

    mid_x = (l_x + r_x)/2
  
    mid_y = (low_y + high_y)/2
    
    return {
        "w":curr_distance,
        "h":height,
        "x":mid_x,
        "y":mid_y
    }

def distanceFormula(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def scaled_bounding_box(human, width, height):
    b = getUserBoundingBox(human)
    if b is None:
        rv = {'w': 0,'x': 0,'y': 0,'h': 0}
    else:
        rv = {'w': b['w'] * 1000,'x': b['x'] * 1000,'y': b['y'] * 720,'h': b['h'] * 720}
    print(rv)
    return rv

if __name__ == "__main__":
    import sys
    from os import getcwd
    sys.path.append(getcwd() + "/tf-pose-estimation")

    from tf_pose.estimator import TfPoseEstimator
    from tf_pose.networks import get_graph_path, model_wh
    import tf_pose.common as common

    import cv2
    import numpy as np

    image = common.read_imgfile("tf-pose-estimation/images/aaa.jpg", None, None)
    if image is None:
        logger.error('Image can not be read, path=%s' % args.image)
        sys.exit(-1)

    model = 'cmu'
    e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
    humans = e.inference(image, resize_to_default=True, upsample_size=4.0)
    print(humans)
    image = TfPoseEstimator.draw_humans(image, humans, bounding_box_fn=scaled_bounding_box)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.set_title('Result')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

    # show network output
    a = fig.add_subplot(2, 2, 2)
    plt.imshow(bgimg, alpha=0.5)
    tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
    plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    tmp2 = e.pafMat.transpose((2, 0, 1))
    tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

    a = fig.add_subplot(2, 2, 3)
    a.set_title('Vectormap-x')
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    a = fig.add_subplot(2, 2, 4)
    a.set_title('Vectormap-y')
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()
    plt.show()
