import numpy as np 
import cv2 
from pose.format_coco import Points, mapIdx, POSE_PAIRS

def getValidPairs(image, output, detected_keypoints, keypoints_list):
    # The function is used to determine the connection between joints A and B.
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_threshold = 0.1
    conf_th = 0.7
    for idx in range(len(mapIdx)):
        pafA = output[0, mapIdx[idx][0], :, :]
        pafB = output[0, mapIdx[idx][1], :, :]
        pafA = cv2.resize(pafA, (image.shape[1], image.shape[0]))
        pafB = cv2.resize(pafB, (image.shape[1], image.shape[0]))

        candA = detected_keypoints[POSE_PAIRS[idx][0]]
        candB = detected_keypoints[POSE_PAIRS[idx][1]]
        n_A = len(candA)
        n_B = len(candB)

        if n_A != 0 and n_B != 0:
            valid_pair = np.zeros((0,3))
            for i in range(n_A):
                j_ok = -1
                avg_max_score = -1
                found = 0
                for j in range(n_B):
                    unitVec_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(unitVec_ij)
                    if norm:
                        unitVec_ij = unitVec_ij / norm
                    else:
                        continue
                    
                    interpolate_p = list(zip(np.linspace(candA[i][0], candB[j][0], num = n_interp_samples),
                                             np.linspace(candA[i][1], candB[j][1], num = n_interp_samples)))
                    
                    PAF_interp = []
                    for k in range(len(interpolate_p)):
                        PAF_interp.append([ pafA[int(round(interpolate_p[k][1])), int(round(interpolate_p[k][0]))],
                                            pafB[int(round(interpolate_p[k][1])), int(round(interpolate_p[k][0]))] ])
                        
                    paf_scores = np.dot(PAF_interp, unitVec_ij)
                    avg_paf_score = sum(paf_scores) / len(paf_scores)

                    if (len(np.where(paf_scores > paf_score_threshold)[0]) / n_interp_samples) > conf_th:
                        if avg_paf_score > avg_max_score:
                            j_ok = j
                            avg_max_score = avg_paf_score
                            found = 1
                            
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[j_ok][3], avg_max_score]], axis=0)

            valid_pairs.append(valid_pair)

        else:
            invalid_pairs.append(idx)
            valid_pairs.append([])

    return valid_pairs, invalid_pairs


def getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list):
    # the last number in each row is the overall score 
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])): 
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score 
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints



def visualizer(image, detected_keypoints, keypoints_list, output):
    #draw pose estimation
    vali, invali = getValidPairs(image, output, detected_keypoints, keypoints_list)
    personWiseKeyPoints = getPersonwiseKeypoints(vali, invali, keypoints_list)
    show = image.copy()
    for i in range(Points):
        for j in range(len(detected_keypoints[i])):
            cv2.circle(show, detected_keypoints[i][j][0:2], 3, [0,0,255], -1, cv2.LINE_AA)
    #return show
    for i in range(17):
        for n in range(len(personWiseKeyPoints)):
            index = personWiseKeyPoints[n][np.array(POSE_PAIRS[i])]
            if -1 in index:
                continue
            B = np.int32(keypoints_list[index.astype(int), 0])
        A = np.int32(keypoints_list[index.astype(int), 1])
        cv2.line(show, (B[0], A[0]), (B[1], A[1]), [255,0,0], 3, cv2.LINE_AA)
    return show




