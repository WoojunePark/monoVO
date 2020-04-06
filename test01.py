import numpy as np
import cv2
import os

from visual_odometry import PinholeCamera, VisualOdometry


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


for pose_id in range(6, 11):
    pose_file_loc = '/database3/KITTI_odometry/dataset/poses/' + str(pose_id).zfill(2) + '.txt'
    pose_file = open(pose_file_loc, 'r')
    pose_file_len = len(pose_file.readlines())
    print("pose_file_len = ", pose_file_len)

    cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
    vo = VisualOdometry(cam, pose_file_loc)

    traj = np.zeros((600, 600, 3), dtype=np.uint8)

    ate_rmse_sum = 0.0

    section_points = 10
    adjustment_x = 0.0
    adjustment_y = 0.0
    rte_rmse_sum = 0.0

    for img_id in range(pose_file_len):
        img_file_loc = '/database3/KITTI_odometry/dataset/sequences/' + str(pose_id).zfill(2) + '/image_2/' + str(
            img_id).zfill(6) + '.png'
        img = cv2.imread(img_file_loc, 0)
        vo.update(img, img_id)
        cur_t = vo.cur_t
        if img_id > 2:
            x, y, z = cur_t[0], cur_t[1], cur_t[2]
        else:
            x, y, z = 0., 0., 0.
        draw_x, draw_y = int(x) + 290, int(z) + 90
        true_x, true_y = int(vo.trueX) + 290, int(vo.trueZ) + 90

        # estimated ATE by point-wise
        true_xy = [true_x, true_y]
        if x == 0.0:
            pred_xy = [x, y]
        else:
            pred_xy = [x[0], y[0]]

        ate_rmse = rmse(np.array(pred_xy), np.array(true_xy))
        ate_rmse_sum = ate_rmse_sum + ate_rmse

        # estimate RTE by N point section-wise, estimate RTE at last point
        if img_id % section_points == 0:
            # rte_benchmark = true_xy
            # A(0,0) - B(1,1) = adjusting_B_to_A(-1, -1)
            # C(2,3) + adjusting_B_to_A(-1, -1) = C'(1,2)
            adjustment_x = true_xy[0] - pred_xy[0]
            adjustment_y = true_xy[1] - pred_xy[1]
        else:
            pass
        adjusted_pred_xy = [pred_xy[0] + adjustment_x, pred_xy[1] + adjustment_y]
        rte_rmse = rmse(np.array(adjusted_pred_xy), np.array(true_xy))
        rte_rmse_sum = rte_rmse_sum + rte_rmse

        # print how's it going every 100
        # if img_id % 100 == 0:
        # 	print("img_id", img_id, "  true_xy:", true_xy, "  pred_xy:", pred_xy)
        # 	print("ate_rmse", ate_rmse)
        # 	print("ate_rmse_sum", ate_rmse_sum)
        # 	print("rte_rmse", rte_rmse)
        # 	print("rte_rmse_sum", rte_rmse_sum)

        # traj line color green to blue
        cv2.circle(traj, (draw_x, draw_y), 1, (img_id * 255 / 4540, 255 - img_id * 255 / 4540, 0), 1)
        cv2.circle(traj, (true_x, true_y), 1, (0, 0, 255), 2)
        cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
        cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
        path = '/home/wjpark/Github/monoVO/output/'

    # if img_id % 10 == 0:
    #     cv2.imwrite(os.path.join(path, 'Road facing camera_', str(img_id), '_.jpg'), img)
    #     cv2.imwrite(os.path.join(path, 'Trajectory_', str(img_id), '_.jpg'), traj)

    ate = ate_rmse_sum / pose_file_len
    print("pose_id", pose_id, " ATE : ", ate)

    rte = rte_rmse_sum / pose_file_len
    print("pose_id", pose_id, " RTE : ", rte)

    cv2.imwrite('/home/wjpark/Github/monoVO/map' + str(pose_id).zfill(2) + '.png', traj)
