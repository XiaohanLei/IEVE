import cv2
import numpy as np


def get_contour_points(pos, origin, size=20):
    x, y, o = pos
    pt1 = (int(x) + origin[0],
           int(y) + origin[1])
    pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
    pt3 = (int(x + size * np.cos(o)) + origin[0],
           int(y + size * np.sin(o)) + origin[1])
    pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])

    return np.array([pt1, pt2, pt3, pt4])


def draw_line(start, end, mat, steps=25, w=1):
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w:x + w, y - w:y + w] = 1
    return mat

def update_text(vis_image, text):
    vis_image[52:52+50, 510+60:870-60] = [255, 255, 255]
    if text == 'Exploitation':
        color_X = [0.12156862745098039, 0.47058823529411764, 0.7058823529411765]
    elif text == 'Exploration':
        color_X = [0.7058823529411765, 0.12156862745098039, 0.47058823529411764]
    else:
        color_X = [0.12156862745098039, 0.7058823529411765, 0.47058823529411764]
    
    color = (int(color_X[2]*255), int(color_X[1]*255), int(color_X[0]*255))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 480 + (360 - textsize[0]) // 2 + 30
    textY = (50+100 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)


def init_vis_image(goal_name, legend):
#     vis_image = np.ones((655, 1165, 3)).astype(np.uint8) * 255
    vis_image = np.ones((655, 1380, 3)).astype(np.uint8) * 255
#     vis_image = np.ones((655, 1380+360+15, 3)).astype(np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    text = f"Instance: {goal_name}"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = (480 - textsize[0]) // 2 + 15
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    text = "Observations"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 480 + (360 - textsize[0]) // 2 + 30
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    text = "Map Memory"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 840 + (480 - textsize[0]) // 2 + 45
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    text = "Curiousity Map"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 840 + 480 + (360 - textsize[0]) // 2 + 60
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    # draw outlines
    color = [100, 100, 100]
    vis_image[49, 15:495] = color
    vis_image[49, 510:870] = color
    vis_image[49, 885:1365] = color
#     vis_image[49, 1380:(1380+360)] = color
    vis_image[50:530, 14] = color
    vis_image[50:530, 495] = color
    vis_image[50:530, 509] = color
    vis_image[50:530, 870] = color
    vis_image[50:530, 884] = color
    vis_image[50:530, 1365] = color
#     vis_image[50:530, 1365+15-1] = color
#     vis_image[50:530, 1365+15+360] = color
    vis_image[530, 15:495] = color
    vis_image[530, 510:870] = color
    vis_image[530, 885:1365] = color
#     vis_image[530, 1380:(1380+360)] = color

    # draw legend
    lx, ly, _ = legend.shape
    vis_image[537:537 + lx, 300:300 + ly, :] = legend

    return vis_image
