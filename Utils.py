import cv2

def process_img(img, face_detection):

    H, W, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out_face = face_detection.process(img_rgb)

    if out_face.detections is not None:

        for detection in out_face.detections:

            bbox = detection.location_data.relative_bounding_box
            x, y, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x * W)
            y1 = int(y * H)
            h = int(h * H)
            w = int(w * W)

            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (50, 50))

    return img