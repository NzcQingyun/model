import sys
import numpy as np
import cv2
import onnxruntime
import time
import queue
import threading
import json
import copy

"""这段代码实现了一个RetinaFace人脸检测器。该检测器基于一个基于RetinaFace的目标检测模型，在输入图像上运行以检测人脸。
以下是代码的一些主要组件和函数：

import语句导入了必要的库和模块。

py_cpu_nms()是一个Python实现的非极大值抑制（NMS）算法，用于从目标框中过滤出最好的框，从而避免多个框重叠。此函数在后面的代码中被调用。

decode()函数接受一个由目标检测器预测的偏移量、一个先验框数组和方差数组，并返回一个经过变换的边界框数组。此函数在后面的代码中被调用。

worker_thread()函数是一个辅助函数，用于在后台线程中运行RetinaFace检测器，以便不会阻塞主线程。此函数在后面的代码中被调用。

RetinaFaceDetector类是一个包装RetinaFace检测器的类，可以在其上调用detect_retina()方法以检测人脸。此类有以下一些主要属性和方法：

__init__()方法创建RetinaFace检测器对象。此方法接受一些可选参数，如模型路径、先验框JSON文件路径、线程数、置信度阈值、NMS阈值、
检测的最大框数和图像大小。此方法还加载目标检测器的ONNX模型和先验框数组，并初始化队列、标志和其他属性。
detect_retina()方法接受一个图像并在其中检测人脸。此方法调整图像大小以适应目标检测器的输入大小，运行目标检测器以获取预测边界框和分数，
然后执行NMS算法以过滤掉低分数的框并将其缩小为指定数量。最后，该方法返回检测到的边界框的数组。"""
def py_cpu_nms(dets, thresh):
    """ Pure Python NMS baseline.
        Copyright (c) 2015 Microsoft
        Licensed under The MIT License
        Written by Ross Girshick
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def decode(loc, priors, variances):
    data = (
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
    )
    boxes = np.concatenate(data, 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def worker_thread(rfd, frame):
    results = rfd.detect_retina(frame, is_background=True)
    rfd.results.put(results, False)
    rfd.finished = True
    rfd.running = False

class RetinaFaceDetector():
    def __init__(self, model_path="models/retinaface_640x640_opt.onnx", json_path="models/priorbox_640x640.json", threads=4, min_conf=0.4, nms_threshold=0.4, top_k=1, res=(640, 640)):
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = threads
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        providersList = onnxruntime.capi._pybind_state.get_available_providers()
        self.session = onnxruntime.InferenceSession(model_path, sess_options=options, providers=providersList)
        self.res_w, self.res_h = res
        with open(json_path, "r") as prior_file:
            self.priorbox = np.array(json.loads(prior_file.read()))
        self.min_conf = min_conf
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.finished = False
        self.running = False
        self.results = queue.Queue()

    def detect_retina(self, frame, is_background=False):
        h, w, _ = frame.shape
        im = None
        im = cv2.resize(frame, (self.res_w, self.res_h), interpolation=cv2.INTER_LINEAR)
        resize_w = w / self.res_w
        resize_w = 1 / resize_w
        resize_h = h / self.res_h
        resize_h = 1 / resize_h
        im = np.float32(im)
        scale = np.array((self.res_w / resize_w, self.res_h / resize_h, self.res_w / resize_w, self.res_h / resize_h))
        im -= (104, 117, 123)
        im = im.transpose(2, 0, 1)
        im = np.expand_dims(im, 0)
        output = self.session.run([], {"input0": im})
        loc, conf = output[0][0], output[1][0]
        boxes = decode(loc, self.priorbox, [0.1, 0.2])
        boxes = boxes * scale
        scores = conf[:, 1]

        inds = np.where(scores > self.min_conf)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        dets = dets[:self.top_k, 0:4]
        dets[:, 2:4] = dets[:, 2:4] - dets[:, 0:2]

        if True:#is_background:
            upsize = dets[:, 2:4] * np.array([[0.15, 0.2]])
            dets[:, 0:2] -= upsize
            dets[:, 2:4] += upsize * 2

        return list(map(tuple, dets))

    def background_detect(self, frame):
        if self.running or self.finished:
            return
        self.running = True
        im = copy.copy(frame)
        thread = threading.Thread(target=worker_thread, args=(self, im))
        thread.start()

    def get_results(self):
        if self.finished:
            results = []
            try:
                while True:
                    detection = self.results.get(False)
                    results.append(detection)
            except:
                "No error"
            self.finished = False
            return list(*results)
        else:
            return []

if __name__== "__main__":
    retina = RetinaFaceDetector(top_k=40, min_conf=0.2)
    im = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    start = time.perf_counter()
    faces = retina.detect_retina(im)
    end = 1000 * (time.perf_counter() - start)
    print(f"Runtime: {end:.3f}ms")
    for (x,y,w,h) in faces:
        im = cv2.rectangle(im, (int(x),int(y)), (int(x+w),int(y+w)), (0,0,255), 1)
    cv2.imshow("Frame", im)
    while cv2.waitKey(0) & 0xff != ord('q'):
        ""
