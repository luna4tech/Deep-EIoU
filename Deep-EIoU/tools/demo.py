import argparse
import os
import os.path as osp
import numpy as np
import time
import cv2
import torch
import sys
sys.path.append('.')

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracking_utils.timer import Timer

from tracker.Deep_EIoU import Deep_EIoU
from reid.torchreid.utils import FeatureExtractor
import torchvision.transforms as T


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("DeepEIoU Demo")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="../demo.mp4", help="path to images or video"
    )
    parser.add_argument(
        "--save_result",
        default=True,
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="yolox/yolox_x_ch_sportsmot.py",
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="number of frames to run detection and Re-ID on per forward pass "
        "(the tracker still runs frame by frame). Use 1 to disable batching.",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--nms_thres", type=float, default=0.7, help='nms threshold')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    # reid args
    parser.add_argument("--with-reid", dest="with_reid", default=True, action="store_true", help="use Re-ID flag.")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info

    def inference_batch(self, imgs):
        """Run detection on a batch of frames in a single forward pass.

        ``imgs`` is a list of BGR ``np.ndarray`` frames, all of the same
        resolution (which always holds for frames from one video). Every frame
        therefore preprocesses to the same tensor shape and shares the same
        resize ``ratio``, so they stack cleanly into one ``(N, 3, H, W)`` batch.

        Returns ``(outputs, img_infos)`` where ``outputs`` is the list returned
        by ``postprocess`` (one detection set, or ``None``, per frame in input
        order) and ``img_infos`` is the matching list of per-frame info dicts.
        """
        img_infos = []
        batch = []
        for img in imgs:
            height, width = img.shape[:2]
            img_info = {
                "id": 0,
                "file_name": None,
                "height": height,
                "width": width,
                "raw_img": img,
            }
            proc_img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
            img_info["ratio"] = ratio
            img_infos.append(img_info)
            batch.append(torch.from_numpy(proc_img))

        batch = torch.stack(batch, dim=0).float().to(self.device)
        if self.fp16:
            batch = batch.half()  # to FP16

        with torch.no_grad():
            outputs = self.model(batch)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_infos


def imageflow_demo(predictor, extractor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    save_path = osp.join(save_folder, args.path.split("/")[-1])
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = Deep_EIoU(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    scale = min(1440 / width, 800 / height)

    batch_size = max(1, args.batch_size)
    if args.trt and batch_size != 1:
        logger.warning(
            "TensorRT engine is built for batch size 1; forcing --batch_size 1."
        )
        batch_size = 1

    stop = False
    while not stop:
        # 1. accumulate up to batch_size frames (the last batch may be smaller)
        frames = []
        for _ in range(batch_size):
            ret_val, frame = cap.read()
            if not ret_val:
                break
            frames.append(frame)
        if not frames:
            break

        logger.info('Processing frame {} ({:.2f} fps)'.format(
            frame_id, frame_id / max(1e-5, timer.total_time)))

        timer.tic()

        # 2. one detector forward for the whole batch
        outputs, img_infos = predictor.inference_batch(frames)

        # 3a. per-frame detections; gather all crops for a single Re-ID forward
        per_frame_dets = []
        crop_counts = []
        all_crops = []
        for output, img_info in zip(outputs, img_infos):
            if output is None:
                per_frame_dets.append(None)
                crop_counts.append(0)
                continue
            det = output.cpu().detach().numpy()
            det /= scale
            rows_to_remove = np.any(det[:, 0:4] < 1, axis=1)  # remove edge detection
            det = det[~rows_to_remove]
            raw_img = img_info['raw_img']
            cropped_imgs = [
                raw_img[max(0, int(y1)):min(height, int(y2)),
                        max(0, int(x1)):min(width, int(x2))]
                for x1, y1, x2, y2, _, _, _ in det
            ]
            per_frame_dets.append(det)
            crop_counts.append(len(cropped_imgs))
            all_crops.extend(cropped_imgs)

        # 3b. one Re-ID forward for every crop across the batch
        if all_crops:
            all_embs = extractor(all_crops).cpu().detach().numpy()
        else:
            all_embs = np.empty((0, 0), dtype=np.float32)

        # 4. tracker runs frame by frame, in original order (NOT batched)
        offset = 0
        overlays = []  # (raw_img, online_tlwhs, online_ids, frame_id)
        for i, (img_info, det) in enumerate(zip(img_infos, per_frame_dets)):
            cur_frame_id = frame_id + i
            raw_img = img_info['raw_img']
            if det is None:
                overlays.append((raw_img, None, None, cur_frame_id))
                continue
            n = crop_counts[i]
            embs = all_embs[offset:offset + n]
            offset += n
            online_targets = tracker.update(det, embs)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = t.last_tlwh
                tid = t.track_id
                if tlwh[2] * tlwh[3] > args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    results.append(
                        f"{cur_frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            overlays.append((raw_img, online_tlwhs, online_ids, cur_frame_id))

        timer.toc()
        batch_fps = len(frames) / max(1e-5, timer.diff)

        # 5. draw + write each frame in order
        for raw_img, online_tlwhs, online_ids, cur_frame_id in overlays:
            if online_tlwhs is None:
                online_im = raw_img
            else:
                online_im = plot_tracking(
                    raw_img, online_tlwhs, online_ids,
                    frame_id=cur_frame_id + 1, fps=batch_fps,
                )
            if args.save_result:
                vid_writer.write(online_im)

        frame_id += len(frames)

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            stop = True

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    vis_folder = osp.join(output_dir, "track_vis")
    os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = "checkpoints/best_ckpt.pth.tar"
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path = 'checkpoints/sports_model.pth.tar-60',
        device='cuda'
    )   

    imageflow_demo(predictor, extractor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
