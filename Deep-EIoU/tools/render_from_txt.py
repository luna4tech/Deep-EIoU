"""Render an annotated tracking video from a source video + a MOT-format .txt.

This is the companion to `tools/demo.py --no_save_video`: run the (faster)
tracking pass without drawing anything, then rebuild the annotated video offline
whenever you actually need to look at it.

The .txt is the standard MOT format written by demo.py:

    frame,id,x,y,w,h,score,-1,-1,-1

where (x, y) is the top-left corner and (w, h) the box size. `frame` is 0-based,
matching demo.py's output; pass --one_indexed for 1-based files (e.g. the MOT
challenge / SportsMOT convention).

Only cv2 + numpy are needed -- no torch/GPU -- so this can run anywhere.

Example:
    python tools/render_from_txt.py \
        --path /path/to/M59-5min-1.mp4 \
        --txt  YOLOX_outputs/yolox_x_ch_sportsmot/track_vis/2026_06_09_10_18_01.txt
"""

import argparse
import importlib.util
import logging
import os
import os.path as osp
from collections import defaultdict

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("render_from_txt")


def load_plot_tracking():
    """Load plot_tracking from yolox/utils/visualize.py directly.

    Loading the file by path avoids importing the heavy `yolox` package (which
    pulls in torch); visualize.py itself only needs cv2 + numpy.
    """
    this_dir = osp.dirname(osp.abspath(__file__))
    vis_path = osp.join(this_dir, "..", "yolox", "utils", "visualize.py")
    if osp.isfile(vis_path):
        spec = importlib.util.spec_from_file_location("deepeiou_visualize", vis_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.plot_tracking
    # Fallback: rely on the package being importable (matches demo.py).
    from yolox.utils.visualize import plot_tracking
    return plot_tracking


def make_parser():
    parser = argparse.ArgumentParser("Render annotated video from MOT .txt")
    parser.add_argument("--path", required=True, help="path to the source video")
    parser.add_argument("--txt", required=True, help="path to the MOT-format results .txt")
    parser.add_argument(
        "--save_path",
        default=None,
        help="output video path (default: <txt_dir>/<txt_name>_rendered.mp4)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="output frame rate (default: source video fps)",
    )
    parser.add_argument(
        "--one_indexed",
        action="store_true",
        help="treat the frame column as 1-based (MOT/SportsMOT). demo.py output is 0-based.",
    )
    return parser


def parse_results(txt_path):
    """Parse a MOT .txt into {frame_index: (tlwhs, ids, scores)}."""
    frames = defaultdict(lambda: ([], [], []))
    n_rows = 0
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split(",")
            if len(fields) < 6:
                continue
            frame = int(float(fields[0]))
            tid = int(float(fields[1]))
            x, y, w, h = (float(v) for v in fields[2:6])
            score = float(fields[6]) if len(fields) > 6 and fields[6] not in ("", "-1") else 1.0
            tlwhs, ids, scores = frames[frame]
            tlwhs.append((x, y, w, h))
            ids.append(tid)
            scores.append(score)
            n_rows += 1
    return frames, n_rows


def main():
    args = make_parser().parse_args()

    if not osp.isfile(args.path):
        raise FileNotFoundError("source video not found: {}".format(args.path))
    if not osp.isfile(args.txt):
        raise FileNotFoundError("results txt not found: {}".format(args.txt))

    plot_tracking = load_plot_tracking()

    frames, n_rows = parse_results(args.txt)
    logger.info("loaded {} boxes across {} annotated frames from {}".format(
        n_rows, len(frames), args.txt))

    cap = cv2.VideoCapture(args.path)
    if not cap.isOpened():
        raise RuntimeError("could not open video: {}".format(args.path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = args.fps if args.fps else (src_fps if src_fps and src_fps > 0 else 30.0)

    if args.save_path:
        save_path = args.save_path
    else:
        stem = osp.splitext(osp.basename(args.txt))[0]
        save_path = osp.join(osp.dirname(osp.abspath(args.txt)), stem + "_rendered.mp4")
    os.makedirs(osp.dirname(osp.abspath(save_path)), exist_ok=True)

    writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height)
    )
    logger.info("rendering -> {} ({}x{} @ {:.2f} fps)".format(save_path, width, height, out_fps))

    frame_idx = 0
    matched_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        key = frame_idx + 1 if args.one_indexed else frame_idx
        if key in frames:
            tlwhs, ids, _ = frames[key]
            online_im = plot_tracking(
                frame, tlwhs, ids, frame_id=frame_idx + 1, fps=out_fps
            )
            matched_frames += 1
        else:
            online_im = frame
        writer.write(online_im)
        frame_idx += 1
        if total > 0 and frame_idx % 100 == 0:
            logger.info("rendered {}/{} frames".format(frame_idx, total))

    cap.release()
    writer.release()

    logger.info("done: wrote {} frames ({} with annotations) to {}".format(
        frame_idx, matched_frames, save_path))
    if matched_frames == 0 and n_rows > 0:
        logger.warning(
            "no .txt frames matched any video frame -- check the frame indexing "
            "(try toggling --one_indexed) or that the .txt belongs to this video."
        )


if __name__ == "__main__":
    main()
