import torch
import argparse
from PIL import Image, ImageDraw
import torch.nn.functional as F
from torch import nn
import numpy as np

from datasets.hico import make_hico_transforms
from models.detr import DETRHOI

from models.backbone import build_backbone
from models.transformer import build_transformer

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

labels = [
    {"supercategory": "person", "id": 1, "name": "person"},
    {"supercategory": "vehicle", "id": 2, "name": "bicycle"},
    {"supercategory": "vehicle", "id": 3, "name": "car"},
    {"supercategory": "vehicle", "id": 4, "name": "motorcycle"},
    {"supercategory": "vehicle", "id": 5, "name": "airplane"},
    {"supercategory": "vehicle", "id": 6, "name": "bus"},
    {"supercategory": "vehicle", "id": 7, "name": "train"},
    {"supercategory": "vehicle", "id": 8, "name": "truck"},
    {"supercategory": "vehicle", "id": 9, "name": "boat"},
    {"supercategory": "outdoor", "id": 10, "name": "traffic light"},
    {"supercategory": "outdoor", "id": 11, "name": "fire hydrant"},
    {"supercategory": "outdoor", "id": 13, "name": "stop sign"},
    {"supercategory": "outdoor", "id": 14, "name": "parking meter"},
    {"supercategory": "outdoor", "id": 15, "name": "bench"},
    {"supercategory": "animal", "id": 16, "name": "bird"},
    {"supercategory": "animal", "id": 17, "name": "cat"},
    {"supercategory": "animal", "id": 18, "name": "dog"},
    {"supercategory": "animal", "id": 19, "name": "horse"},
    {"supercategory": "animal", "id": 20, "name": "sheep"},
    {"supercategory": "animal", "id": 21, "name": "cow"},
    {"supercategory": "animal", "id": 22, "name": "elephant"},
    {"supercategory": "animal", "id": 23, "name": "bear"},
    {"supercategory": "animal", "id": 24, "name": "zebra"},
    {"supercategory": "animal", "id": 25, "name": "giraffe"},
    {"supercategory": "accessory", "id": 27, "name": "backpack"},
    {"supercategory": "accessory", "id": 28, "name": "umbrella"},
    {"supercategory": "accessory", "id": 31, "name": "handbag"},
    {"supercategory": "accessory", "id": 32, "name": "tie"},
    {"supercategory": "accessory", "id": 33, "name": "suitcase"},
    {"supercategory": "sports", "id": 34, "name": "frisbee"},
    {"supercategory": "sports", "id": 35, "name": "skis"},
    {"supercategory": "sports", "id": 36, "name": "snowboard"},
    {"supercategory": "sports", "id": 37, "name": "sports ball"},
    {"supercategory": "sports", "id": 38, "name": "kite"},
    {"supercategory": "sports", "id": 39, "name": "baseball bat"},
    {"supercategory": "sports", "id": 40, "name": "baseball glove"},
    {"supercategory": "sports", "id": 41, "name": "skateboard"},
    {"supercategory": "sports", "id": 42, "name": "surfboard"},
    {"supercategory": "sports", "id": 43, "name": "tennis racket"},
    {"supercategory": "kitchen", "id": 44, "name": "bottle"},
    {"supercategory": "kitchen", "id": 46, "name": "wine glass"},
    {"supercategory": "kitchen", "id": 47, "name": "cup"},
    {"supercategory": "kitchen", "id": 48, "name": "fork"},
    {"supercategory": "kitchen", "id": 49, "name": "knife"},
    {"supercategory": "kitchen", "id": 50, "name": "spoon"},
    {"supercategory": "kitchen", "id": 51, "name": "bowl"},
    {"supercategory": "food", "id": 52, "name": "banana"},
    {"supercategory": "food", "id": 53, "name": "apple"},
    {"supercategory": "food", "id": 54, "name": "sandwich"},
    {"supercategory": "food", "id": 55, "name": "orange"},
    {"supercategory": "food", "id": 56, "name": "broccoli"},
    {"supercategory": "food", "id": 57, "name": "carrot"},
    {"supercategory": "food", "id": 58, "name": "hot dog"},
    {"supercategory": "food", "id": 59, "name": "pizza"},
    {"supercategory": "food", "id": 60, "name": "donut"},
    {"supercategory": "food", "id": 61, "name": "cake"},
    {"supercategory": "furniture", "id": 62, "name": "chair"},
    {"supercategory": "furniture", "id": 63, "name": "couch"},
    {"supercategory": "furniture", "id": 64, "name": "potted plant"},
    {"supercategory": "furniture", "id": 65, "name": "bed"},
    {"supercategory": "furniture", "id": 67, "name": "dining table"},
    {"supercategory": "furniture", "id": 70, "name": "toilet"},
    {"supercategory": "electronic", "id": 72, "name": "tv"},
    {"supercategory": "electronic", "id": 73, "name": "laptop"},
    {"supercategory": "electronic", "id": 74, "name": "mouse"},
    {"supercategory": "electronic", "id": 75, "name": "remote"},
    {"supercategory": "electronic", "id": 76, "name": "keyboard"},
    {"supercategory": "electronic", "id": 77, "name": "cell phone"},
    {"supercategory": "appliance", "id": 78, "name": "microwave"},
    {"supercategory": "appliance", "id": 79, "name": "oven"},
    {"supercategory": "appliance", "id": 80, "name": "toaster"},
    {"supercategory": "appliance", "id": 81, "name": "sink"},
    {"supercategory": "appliance", "id": 82, "name": "refrigerator"},
    {"supercategory": "indoor", "id": 84, "name": "book"},
    {"supercategory": "indoor", "id": 85, "name": "clock"},
    {"supercategory": "indoor", "id": 86, "name": "vase"},
    {"supercategory": "indoor", "id": 87, "name": "scissors"},
    {"supercategory": "indoor", "id": 88, "name": "teddy bear"},
    {"supercategory": "indoor", "id": 89, "name": "hair drier"},
    {"supercategory": "indoor", "id": 90, "name": "toothbrush"},
]

verb_classes = [
    "adjust",
    "assemble",
    "block",
    "blow",
    "board",
    "break",
    "brush_with",
    "buy",
    "carry",
    "catch",
    "chase",
    "check",
    "clean",
    "control",
    "cook",
    "cut",
    "cut_with",
    "direct",
    "drag",
    "dribble",
    "drink_with",
    "drive",
    "dry",
    "eat",
    "eat_at",
    "exit",
    "feed",
    "fill",
    "flip",
    "flush",
    "fly",
    "greet",
    "grind",
    "groom",
    "herd",
    "hit",
    "hold",
    "hop_on",
    "hose",
    "hug",
    "hunt",
    "inspect",
    "install",
    "jump",
    "kick",
    "kiss",
    "lasso",
    "launch",
    "lick",
    "lie_on",
    "lift",
    "light",
    "load",
    "lose",
    "make",
    "milk",
    "move",
    "no_interaction",
    "open",
    "operate",
    "pack",
    "paint",
    "park",
    "pay",
    "peel",
    "pet",
    "pick",
    "pick_up",
    "point",
    "pour",
    "pull",
    "push",
    "race",
    "read",
    "release",
    "repair",
    "ride",
    "row",
    "run",
    "sail",
    "scratch",
    "serve",
    "set",
    "shear",
    "sign",
    "sip",
    "sit_at",
    "sit_on",
    "slide",
    "smell",
    "spin",
    "squeeze",
    "stab",
    "stand_on",
    "stand_under",
    "stick",
    "stir",
    "stop_at",
    "straddle",
    "swing",
    "tag",
    "talk_on",
    "teach",
    "text_on",
    "throw",
    "tie",
    "toast",
    "train",
    "turn",
    "type_on",
    "walk",
    "wash",
    "watch",
    "wave",
    "wear",
    "wield",
    "zip",
]


def getImage(imagePath="download.png"):
    image = Image.open(imagePath).convert("RGB")
    target = {}
    w, h = image.size
    target["orig_size"] = torch.as_tensor([int(h), int(w)])
    target["size"] = torch.as_tensor([int(h), int(w)])
    transformFunc = make_hico_transforms("val")
    img, tar = transformFunc(image, target)
    return img, tar


def getModel(args):
    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETRHOI(
        backbone,
        transformer,
        num_obj_classes=80,
        num_verb_classes=117,
        num_queries=100,
        aux_loss=args.aux_loss,
    )

    return model


class PostProcessHOI(nn.Module):
    # def __init__(self, num_queries, subject_category_id, correct_mat):
    def __init__(self, subject_category_id):
        super().__init__()
        self.max_hois = 100

        self.subject_category_id = subject_category_id

        # correct_mat = np.concatenate((correct_mat, np.ones((correct_mat.shape[0], 1))), axis=1)
        # self.register_buffer('correct_mat', torch.from_numpy(correct_mat))

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = (
            outputs["pred_obj_logits"],
            outputs["pred_verb_logits"],
            outputs["pred_sub_boxes"],
            outputs["pred_obj_boxes"],
        )

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = F.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        verb_scores = out_verb_logits.sigmoid()

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(
            verb_scores.device
        )
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for os, ol, vs, sb, ob in zip(
            obj_scores, obj_labels, verb_scores, sub_boxes, obj_boxes
        ):
            output = []

            for a, b, c, d, e in zip(
                os.to("cpu").numpy(),
                ol.to("cpu").numpy(),
                vs.to("cpu").numpy(),
                sb.to("cpu").numpy(),
                ob.to("cpu").numpy(),
            ):
                data = {
                    "score": a,
                    "label": b,
                    "verb": verb_classes[np.ndarray.argmax(c).item()],
                    "hbox": d,
                    "obox": e,
                }
                output.append(data)

            results.append(output)

        return results


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--lr_drop", default=100, type=int)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )
    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--num_queries", default=100, type=int, help="Number of query slots"
    )
    parser.add_argument("--pre_norm", action="store_true")

    # * Segmentation
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Train segmentation head if the flag is provided",
    )

    # HOI
    parser.add_argument(
        "--hoi", action="store_true", help="Train for HOI if the flag is provided"
    )
    parser.add_argument(
        "--num_obj_classes", type=int, default=80, help="Number of object classes"
    )
    parser.add_argument(
        "--num_verb_classes", type=int, default=117, help="Number of verb classes"
    )
    parser.add_argument(
        "--pretrained", type=str, default="", help="Pretrained model path"
    )
    parser.add_argument("--subject_category_id", default=0, type=int)
    parser.add_argument(
        "--verb_loss_type",
        type=str,
        default="focal",
        help="Loss type for the verb classification",
    )

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=1,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_obj_class",
        default=1,
        type=float,
        help="Object class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_verb_class",
        default=1,
        type=float,
        help="Verb class coefficient in the matching cost",
    )

    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--obj_loss_coef", default=1, type=float)
    parser.add_argument("--verb_loss_coef", default=1, type=float)
    parser.add_argument(
        "--eos_coef",
        default=0.1,
        type=float,
        help="Relative classification weight of the no-object class",
    )

    # dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--coco_path", type=str)
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--hoi_path", type=str)

    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=2, type=int)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    return parser


def generate(results, targets, missing_category_id=80):
    detections = []

    for img_results, img_targets in zip(results, targets):
        for hoi in img_results["hoi_prediction"]:
            detection = {
                "person_box": img_results["predictions"][hoi["subject_id"]][
                    "bbox"
                ].tolist(),
            }
            if (
                img_results["predictions"][hoi["object_id"]]["category_id"]
                == missing_category_id
            ):
                object_box = [np.nan, np.nan, np.nan, np.nan]
            else:
                object_box = img_results["predictions"][hoi["object_id"]][
                    "bbox"
                ].tolist()
            cut_agent = 0
            hit_agent = 0
            eat_agent = 0
            for idx, score in zip(hoi["category_id"], hoi["score"]):
                verb_class = verb_classes[idx]
                score = score.item()
                if len(verb_class.split("_")) == 1:
                    detection["{}_agent".format(verb_class)] = score
                elif "cut_" in verb_class:
                    detection[verb_class] = object_box + [score]
                    cut_agent = score if score > cut_agent else cut_agent
                elif "hit_" in verb_class:
                    detection[verb_class] = object_box + [score]
                    hit_agent = score if score > hit_agent else hit_agent
                elif "eat_" in verb_class:
                    detection[verb_class] = object_box + [score]
                    eat_agent = score if score > eat_agent else eat_agent
                else:
                    detection[verb_class] = object_box + [score]
                    detection[
                        "{}_agent".format(
                            verb_class.replace("_obj", "").replace("_instr", "")
                        )
                    ] = score
            detection["cut_agent"] = cut_agent
            detection["hit_agent"] = hit_agent
            detection["eat_agent"] = eat_agent
            detections.append(detection)

    return detections


def run(imagePath="download.png"):
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    image, target = getImage(imagePath)
    targets = [{k: (v.to(device) if k != "id" else v) for k, v in target.items()}]

    model = getModel(args)
    model.load_state_dict(
        torch.load(
            "logs/qpic_resnet50_hico.pth",
            map_location=torch.device("cpu"),
        )["model"]
    )
    model_result = model([image])

    postprocessor = PostProcessHOI(args.subject_category_id)
    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    results = postprocessor(model_result, orig_target_sizes)
    result = results[0]

    final = []

    for data in result:
        # hbox = data["hbox"]
        # obox = data["obox"]
        # label = data["label"]
        verb = data["verb"]
        score = data["score"]

        index = -1
        for j, d in enumerate(final):
            if d["verb"] == verb:
                if d["score"] < score:
                    final[j] = data
                index = j
                break

        if index == -1:
            final.append(data)

    final.sort(key=lambda v: v["score"], reverse=True)
    return final


if __name__ == "__main__":
    imagePath = "download.jpg"
    result = run(imagePath)

    image = Image.open(imagePath)
    draw = ImageDraw.Draw(image)

    for data in result:
        hbox = data["hbox"]
        obox = data["obox"]
        label = labels[data["label"]]["name"]
        verb = data["verb"]

        draw.rectangle(hbox, width=2, outline="green")
        draw.rectangle(obox, width=2, outline="red")
        draw.text([hbox[0].item(), hbox[1].item()], fill="green", text=verb)
        draw.text([obox[0].item(), obox[1].item()], fill="red", text=label)

    imageFileExtension = imagePath.split(".")[-1]
    image.save(f"output.{imageFileExtension}")
