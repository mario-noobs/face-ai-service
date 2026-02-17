import base64
import logging
import time

import numpy as np
import torch
import torch.nn as nn

from face_ai_service.models.facenet.facenet import Facenet
from face_ai_service.models.retinaface.retinaface import RetinaFace
from face_ai_service.utils.anchor_utils import Anchors
from face_ai_service.utils.bbox_utils import (
    decode,
    decode_landm,
    non_max_suppression,
    retinaface_correct_boxes,
)
from face_ai_service.utils.face_utils import alignment, letterbox_image, preprocess_input
from face_ai_service.utils.model_config import cfg_mnet, cfg_re50

logger = logging.getLogger(__name__)

# Maps algorithm names to their configuration
DETECTION_CONFIG = {
    "retinaface_mobilenet": {"cfg": cfg_mnet, "backbone": "mobilenet"},
    "retinaface_resnet50": {"cfg": cfg_re50, "backbone": "resnet50"},
}

RECOGNITION_CONFIG = {
    "facenet_mobilenet": {"backbone": "mobilenet"},
    "facenet_inception_resnetv1": {"backbone": "inception_resnetv1"},
}


class InferenceEngine:
    """Stateless face detection and recognition engine.

    Loads multiple detection/recognition model pairs at startup.
    Each request specifies which algorithm pair to use.
    """

    def __init__(self, model_store, config):
        self.config = config
        self.use_cuda = config.USE_CUDA and torch.cuda.is_available()
        self.retinaface_input_shape = config.RETINAFACE_INPUT_SHAPE
        self.facenet_input_shape = config.FACENET_INPUT_SHAPE
        self.confidence = config.RETINAFACE_CONFIDENCE
        self.nms_iou = config.RETINAFACE_NMS_IOU
        self.letterbox = config.LETTERBOX_IMAGE
        self.default_threshold = config.FACENET_THRESHOLD

        self.detection_models = {}
        self.recognition_models = {}
        self._anchors_cache = {}

        self._load_models(model_store, config.ENABLED_MODELS)

    def _load_models(self, model_store, enabled_models: str):
        pairs = [m.strip() for m in enabled_models.split(",") if m.strip()]

        for pair in pairs:
            if pair in DETECTION_CONFIG:
                self._load_detection_model(pair, model_store)
            elif pair in RECOGNITION_CONFIG:
                self._load_recognition_model(pair, model_store)
            else:
                logger.warning("Unknown model: %s, skipping", pair)

        if not self.detection_models:
            raise RuntimeError("No detection models loaded")
        if not self.recognition_models:
            raise RuntimeError("No recognition models loaded")

        logger.info(
            "Loaded detection models: %s, recognition models: %s",
            list(self.detection_models.keys()),
            list(self.recognition_models.keys()),
        )

    def _load_detection_model(self, algorithm: str, model_store):
        det_config = DETECTION_CONFIG[algorithm]
        cfg = det_config["cfg"]

        model_path = model_store.get_model_path("detection", algorithm)
        net = RetinaFace(cfg=cfg, phase="eval", pre_train=False).eval()
        state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
        net.load_state_dict(state_dict)

        if self.use_cuda:
            net = nn.DataParallel(net).cuda()

        self.detection_models[algorithm] = {"net": net, "cfg": cfg}

        # Pre-compute anchors for default input shape
        anchors = Anchors(
            cfg,
            image_size=(self.retinaface_input_shape[0], self.retinaface_input_shape[1]),
        ).get_anchors()
        self._anchors_cache[algorithm] = anchors

        logger.info("Loaded detection model: %s from %s", algorithm, model_path)

    def _load_recognition_model(self, algorithm: str, model_store):
        rec_config = RECOGNITION_CONFIG[algorithm]

        model_path = model_store.get_model_path("recognition", algorithm)
        facenet = Facenet(backbone=rec_config["backbone"], mode="predict").eval()
        state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
        facenet.load_state_dict(state_dict, strict=False)

        if self.use_cuda:
            facenet = nn.DataParallel(facenet).cuda()

        self.recognition_models[algorithm] = facenet

        logger.info("Loaded recognition model: %s from %s", algorithm, model_path)

    def get_loaded_algorithms(self) -> dict:
        return {
            "detection": list(self.detection_models.keys()),
            "recognition": list(self.recognition_models.keys()),
        }

    def _detect_faces(self, image_np: np.ndarray, algorithm: str):
        """Detect faces in an image, returning boxes_conf_landms array."""
        if algorithm not in self.detection_models:
            raise ValueError(f"Detection model not loaded: {algorithm}")

        det = self.detection_models[algorithm]
        net = det["net"]
        cfg = det["cfg"]

        old_image = image_np.copy()
        image = np.array(image_np, np.float32)
        im_height, im_width, _ = np.shape(image)

        scale = [
            np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0],
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0],
        ]

        if self.letterbox:
            image = letterbox_image(
                image,
                [self.retinaface_input_shape[1], self.retinaface_input_shape[0]],
            )
            anchors = self._anchors_cache.get(algorithm)
            if anchors is None:
                anchors = Anchors(
                    cfg,
                    image_size=(self.retinaface_input_shape[0], self.retinaface_input_shape[1]),
                ).get_anchors()
        else:
            anchors = Anchors(cfg, image_size=(im_height, im_width)).get_anchors()

        with torch.no_grad():
            image_tensor = (
                torch.from_numpy(preprocess_input(image).transpose(2, 0, 1))
                .unsqueeze(0)
                .type(torch.FloatTensor)
            )
            if self.use_cuda:
                image_tensor = image_tensor.cuda()
                anchors = anchors.cuda()

            loc, conf, landms = net(image_tensor)

            boxes = decode(loc.data.squeeze(0), anchors, cfg["variance"])
            conf = conf.data.squeeze(0)[:, 1:2]
            landms = decode_landm(landms.data.squeeze(0), anchors, cfg["variance"])

            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            if len(boxes_conf_landms) <= 0:
                return None, old_image

            if self.letterbox:
                boxes_conf_landms = retinaface_correct_boxes(
                    boxes_conf_landms,
                    np.array([self.retinaface_input_shape[0], self.retinaface_input_shape[1]]),
                    np.array([im_height, im_width]),
                )

        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks

        return boxes_conf_landms, old_image

    def _select_best_face(self, boxes_conf_landms):
        """Select the face with the largest bounding box area."""
        best_face = None
        biggest_area = 0
        for result in boxes_conf_landms:
            left, top, right, bottom = result[0:4]
            w = right - left
            h = bottom - top
            if w * h > biggest_area:
                biggest_area = w * h
                best_face = result
        return best_face

    def _encode_single_face(self, old_image, face_detection, reg_algorithm: str):
        """Crop, align, and encode a single detected face."""
        if reg_algorithm not in self.recognition_models:
            raise ValueError(f"Recognition model not loaded: {reg_algorithm}")

        facenet = self.recognition_models[reg_algorithm]

        crop_img = old_image[
            int(face_detection[1]):int(face_detection[3]),
            int(face_detection[0]):int(face_detection[2]),
        ]
        landmark = np.reshape(face_detection[5:], (5, 2)) - np.array(
            [int(face_detection[0]), int(face_detection[1])]
        )
        crop_img, _ = alignment(crop_img, landmark)

        crop_img = (
            np.array(
                letterbox_image(
                    np.uint8(crop_img),
                    (self.facenet_input_shape[1], self.facenet_input_shape[0]),
                )
            )
            / 255
        )
        crop_img = crop_img.transpose(2, 0, 1)
        crop_img = np.expand_dims(crop_img, 0)

        with torch.no_grad():
            crop_tensor = torch.from_numpy(crop_img).type(torch.FloatTensor)
            if self.use_cuda:
                crop_tensor = crop_tensor.cuda()
            face_encoding = facenet(crop_tensor)[0].cpu().numpy()

        return face_encoding

    def encode_face(
        self,
        image_np: np.ndarray,
        det_algorithm: str = "retinaface_mobilenet",
        reg_algorithm: str = "facenet_mobilenet",
    ) -> dict:
        """Detect the largest face and return its encoding.

        Returns:
            dict with keys: encoding (base64), encoding_shape, confidence, bbox,
            algorithmDet, algorithmReg
        """
        t0 = time.perf_counter()

        boxes_conf_landms, old_image = self._detect_faces(image_np, det_algorithm)
        t_det = time.perf_counter()

        if boxes_conf_landms is None:
            logger.debug("encode_face: no face detected (detection=%.0fms)", (t_det - t0) * 1000)
            return None

        best_face = self._select_best_face(boxes_conf_landms)
        if best_face is None:
            return None

        face_encoding = self._encode_single_face(old_image, best_face, reg_algorithm)
        t_enc = time.perf_counter()

        encoding_bytes = face_encoding.tobytes()
        encoding_base64 = base64.b64encode(encoding_bytes).decode("utf-8")

        bbox = [float(best_face[0]), float(best_face[1]), float(best_face[2]), float(best_face[3])]
        confidence = float(best_face[4])

        logger.debug(
            "encode_face: faces_found=%d confidence=%.4f detection=%.0fms encoding=%.0fms total=%.0fms",
            len(boxes_conf_landms), confidence,
            (t_det - t0) * 1000, (t_enc - t_det) * 1000, (t_enc - t0) * 1000,
        )

        return {
            "encoding": encoding_base64,
            "encoding_shape": list(face_encoding.shape),
            "confidence": confidence,
            "bbox": bbox,
            "algorithmDet": det_algorithm,
            "algorithmReg": reg_algorithm,
        }

    def search_faces(
        self,
        image_np: np.ndarray,
        candidates: list,
        det_algorithm: str = "retinaface_mobilenet",
        reg_algorithm: str = "facenet_mobilenet",
        threshold: float = None,
    ) -> dict:
        """Detect face, encode, and compare against candidate encodings.

        Args:
            image_np: Input image as numpy array (BGR).
            candidates: List of dicts with keys 'userId' and 'encoding' (base64).
            det_algorithm: Detection algorithm name.
            reg_algorithm: Recognition algorithm name.
            threshold: Distance threshold for match (default from config).

        Returns:
            dict with 'query_encoding' (base64) and 'matches' list.
        """
        if threshold is None:
            threshold = self.default_threshold

        logger.debug(
            "search_faces: candidates=%d threshold=%.3f det=%s reg=%s",
            len(candidates), threshold, det_algorithm, reg_algorithm,
        )

        # Encode the query face
        t0 = time.perf_counter()
        encode_result = self.encode_face(image_np, det_algorithm, reg_algorithm)
        if encode_result is None:
            return None
        t_enc = time.perf_counter()

        query_encoding_bytes = base64.b64decode(encode_result["encoding"])
        query_encoding = np.frombuffer(query_encoding_bytes, dtype=np.float32)

        # Compare against candidates
        matches = []
        for candidate in candidates:
            candidate_bytes = base64.b64decode(candidate["encoding"])
            candidate_encoding = np.frombuffer(candidate_bytes, dtype=np.float32)

            distance = float(np.linalg.norm(query_encoding - candidate_encoding))
            matched = distance <= threshold

            matches.append({
                "userId": candidate["userId"],
                "distance": round(distance, 6),
                "matched": matched,
            })

        # Sort by distance ascending
        matches.sort(key=lambda m: m["distance"])
        t_cmp = time.perf_counter()

        matched_count = sum(1 for m in matches if m["matched"])
        logger.debug(
            "search_faces: matched=%d/%d encode=%.0fms compare=%.0fms total=%.0fms",
            matched_count, len(matches),
            (t_enc - t0) * 1000, (t_cmp - t_enc) * 1000, (t_cmp - t0) * 1000,
        )

        return {
            "query_encoding": encode_result["encoding"],
            "matches": matches,
        }
