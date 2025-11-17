#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import YOLO

from sam2.lib.v2_sam.sam_v2_model import SAMV2Model
from sam2.lib.make_sam import make_sam_from_state_dict
from sam2.lib.helpers.video_data_storage import SAM2VideoObjectResults

import cv2
import numpy as np
import torch
from collections import defaultdict, deque
from time import perf_counter
import torch.nn.functional as F
import colorsys
import random
from scipy.optimize import linear_sum_assignment
import itertools  # 병합/가려짐용


CONF = 0.7

class SAM2TRACKER:
    def __init__(self):
        self.param_init()
        self.buffer_init()
        self.sam2_model_init()

    def param_init(self):
        """
        초기 파라미터 설정
        """
        self.frame_idx       = 1
        self.image_w         = 1024
        self.image_h         = 1024
        self.obj_id          = 1

        # 시각화/경로 관련
        self.angle           = 0.618033988749895
        self.hue             = random.random()
        self.max_path_length = 30           # 경로는 최근 30프레임만 유지

        # 추적 품질 관련
        self.dist_thress        = 20        # 정상 이동으로 볼 최대 거리
        self.sudden_jump_thresh = 40        # 이 이상 점프하면 mistrack
        self.tracking_miss      = 30        # 이 횟수 이상 miss면 트랙 삭제
        self.obj_score_thr      = 0.0       # SAM2 score threshold

        # YOLO → SAM2 프롬프트 기반 새 트랙 판정용 파라미터
        self.untracked_region_thresh = 0.4  # M_non: 비점유 영역 비율 임계값
        self.M_confirm               = 2    # 최근 N프레임 중 M번 True면 새 트랙 확정
        self.N_window                = 10   # 후보 히스토리 길이
        self.candidate_lost_thresh   = 5   # 이 프레임 수 동안 안 보이면 후보 삭제
        self.cand_match_dist         = 2.0  # 후보와 새 프롬프트 중심 간 최대 거리 (픽셀)

        # 병합/가려짐 처리 파라미터
        self.merge_iou_thresh        = 0.1  # 이 IoU 이상이면 병합/가려짐으로 간주

        # 프롬프트 재활용(기존 트랙과 거리 기반)용
        self.track_reuse_dist        =5.0 # 기존 트랙 중심과 이 거리 이하면 새 candidate로 안 만듦

        fps    = 30
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.out_tracked = cv2.VideoWriter(
            '../ulsan_day1_어선군.avi', fourcc, fps, (self.image_w, self.image_h)
        )

    def buffer_init(self):
        """
        버퍼 초기화
        """
        # 확정 트랙 관련
        self.generated_hues   = []
        self.track_points     = defaultdict(list)              # obj_id -> [(x,y), ...]
        self.mask_list        = {}                            # obj_id -> binary mask
        self.ptr_list         = {}                            # obj_id -> pointer tensor
        self.miss_counts      = {}                            # obj_id -> int
        self.memory_list      = defaultdict(SAM2VideoObjectResults.create)
        self.mot_log          = []

        # 트랙 상태: 'active', 'occluded', 'lost' 등 표시 가능
        self.track_status     = {}                            # obj_id -> str

        # 트랙별 색상
        self.object_color_map = {}                            # obj_id -> (B,G,R)

        # 새 물체 후보(candidates)
        # cand_id -> dict(mask, emb, ptr, ctr, history, last_seen_frame, seen_this_frame)
        self.candidates   = {}
        self.next_cand_id = 0
        self.raw_detections = []

    def sam2_model_init(self):
        """
        SAM2 모델 초기화
        """
        model_path = '/home/anhong/ws_radar_tracking/src/tracker/checkpoints/sam2.1_hiera_tiny.pt'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_config_dict, self.sammodel = make_sam_from_state_dict(model_path)
        assert isinstance(self.sammodel, SAMV2Model), "Only SAMv2 models are supported for video predictions!"
        self.sammodel.to(device=device)
        self.sammodel.eval()

    # ------------------------------------------------------------------
    # 유틸 함수
    # ------------------------------------------------------------------

    def mask_to_tight_bbox(self, mask):
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = (mask > 0).astype(np.uint8) * 255
        cleaned_mask = cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )
        contours, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        return x, y, w, h

    def bbox_iou(self, boxA, boxB):
        """ boxA, boxB: (x1,y1,x2,y2) 픽셀단위 """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        inter = interW * interH
        areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
        union = areaA + areaB - inter
        return inter/union if union > 0 else 0.0

    def mask_iou_matrix(self, prev_masks, curr_masks):
        Np, Nc = len(prev_masks), len(curr_masks)
        iou = np.zeros((Np, Nc), dtype=float)
        for i, pm in enumerate(prev_masks):
            for j, cm in enumerate(curr_masks):
                inter = np.logical_and(pm, cm).sum()
                union = np.logical_or(pm, cm).sum()
                iou[i, j] = inter / union if union > 0 else 0.0
        return iou

    def mask_iou(self, mask_a, mask_b):
        """
        두 이진 마스크 간 IoU 계산
        """
        inter = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        return inter / union if union > 0 else 0.0

    def mask_centroid(self, mask):
        ys, xs = np.where(mask)
        if ys.size == 0:
            return None
        cx = float(xs.mean())
        cy = float(ys.mean())
        return cx, cy

    def get_next_color(self):
        """
        트랙마다 고유 색상 생성 (HSV 골든앵글 회전, BGR로 반환)
        """
        self.hue = (self.hue + self.angle) % 1.0
        self.generated_hues.append(self.hue)
        r, g, b = colorsys.hsv_to_rgb(self.hue, 1.0, 1.0)
        color = (int(b * 255), int(g * 255), int(r * 255))  # OpenCV: BGR
        return color

    # ------------------------------------------------------------------
    # 헝가리안: 현재 프롬프트 vs 기존 트랙 매칭 (mask IoU 기반)
    # ------------------------------------------------------------------

    def associate_with_hungarian(self, curr_masks, iou_thresh=0.3):
        """
        기존 트랙(self.mask_list)과 현재 프롬프트(curr_masks)를
        헝가리안으로 매칭해서
          - matches: [(prev_id, curr_idx), ...]
          - unmatched_curr: [idx, ...]
        반환
        """
        prev_ids = list(self.mask_list.keys())
        if not prev_ids or len(curr_masks) == 0:
            return [], list(range(len(curr_masks)))

        prev_masks = [self.mask_list[i] for i in prev_ids]
        iou = self.mask_iou_matrix(prev_masks, curr_masks)
        cost = 1.0 - iou

        row_idx, col_idx = linear_sum_assignment(cost)

        matches = []
        used_curr = set()
        for r, c in zip(row_idx, col_idx):
            if iou[r, c] >= iou_thresh:
                matches.append((prev_ids[r], c))
                used_curr.add(c)

        unmatched = [j for j in range(len(curr_masks)) if j not in used_curr]
        return matches, unmatched

    # ------------------------------------------------------------------
    # 병합/가려짐 처리 (Strategy 2)
    # ------------------------------------------------------------------

    def resolve_conflicts(self, propagated_results):
        """
        2단계 (Strategy 2): 예측 기반 분배 (병합/가려짐 해결)
        - 'tracked' 상태인 트랙들의 마스크를 비교하여 겹치면(병합) 처리
        """
        resolved_results = propagated_results.copy()

        # 'tracked' 상태인 트랙만
        tracked_ids = [obj_id for obj_id, res in resolved_results.items()
                       if res['status'] == 'tracked']

        # 모든 트랙 쌍에 대해 IoU 계산
        for id_a, id_b in itertools.combinations(tracked_ids, 2):
            if resolved_results[id_a]['status'] != 'tracked' or resolved_results[id_b]['status'] != 'tracked':
                continue

            mask_a = resolved_results[id_a]['mask']
            mask_b = resolved_results[id_b]['mask']

            iou = self.mask_iou(mask_a, mask_b)

            if iou > self.merge_iou_thresh:
                # 병합/가려짐 발생 → logits score(추적 신뢰도)를 비교
                score_a = resolved_results[id_a]['score']
                score_b = resolved_results[id_b]['score']

                if score_a > score_b:
                    loser_id = id_b
                    winner_id = id_a
                else:
                    loser_id = id_a
                    winner_id = id_b

                print(
                    f"[Strategy 2] Conflict: Track {winner_id} (Score {max(score_a, score_b):.2f}) wins "
                    f"against Track {loser_id} (Score {min(score_a, score_b):.2f})"
                )

                # 패자 트랙은 메모리 갱신 방지 + 상태를 occluded로
                resolved_results[loser_id]['update_memory'] = False
                resolved_results[loser_id]['status'] = 'occluded'

        return resolved_results

    # ------------------------------------------------------------------
    # YOLO → SAM2 프롬프트 처리 + 새 트랙 판정 (M/N + M_non)
    # ------------------------------------------------------------------

    def prompt_embedding(self, image, frame_idx, encoded_img, bboxes):
        H, W = image.shape[:2]

        # 1) YOLO bbox → SAM2 initialize 로부터 mask / embedding / pointer 추출
        curr_masks, curr_embs, curr_ptrs, curr_ctrs = [], [], [], []
        for box in bboxes:
            prompts = ([box], [], [])
            best_mask_pred, init_enc, init_ptr = \
                self.sammodel.initialize_video_masking(encoded_img, *prompts)

            logit_mask = F.interpolate(
                best_mask_pred, size=(H, W),
                mode="bilinear", align_corners=False
            )
            mask = (logit_mask > 0).cpu().numpy().squeeze()
            if not np.any(mask):
                continue

            cen = self.mask_centroid(mask)
            if cen is None:
                continue

            curr_masks.append(mask)
            curr_embs.append(init_enc)
            curr_ptrs.append(init_ptr)
            curr_ctrs.append(cen)

        # 후보들 seen_this_frame 플래그 초기화
        for cand in self.candidates.values():
            cand['seen_this_frame'] = False

        if len(curr_masks) == 0:
            # 새 프롬프트가 없는 프레임: 후보 lifecycle만 업데이트
            self._update_candidates_lifecycle(frame_idx)
            return

        # 2) 헝가리안 매칭: 기존 트랙 vs 현재 프롬프트
        matches, unmatched_curr = self.associate_with_hungarian(curr_masks)
        # matches 는 "이미 존재하는 트랙"에 해당하는 프롬프트이므로
        # 여기서는 새 객체 후보 생성에만 관심 → unmatched만 사용

        # 3) M_non 계산: 기존 트랙이 점유한 영역 vs 비점유 영역
        occupied_mask = np.zeros((H, W), dtype=bool)
        for _, m in self.mask_list.items():
            if m is None:
                continue
            occupied_mask = np.logical_or(occupied_mask, m)

        if occupied_mask.any():
            m_non = np.logical_not(occupied_mask)
        else:
            m_non = np.ones((H, W), dtype=bool)

        # 4) unmatched 프롬프트들 → "기존 트랙 재사용" or "새 candidate"
        for j in unmatched_curr:
            new_mask = curr_masks[j]
            new_emb  = curr_embs[j]
            new_ptr  = curr_ptrs[j]
            new_ctr  = curr_ctrs[j]

            det_area = new_mask.sum()
            if det_area == 0:
                continue

            # 4-0) 먼저 기존 트랙과의 거리 기반 재사용 시도
            reuse_track_id = None
            best_dist = None
            for tid, pts in self.track_points.items():
                if len(pts) == 0:
                    continue
                last_x, last_y = pts[-1]
                dx = new_ctr[0] - last_x
                dy = new_ctr[1] - last_y
                dist = (dx*dx + dy*dy) ** 0.5
                if dist <= self.track_reuse_dist and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    reuse_track_id = tid

            if reuse_track_id is not None:
                # 이 프롬프트는 새 물체가 아니라 기존 트랙의 재관측으로 간주
                print(f"[REUSE] frame {frame_idx}, prompt {j} → track {reuse_track_id} (dist={best_dist:.1f})")
                # IoU 기반 매칭을 위해 해당 트랙의 mask만 최신으로 업데이트
                self.mask_list[reuse_track_id] = new_mask
                # 여기서는 candidate 생성 X
                continue

            # 4-1) M_non 비율 계산
            overlap_area  = np.logical_and(new_mask, m_non).sum()
            overlap_ratio = overlap_area / det_area

            print(f"[PROMPT] frame {frame_idx}, curr {j}, M_non_overlap={overlap_ratio:.3f}, ctr={new_ctr}")

            # M_non 기준: 기존 트랙과 너무 겹치면 새 물체로 보지 않음
            if overlap_ratio < self.untracked_region_thresh:
                print(f"  -> REJECT by M_non (overlap={overlap_ratio:.2f} < {self.untracked_region_thresh:.2f})")
                continue

            # 4-2) 기존 candidate와 근접한 것이 있는지 거리 기준으로 확인
            best_cand_id = None
            best_dist    = None
            for cand_id, cand in self.candidates.items():
                c_ctr = cand['ctr']
                dx = new_ctr[0] - c_ctr[0]
                dy = new_ctr[1] - c_ctr[1]
                dist = (dx*dx + dy*dy) ** 0.5
                if dist <= self.cand_match_dist and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    best_cand_id = cand_id

            if best_cand_id is None:
                # 새 candidate 생성
                cand_id = self.next_cand_id
                self.next_cand_id += 1

                self.candidates[cand_id] = {
                    'mask': new_mask,
                    'emb':  new_emb,
                    'ptr':  new_ptr,
                    'ctr':  new_ctr,
                    'history': deque([True], maxlen=self.N_window),
                    'last_seen_frame': frame_idx,
                    'seen_this_frame': True
                }
                print(f"[CAND-CREATE] cand {cand_id} from curr {j}, ctr={new_ctr}, M_non={overlap_ratio:.2f}")
            else:
                # 기존 candidate 업데이트
                cand = self.candidates[best_cand_id]
                cand['mask'] = new_mask
                cand['emb']  = new_emb
                cand['ptr']  = new_ptr
                cand['ctr']  = new_ctr
                cand['history'].append(True)
                cand['last_seen_frame'] = frame_idx
                cand['seen_this_frame'] = True
                print(f"[CAND-UPDATE] cand {best_cand_id} <- curr {j}, dist={best_dist:.1f}")

        # 5) 후보 lifecycle & M/N 승격 처리
        self._update_candidates_lifecycle(frame_idx)
        self._promote_candidates_to_tracks(frame_idx)

    def _update_candidates_lifecycle(self, frame_idx):
        """
        후보(candidates)에 대해:
          - 이번 프레임에 안 보인 경우 history에 False 추가
          - 오래 안 보인 후보 삭제
        """
        to_delete = []
        for cand_id, cand in self.candidates.items():
            if not cand['seen_this_frame']:
                cand['history'].append(False)

            if (frame_idx - cand['last_seen_frame']) >= self.candidate_lost_thresh:
                print(f"[CAND-DELETE] cand {cand_id} removed (last_seen={cand['last_seen_frame']}, now={frame_idx})")
                to_delete.append(cand_id)

        for cand_id in to_delete:
            del self.candidates[cand_id]

    def _promote_candidates_to_tracks(self, frame_idx):
        """
        M/N 로직:
          - 최근 N프레임 history에서 True 합이 M 이상인 후보만 새 트랙으로 승격
        """
        promote_ids = []
        for cand_id, cand in self.candidates.items():
            hist_sum = sum(cand['history'])
            print(f"[CAND-MN] cand {cand_id}: hist_sum={hist_sum}, len={len(cand['history'])}")
            if hist_sum >= self.M_confirm:
                promote_ids.append(cand_id)

        for cand_id in promote_ids:
            cand = self.candidates[cand_id]
            new_id = self.obj_id
            self.obj_id += 1

            # SAM2 메모리 초기화
            self.memory_list[new_id].store_prompt_result(
                frame_idx,
                cand['emb'],
                cand['ptr']
            )
            # 트랙 버퍼 초기화
            self.track_points[new_id] = [tuple(cand['ctr'])]
            self.mask_list[new_id]    = cand['mask']
            self.ptr_list[new_id]     = cand['ptr']
            self.miss_counts[new_id]  = 0
            self.track_status[new_id] = 'active'

            # 색상 할당
            self.object_color_map[new_id] = self.get_next_color()

            print(f"[CAND-PROMOTE] cand {cand_id} -> track {new_id}, ctr={cand['ctr']}, hist_sum={sum(cand['history'])}")

            del self.candidates[cand_id]

    # ------------------------------------------------------------------
    # SAM2 기반 tracking (+ 병합/가려짐 해소, 상태 업데이트)
    # ------------------------------------------------------------------

    def tracking(self, frame, frame_idx, encoded_img):
        """
        SAM2 기반 tracking (+ 병합/가려짐 해소, 상태 업데이트)
        1) 모든 트랙에 대해 step_video_masking → propagated_results 생성
        2) resolve_conflicts로 병합/가려짐 처리
        3) sudden_jump / dist_thress / miss_count / MOT 로그 업데이트
        """
        H, W = frame.shape[:2]

        # 0) SAM2 전파 결과 모으기
        propagated_results = {}

        for obj_key_name, obj_memory in self.memory_list.items():
            obj_score, best_mask_idx, mask_preds, mem_enc, obj_ptr = \
                self.sammodel.step_video_masking(encoded_img, **obj_memory.to_dict())

            obj_score = obj_score.item()

            # score가 너무 낮으면 이번 프레임은 "lost"
            if obj_score < self.obj_score_thr:
                propagated_results[obj_key_name] = {
                    'status': 'lost',
                    'score': obj_score
                }
                continue

            # 마스크 업샘플링
            obj_mask = torch.nn.functional.interpolate(
                mask_preds[:, best_mask_idx, :, :],
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            obj_mask_binary = (obj_mask > 0.0).cpu().numpy().squeeze()

            if not np.any(obj_mask_binary):
                propagated_results[obj_key_name] = {
                    'status': 'lost',
                    'score': obj_score
                }
                continue

            propagated_results[obj_key_name] = {
                'status': 'tracked',
                'score': obj_score,
                'mask': obj_mask_binary,
                'mem_enc': mem_enc,
                'ptr': obj_ptr,
                'update_memory': True
            }

        # 1) 병합/가려짐 처리
        resolved_results = self.resolve_conflicts(propagated_results)

        # 2) 최종 결과를 이용해 트랙 상태 업데이트
        for obj_key_name, obj_memory in self.memory_list.items():
            pts = self.track_points[obj_key_name]
            res = resolved_results.get(obj_key_name, None)

            # SAM2가 아예 못 찾았거나(lost), 결과 자체가 없으면 miss
            if res is None or res['status'] == 'lost':
                self.miss_counts[obj_key_name] = self.miss_counts.get(obj_key_name, 0) + 1
                self.track_status[obj_key_name] = 'lost'
                continue

            # 병합/가려짐에서 진 트랙(occluded): 메모리/마스크 업데이트하지 않고, miss_count도 올리지 않음(아이디 유지)
            if res['status'] == 'occluded':
                print(f"[OCCLUDED] track {obj_key_name} occluded this frame")
                self.track_status[obj_key_name] = 'occluded'
                continue

            # 여기까지 왔다면 status == 'tracked'
            obj_mask_binary = res['mask']
            self.track_status[obj_key_name] = 'active'

            ys, xs = np.where(obj_mask_binary)
            if not ys.size:
                self.miss_counts[obj_key_name] = self.miss_counts.get(obj_key_name, 0) + 1
                self.track_status[obj_key_name] = 'lost'
                continue

            cx = int(xs.mean())
            cy = int(ys.mean())

            if len(pts) > 0:
                prev_x, prev_y = pts[-1]
            else:
                prev_x, prev_y = cx, cy

            dist = np.hypot(cx - prev_x, cy - prev_y)

            # 🔥 말도 안 되게 크게 점프한 경우 → mistrack으로 보고 이번 프레임은 버린다
            if dist > self.sudden_jump_thresh:
                print(f"[JUMP] track {obj_key_name}: dist={dist:.1f}px → mistrack, ignore update")
                self.miss_counts[obj_key_name] = self.miss_counts.get(obj_key_name, 0) + 1
                continue

            # 정상 범위 내 이동 → SAM2 결과 반영
            if res.get('update_memory', True):
                obj_memory.store_result(frame_idx, res['mem_enc'], res['ptr'])

            self.mask_list[obj_key_name] = obj_mask_binary
            self.ptr_list[obj_key_name]  = res['ptr']

            # 경로 업데이트 (최근 max_path_length 프레임만 유지)
            pts.append((cx, cy))
            if len(pts) > self.max_path_length:
                while len(pts) > self.max_path_length:
                    pts.pop(0)

            # dist_thress 기반으로 miss 카운트 조정
            if dist <= self.dist_thress:
                self.miss_counts[obj_key_name] = 0
            else:
                # dist_thress < dist <= sudden_jump_thresh
                self.miss_counts[obj_key_name] = self.miss_counts.get(obj_key_name, 0) + 1

            # MOT 로그 기록
            bbox = self.mask_to_tight_bbox(obj_mask_binary)
            if bbox:
                x, y, w, h = bbox
                line = f"{frame_idx},{obj_key_name},{x},{y},{w},{h},{res['score']:.3f},{1},{1.0}"
                self.mot_log.append(line)

    # ------------------------------------------------------------------
    # 가시화 전용: 바운딩박스(서로 다른 색) + 경로(30프레임)
    # ------------------------------------------------------------------

    def visualize_tracks(self, frame):
        vis_frame = frame.copy()

        DETECTION_COLOR = (0, 255, 255)  # BGR 노란색
        for det in self.raw_detections:
            x1, y1, x2, y2, conf = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(
                vis_frame, (x1, y1), (x2, y2),
                DETECTION_COLOR, thickness=1, lineType=cv2.LINE_AA
            )
            cv2.putText(
                vis_frame, f"{conf:.2f}",
                (x1, min(y2 + 15, vis_frame.shape[0]-2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, DETECTION_COLOR, 1, cv2.LINE_AA
            )

        for obj_key_name, pts in self.track_points.items():
            # 해당 트랙에 색상 없으면 생성
            if obj_key_name not in self.object_color_map:
                self.object_color_map[obj_key_name] = self.get_next_color()
            color = self.object_color_map[obj_key_name]

            # 현재 마스크에서 bbox 그리기
            mask = self.mask_list.get(obj_key_name, None)
            if mask is not None and np.any(mask):
                bbox = self.mask_to_tight_bbox(mask)
                if bbox:
                    x, y, w, h = bbox
                    x_min, y_min = x, y
                    x_max, y_max = x + w, y + h
                    cv2.rectangle(
                        vis_frame,
                        (x_min, y_min),
                        (x_max, y_max),
                        color,
                        thickness=1,
                        lineType=cv2.LINE_AA
                    )

            # 경로 polyline (최근 max_path_length 프레임)
            if len(pts) > 1:
                pts_arr = np.array(pts, np.int32).reshape(-1, 1, 2)
                cv2.polylines(
                    vis_frame,
                    [pts_arr],
                    isClosed=False,
                    color=color,
                    thickness=1,
                    lineType=cv2.LINE_AA
                )

            # 텍스트 라벨 (마지막 위치 근처)
            if len(pts) > 0:
                x_last, y_last = pts[-1]
                label_pos = (int(x_last - 5), int(max(y_last - 10, 0)))
                cv2.putText(
                    vis_frame,
                    str(obj_key_name),
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        cv2.imshow("Video Segmentation Result - q to quit", vis_frame)
        self.out_tracked.write(vis_frame)
        key = cv2.waitKey(1) & 0xFF

    def track_lifecycle(self):
        delet_id = []
        for obj_key_name, count in self.miss_counts.items():
            if count >= self.tracking_miss:
                print(f"track {obj_key_name} miss count: {count}")
                delet_id.append(obj_key_name)

        for obj_key_name in delet_id:
            if obj_key_name in self.track_points:
                del self.track_points[obj_key_name]
            if obj_key_name in self.mask_list:
                del self.mask_list[obj_key_name]
            if obj_key_name in self.ptr_list:
                del self.ptr_list[obj_key_name]
            if obj_key_name in self.miss_counts:
                del self.miss_counts[obj_key_name]
            if obj_key_name in self.memory_list:
                del self.memory_list[obj_key_name]
            if obj_key_name in self.object_color_map:
                del self.object_color_map[obj_key_name]
            if obj_key_name in self.track_status:
                del self.track_status[obj_key_name]
            print(f"track {obj_key_name} deleted")

    # ------------------------------------------------------------------
    # 메인 파이프라인
    # ------------------------------------------------------------------

    def process_segmentation(self, image, bbox, raw_dets=None):
        if image is None or bbox is None:
            return
        self.raw_detections = raw_dets if raw_dets is not None else []
        print(f"-------------frame {self.frame_idx}---------------")
        t1 = perf_counter()
        imgenc_config_dict = {"max_side_length": 1024, "use_square_sizing": True}
        encoded_img, token_hw, preencode_img_hw = self.sammodel.encode_image(image, **imgenc_config_dict)
        t2 = perf_counter()
        print(f"Processed image encoding time: {round(1000 * (t2 - t1))} ms")

        t1 = perf_counter()
        self.prompt_embedding(image, self.frame_idx, encoded_img, bbox)
        t2 = perf_counter()
        print(f"Processed image prompt embedding time: {round(1000 * (t2 - t1))} ms")

        t1 = perf_counter()
        self.tracking(image, self.frame_idx, encoded_img)   # 상태 업데이트 (+ 병합/가려짐)
        t2 = perf_counter()
        print(f"Processed image tracking time: {round(1000 * (t2 - t1))} ms")

        t1 = perf_counter()
        self.track_lifecycle()
        t2 = perf_counter()
        print(f"Processed image track lifecycle time: {round(1000 * (t2 - t1))} ms")

        # 마지막에 가시화
        self.visualize_tracks(image)

        self.frame_idx += 1

    def save_results(self):
        self.out_tracked.release()
        cv2.destroyAllWindows()
        with open("../ulsan_day1_어선군.txt", "w") as f:
            for line in self.mot_log:
                f.write(line + "\n")


def main():
    tracker = SAM2TRACKER()
    yolo = YOLO('/home/anhong/ws_radar_tracking/src/tracker/yolo_weight/best.pt')
    cap = cv2.VideoCapture('/home/anhong/ws_radar_tracking/data/scenario/ulsan_day1_어선군.avi')
    frame_idx = 0
    image_w = 1024
    image_h = 1024

    while True:
        ret, frame = cap.read()
        if not ret:
            print('no video!!')
            break
        frame_idx += 1

        bbox = []
        raw_dets = []
        results = yolo(frame, verbose=False)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            for box, conf, class_id in zip(boxes, confs, class_ids):
                rx1, ry1, rx2, ry2 = box
                raw_dets.append((float(rx1), float(ry1), float(rx2), float(ry2), float(conf)))
                if conf > CONF:
                    x1, y1, x2, y2 = box
                    obj = [x1, y1, x2, y2]
                    bbox.extend(obj)

        bboxes = [((bbox[i] / image_w, bbox[i+1] / image_h),
                   (bbox[i+2] / image_w, bbox[i+3] / image_h))
                  for i in range(0, len(bbox), 4)]

        tracker.process_segmentation(frame, bboxes, raw_dets=raw_dets)

    tracker.save_results()


if __name__ == '__main__':
    main()
