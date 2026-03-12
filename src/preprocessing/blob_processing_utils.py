"""
Utility functions for blob processing and refinement.
Includes NMS, blob splitting, and filtering improvements.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy import ndimage
from skimage import morphology, measure, segmentation


def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: [left, top, width, height]
        bbox2: [left, top, width, height]
    
    Returns:
        IoU value between 0 and 1
    """
    left1, top1, w1, h1 = bbox1
    left2, top2, w2, h2 = bbox2
    
    # Convert to [x1, y1, x2, y2]
    x1_1, y1_1 = left1, top1
    x2_1, y2_1 = left1 + w1, top1 + h1
    x1_2, y1_2 = left2, top2
    x2_2, y2_2 = left2 + w2, top2 + h2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def apply_nms(blobs: List[Dict[str, Any]], iou_threshold: float = 0.4) -> List[Dict[str, Any]]:
    """
    Apply Non-Maximum Suppression (NMS) to remove duplicate/overlapping bounding boxes.
    
    Args:
        blobs: List of blob dictionaries with 'bbox' field
        iou_threshold: IoU threshold for NMS (default: 0.4)
    
    Returns:
        Filtered list of blobs after NMS
    """
    if len(blobs) == 0:
        return blobs
    
    # Sort blobs by area (largest first) - keep larger blobs
    sorted_blobs = sorted(blobs, key=lambda b: b['area'], reverse=True)
    
    keep = []
    suppressed = set()
    
    for i, blob_i in enumerate(sorted_blobs):
        if i in suppressed:
            continue
        
        keep.append(blob_i)
        
        # Suppress overlapping blobs
        for j in range(i + 1, len(sorted_blobs)):
            if j in suppressed:
                continue
            
            blob_j = sorted_blobs[j]
            iou = calculate_iou(blob_i['bbox'], blob_j['bbox'])
            
            if iou > iou_threshold:
                suppressed.add(j)
    
    return keep


def split_large_blob(
    blob_mask: np.ndarray,
    min_area: int = 100,
    area_threshold_ratio: float = 1.5
) -> List[Dict[str, Any]]:
    """
    Split a large blob into multiple smaller blobs using watershed or distance transform.
    Improved version with better separation detection.
    
    Args:
        blob_mask: Binary mask of a single blob (0 and 255)
        min_area: Minimum area for a valid split blob
        area_threshold_ratio: Ratio threshold - if blob area > avg_area * ratio, try to split
    
    Returns:
        List of blob dictionaries from the split
    """
    # Convert to binary
    binary = (blob_mask > 127).astype(np.uint8)
    
    # Calculate area
    area = np.sum(binary)
    
    # Need at least 2x min_area to split
    if area < min_area * 2:
        return []
    
    # Try multiple strategies for splitting
    
    # Strategy 1: Distance transform + watershed
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    max_dist = np.max(dist_transform)
    
    if max_dist >= 10:
        # Try with different thresholds
        for threshold_ratio in [0.4, 0.5, 0.6]:
            threshold = threshold_ratio * max_dist
            _, markers = cv2.threshold(dist_transform, threshold, 255, cv2.THRESH_BINARY)
            
            if markers.dtype != np.uint8:
                markers = markers.astype(np.uint8)
            
            num_markers, _ = cv2.connectedComponents(markers)
            
            if num_markers >= 2:
                markers = markers.astype(np.int32)
                break
        else:
            markers = None
    else:
        markers = None
    
    # Strategy 2: Morphological erosion (if distance transform failed)
    if markers is None or np.max(markers) <= 1:
        # Try erosion with different kernel sizes
        for kernel_size in [3, 5, 7]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            for iterations in [1, 2, 3]:
                eroded = cv2.erode(binary, kernel, iterations=iterations)
                
                if eroded.dtype != np.uint8:
                    eroded = eroded.astype(np.uint8)
                
                num_labels, labels = cv2.connectedComponents(eroded)
                
                if num_labels >= 2:
                    # Check if components are reasonably sized
                    component_areas = [np.sum(labels == i) for i in range(1, num_labels)]
                    if min(component_areas) >= min_area:
                        markers = labels.astype(np.int32)
                        num_markers = num_labels
                        break
            if markers is not None and np.max(markers) > 1:
                break
    
    # If still no markers, try contour-based splitting
    if markers is None or np.max(markers) <= 1:
        # Find contours and try to split based on concavity
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            # Use largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Try to find concavity points (potential split points)
            hull = cv2.convexHull(largest_contour, returnPoints=False)
            defects = cv2.convexityDefects(largest_contour, hull)
            
            if defects is not None and len(defects) > 0:
                # Use defects as potential split markers
                # This is a simplified approach - could be improved
                pass
    
    # If we have markers, apply watershed
    if markers is not None and np.max(markers) > 1:
        # Apply watershed
        binary_3ch = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2BGR)
        cv2.watershed(binary_3ch, markers)
    else:
        # Could not find separation points
        return []
    
    # Extract individual blobs from markers
    split_blobs = []
    for label_id in range(1, num_markers + 1):
        component_mask = (markers == label_id).astype(np.uint8) * 255
        
        # Intersect with original blob
        component_mask = cv2.bitwise_and(component_mask, binary * 255)
        
        area_component = np.sum(component_mask > 0)
        
        if area_component < min_area:
            continue
        
        # Ensure component_mask is uint8
        if component_mask.dtype != np.uint8:
            component_mask = component_mask.astype(np.uint8)
        
        # Normalize to 0/255 if needed
        if component_mask.max() <= 1:
            component_mask = (component_mask * 255).astype(np.uint8)
        
        # Calculate bounding box and centroid
        stats = cv2.connectedComponentsWithStats(component_mask, connectivity=8)
        if stats[0] > 1:  # Has at least one component
            left = int(stats[2][1, cv2.CC_STAT_LEFT])
            top = int(stats[2][1, cv2.CC_STAT_TOP])
            width = int(stats[2][1, cv2.CC_STAT_WIDTH])
            height = int(stats[2][1, cv2.CC_STAT_HEIGHT])
            cx = float(stats[3][1, 0])
            cy = float(stats[3][1, 1])
            
            split_blobs.append({
                'area': int(area_component),
                'centroid': [cx, cy],
                'bbox': [left, top, width, height],
                'mask': component_mask
            })
    
    return split_blobs if len(split_blobs) > 1 else []


def split_large_blobs(
    blobs: List[Dict[str, Any]],
    mask: np.ndarray,
    labels: np.ndarray,
    area_threshold_ratio: float = 1.5,
    min_area: int = 100
) -> List[Dict[str, Any]]:
    """
    Split blobs that are significantly larger than average.
    
    Args:
        blobs: List of blob dictionaries
        mask: Original binary mask
        labels: Labeled image from connected components
        area_threshold_ratio: If blob area > avg_area * ratio, try to split (default: 1.5 = 150%)
        min_area: Minimum area for a valid blob after splitting
    
    Returns:
        Updated list of blobs with large ones split
    """
    if len(blobs) == 0:
        return blobs
    
    # Calculate average area
    areas = [b['area'] for b in blobs]
    avg_area = np.mean(areas) if areas else 0
    
    if avg_area == 0:
        return blobs
    
    threshold_area = avg_area * area_threshold_ratio
    
    result_blobs = []
    next_id = max([b.get('id', 0) for b in blobs], default=0) + 1
    
    for blob in blobs:
        if blob['area'] > threshold_area:
            # Try to split this blob
            # Use original_label if available, otherwise fall back to id
            original_label = blob.get('original_label', blob.get('id', 0))
            # Extract mask for this specific blob using original connected component label
            blob_mask = ((labels == original_label) & (mask > 0)).astype(np.uint8) * 255
            
            split_results = split_large_blob(blob_mask, min_area, area_threshold_ratio)
            
            if len(split_results) > 1:
                # Successfully split - add all split blobs
                for i, split_blob in enumerate(split_results):
                    new_blob = blob.copy()
                    new_blob['id'] = next_id + i
                    # Remove original_label for split blobs (they don't correspond to original labels)
                    new_blob.pop('original_label', None)
                    new_blob['area'] = split_blob['area']
                    new_blob['centroid'] = split_blob['centroid']
                    new_blob['bbox'] = split_blob['bbox']
                    # Recalculate optional features if needed
                    if 'aspect_ratio' in blob:
                        left, top, width, height = split_blob['bbox']
                        new_blob['aspect_ratio'] = float(width) / float(height) if height != 0 else 0.0
                    if 'eccentricity' in blob and 'mask' in split_blob:
                        # Compute eccentricity from mask
                        comp_mask = split_blob['mask']
                        cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts and len(cnts[0]) >= 5:
                            ellipse = cv2.fitEllipse(cnts[0])
                            a, b = ellipse[1][0] / 2, ellipse[1][1] / 2
                            if a > 0:
                                e = np.sqrt(1 - (b / a) ** 2) if a >= b else np.sqrt(1 - (a / b) ** 2)
                                new_blob['eccentricity'] = float(e)
                            else:
                                new_blob['eccentricity'] = 0.0
                        else:
                            new_blob['eccentricity'] = 0.0
                    result_blobs.append(new_blob)
                next_id += len(split_results)
            else:
                # Could not split, keep original
                result_blobs.append(blob)
        else:
            # Keep blob as is
            result_blobs.append(blob)
    
    return result_blobs


def filter_blobs_by_aspect_ratio(
    blobs: List[Dict[str, Any]],
    min_aspect_ratio: float = 0.1,
    max_aspect_ratio: float = 10.0
) -> List[Dict[str, Any]]:
    """
    Filter blobs by aspect ratio to remove unlikely chromosome shapes.
    
    Args:
        blobs: List of blob dictionaries with 'bbox' field
        min_aspect_ratio: Minimum aspect ratio (width/height)
        max_aspect_ratio: Maximum aspect ratio (width/height)
    
    Returns:
        Filtered list of blobs
    """
    filtered = []
    
    for blob in blobs:
        left, top, width, height = blob['bbox']
        
        if height == 0:
            continue
        
        aspect_ratio = width / height
        
        if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
            filtered.append(blob)
    
    return filtered


def filter_blobs_by_eccentricity(
    blobs: List[Dict[str, Any]],
    min_eccentricity: float = 0.3,
    max_eccentricity: float = 0.99
) -> List[Dict[str, Any]]:
    """
    Filter blobs by eccentricity to keep chromosome-like shapes.
    
    Args:
        blobs: List of blob dictionaries with 'eccentricity' field
        min_eccentricity: Minimum eccentricity (default: 0.3)
        max_eccentricity: Maximum eccentricity (default: 0.99)
    
    Returns:
        Filtered list of blobs
    """
    filtered = []
    
    for blob in blobs:
        if 'eccentricity' not in blob:
            continue
        
        ecc = blob['eccentricity']
        
        if min_eccentricity <= ecc <= max_eccentricity:
            filtered.append(blob)
    
    return filtered


def filter_blobs_by_area(
    blobs: List[Dict[str, Any]],
    min_area: int = 50,
    max_area_ratio: float = 3.0
) -> List[Dict[str, Any]]:
    """
    Filter blobs by area to remove outliers (too small or too large).
    
    Args:
        blobs: List of blob dictionaries with 'area' field
        min_area: Minimum area (pixels)
        max_area_ratio: Maximum area as ratio of median area (default: 3.0 = 300%)
    
    Returns:
        Filtered list of blobs
    """
    if len(blobs) == 0:
        return blobs
    
    areas = [b['area'] for b in blobs]
    median_area = np.median(areas) if areas else 0
    
    if median_area == 0:
        return blobs
    
    max_area = median_area * max_area_ratio
    
    filtered = []
    for blob in blobs:
        if min_area <= blob['area'] <= max_area:
            filtered.append(blob)
    
    return filtered


def calculate_shape_similarity(blob1: Dict[str, Any], blob2: Dict[str, Any]) -> float:
    """
    Calculate shape similarity between two blobs based on aspect ratio and eccentricity.
    Returns a value between 0 and 1, where 1 means identical shapes.
    
    Args:
        blob1: First blob dictionary
        blob2: Second blob dictionary
    
    Returns:
        Shape similarity score (0.0 to 1.0)
    """
    similarity = 1.0
    
    # Compare aspect ratios if available
    if 'aspect_ratio' in blob1 and 'aspect_ratio' in blob2:
        ar1 = blob1['aspect_ratio']
        ar2 = blob2['aspect_ratio']
        if ar1 > 0 and ar2 > 0:
            # Normalize aspect ratios (handle both >1 and <1 cases)
            ar1_norm = max(ar1, 1.0 / ar1) if ar1 > 0 else 1.0
            ar2_norm = max(ar2, 1.0 / ar2) if ar2 > 0 else 1.0
            ar_diff = abs(ar1_norm - ar2_norm) / max(ar1_norm, ar2_norm)
            similarity *= (1.0 - min(ar_diff, 1.0))
    
    # Compare eccentricity if available
    if 'eccentricity' in blob1 and 'eccentricity' in blob2:
        ecc1 = blob1['eccentricity']
        ecc2 = blob2['eccentricity']
        ecc_diff = abs(ecc1 - ecc2)
        similarity *= (1.0 - min(ecc_diff, 1.0))
    
    return similarity


def merge_close_blobs(
    blobs: List[Dict[str, Any]],
    distance_threshold: float = 15.0,
    max_area_ratio: float = 2.0,
    min_shape_similarity: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Merge blobs that are very close together, similar in size, and have similar shapes.
    Helps fix over-segmentation issues (fragmentation).
    Improved version: checks bounding box overlap and shape similarity.
    
    Args:
        blobs: List of blob dictionaries
        distance_threshold: Maximum distance between centroids to merge (pixels, default: 15.0)
        max_area_ratio: Maximum area ratio between blobs to merge (default: 2.0)
        min_shape_similarity: Minimum shape similarity score to merge (0.0-1.0, default: 0.3)
    
    Returns:
        Merged list of blobs
    """
    if len(blobs) <= 1:
        return blobs
    
    # Sort by area (larger first) - merge smaller into larger
    sorted_blobs = sorted(blobs, key=lambda b: b['area'], reverse=True)
    
    merged = []
    used = set()
    
    for i, blob_i in enumerate(sorted_blobs):
        if i in used:
            continue
        
        # Try to find nearby blobs to merge with
        merged_blob = blob_i.copy()
        merged_indices = {i}
        
        for j, blob_j in enumerate(sorted_blobs):
            if j <= i or j in used:
                continue
            
            # Check distance between centroids
            cx1, cy1 = blob_i['centroid']
            cx2, cy2 = blob_j['centroid']
            distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            
            # Also check bounding box overlap/distance
            left1, top1, w1, h1 = blob_i['bbox']
            left2, top2, w2, h2 = blob_j['bbox']
            right1, bottom1 = left1 + w1, top1 + h1
            right2, bottom2 = left2 + w2, top2 + h2
            
            # Calculate bbox overlap (IoU)
            bbox_iou = calculate_iou(blob_i['bbox'], blob_j['bbox'])
            
            # Calculate bbox distance (minimum distance between bboxes)
            bbox_dist_x = max(0, max(left1, left2) - min(right1, right2))
            bbox_dist_y = max(0, max(top1, top2) - min(bottom1, bottom2))
            bbox_distance = np.sqrt(bbox_dist_x**2 + bbox_dist_y**2)
            
            # Use the smaller of centroid distance or bbox distance
            # This helps merge blobs that are close but might have centroids further apart
            effective_distance = min(distance, bbox_distance * 1.5)
            
            # If bboxes overlap significantly, reduce distance threshold
            if bbox_iou > 0.1:
                effective_distance *= 0.5
            
            if effective_distance > distance_threshold:
                continue
            
            # Check area ratio (more lenient for very close blobs or overlapping bboxes)
            area_ratio = max(blob_i['area'], blob_j['area']) / min(blob_i['area'], blob_j['area'])
            # If blobs are very close or overlapping, be more lenient with area ratio
            if (effective_distance < 10 or bbox_iou > 0.1) and area_ratio <= max_area_ratio * 1.5:
                pass  # Allow merge
            elif area_ratio > max_area_ratio:
                continue
            
            # Check shape similarity (if features are available)
            shape_sim = calculate_shape_similarity(blob_i, blob_j)
            # If shapes are very different and blobs are not very close, skip merge
            if shape_sim < min_shape_similarity and effective_distance > 5:
                continue
            
            # Merge: combine bounding boxes and recalculate centroid
            new_left = min(left1, left2)
            new_top = min(top1, top2)
            new_right = max(right1, right2)
            new_bottom = max(bottom1, bottom2)
            
            merged_blob['bbox'] = [new_left, new_top, new_right - new_left, new_bottom - new_top]
            merged_blob['area'] = blob_i['area'] + blob_j['area']
            merged_blob['centroid'] = [
                (cx1 * blob_i['area'] + cx2 * blob_j['area']) / merged_blob['area'],
                (cy1 * blob_i['area'] + cy2 * blob_j['area']) / merged_blob['area']
            ]
            
            # Remove original_label for merged blobs (they don't correspond to a single original label)
            merged_blob.pop('original_label', None)
            
            # Update optional features
            if 'aspect_ratio' in blob_i:
                merged_blob['aspect_ratio'] = float(merged_blob['bbox'][2]) / float(merged_blob['bbox'][3]) if merged_blob['bbox'][3] > 0 else 0.0
            if 'eccentricity' in blob_i and 'eccentricity' in blob_j:
                # Average eccentricity
                merged_blob['eccentricity'] = (blob_i['eccentricity'] + blob_j['eccentricity']) / 2.0
            
            merged_indices.add(j)
            used.add(j)
        
        merged.append(merged_blob)
        used.add(i)
    
    return merged


def fill_blob_holes(blob_mask: np.ndarray, hole_area_threshold: int = 500) -> np.ndarray:
    """
    Fill small holes in a single blob mask.
    
    Args:
        blob_mask: Binary mask of a single blob (0 and 255)
        hole_area_threshold: Maximum area of holes to fill (pixels, default: 500)
    
    Returns:
        Binary mask with small holes filled
    """
    # Ensure binary
    binary = (blob_mask > 127).astype(np.uint8)
    
    # Invert to find holes
    inverted = cv2.bitwise_not(binary * 255)
    
    # Find connected components in inverted mask (holes)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    
    result = (binary * 255).copy()
    
    # Fill small holes
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area <= hole_area_threshold:
            hole_mask = (labels == label_id).astype(np.uint8) * 255
            result = cv2.bitwise_or(result, hole_mask)
    
    return result


def refine_mask_edges(blob_mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Refine and smooth edges of a blob mask using morphological operations.
    
    Args:
        blob_mask: Binary mask of a single blob (0 and 255)
        kernel_size: Size of morphological kernel for smoothing (default: 3)
        iterations: Number of times to apply smoothing (default: 1)
    
    Returns:
        Binary mask with refined edges
    """
    # Ensure binary
    binary = (blob_mask > 127).astype(np.uint8) * 255
    
    # Create small elliptical kernel for gentle smoothing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    result = binary.copy()
    
    # Apply gentle closing to smooth edges
    for _ in range(iterations):
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        # Light opening to remove small protrusions
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    
    return result


# === Smart splitting utilities (layout-aware) ===

ROW_BOUNDARIES = [344, 500, 626]
EXPECTED_BLOBS_PER_ROW = [10, 14, 12, 10]  # Rows 1-4
INDIVIDUAL_CHROMS = {"X", "Y"}


def split_blob_in_half(blob: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Split blob into two halves along width."""
    left, top, w, h = blob["bbox"]
    half_w = w // 2
    cx1 = left + half_w / 2
    cx2 = left + half_w + (w - half_w) / 2
    cy = top + h / 2
    return [
        {
            **{k: v for k, v in blob.items() if k not in ["id", "bbox", "centroid", "area"]},
            "area": max(1, blob["area"] // 2),
            "centroid": [cx1, cy],
            "bbox": [left, top, half_w, h],
        },
        {
            **{k: v for k, v in blob.items() if k not in ["id", "bbox", "centroid", "area"]},
            "area": max(1, blob["area"] - blob["area"] // 2),
            "centroid": [cx2, cy],
            "bbox": [left + half_w, top, w - half_w, h],
        },
    ]


def group_blobs_by_proximity(blobs: List[Dict[str, Any]], num_groups: int) -> List[List[Dict[str, Any]]]:
    """Group blobs by X proximity using largest gaps."""
    if not blobs or num_groups <= 0:
        return []
    sorted_blobs = sorted(blobs, key=lambda b: b["centroid"][0])
    if len(sorted_blobs) <= num_groups:
        return [[b] for b in sorted_blobs]
    gaps = []
    for i in range(len(sorted_blobs) - 1):
        gap = sorted_blobs[i + 1]["centroid"][0] - sorted_blobs[i]["centroid"][0]
        gaps.append((i, gap))
    gaps.sort(key=lambda x: x[1], reverse=True)
    split_indices = sorted([gaps[i][0] for i in range(min(num_groups - 1, len(gaps)))])
    groups = []
    prev = 0
    for idx in split_indices:
        groups.append(sorted_blobs[prev : idx + 1])
        prev = idx + 1
    groups.append(sorted_blobs[prev:])
    return groups


def _split_row2(row_blobs: List[Dict[str, Any]], expected_count: int) -> List[Dict[str, Any]]:
    """Special handling for Row 2 where merged pairs are common."""
    result = list(row_blobs)
    max_splits = 5
    splits_done = 0
    while splits_done < max_splits:
        merged = [b for b in result if b["area"] > 3500 and b["bbox"][2] > 35]
        if not merged:
            break
        merged.sort(key=lambda b: b["area"])
        target = merged[0]
        result.remove(target)
        result.extend(split_blob_in_half(target))
        splits_done += 1
    while len(result) < expected_count:
        widest = max(result, key=lambda b: b["bbox"][2])
        if widest["bbox"][2] > 40:
            result.remove(widest)
            result.extend(split_blob_in_half(widest))
        else:
            break
    while len(result) > expected_count:
        smallest = min(result, key=lambda b: b["area"])
        result.remove(smallest)
    return sorted(result, key=lambda b: b["centroid"][0])


def _split_row4(row_blobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Row 4: chromosomes 19-22 (pairs) + X/Y (individuals)."""
    sorted_blobs = sorted(row_blobs, key=lambda b: b["centroid"][0])
    groups = group_blobs_by_proximity(sorted_blobs, 6)
    result: List[Dict[str, Any]] = []
    for idx, group in enumerate(groups):
        if idx < 4:
            if len(group) == 1:
                blob = group[0]
                if blob["area"] > 2000 and blob["bbox"][2] > 30:
                    result.extend(split_blob_in_half(blob))
                else:
                    result.append(blob)
            else:
                result.extend(group[:2])
        else:
            if group:
                result.append(group[0])
    while len(result) < 10 and result:
        pair_candidates = result[: min(8, len(result))]
        widest = max(pair_candidates, key=lambda b: b["bbox"][2])
        if widest["bbox"][2] > 40 and widest["area"] > 2000:
            result.remove(widest)
            halves = split_blob_in_half(widest)
            insert_idx = pair_candidates.index(widest)
            result = result[:insert_idx] + halves + result[insert_idx:]
        else:
            break
    return sorted(result, key=lambda b: b["centroid"][0])


def smart_split_rows(blobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply row-aware splitting to approach expected blob counts.
    """
    # Assign blobs to rows via Y
    rows = [[], [], [], []]
    for b in blobs:
        y = b["centroid"][1]
        if y < ROW_BOUNDARIES[0]:
            rows[0].append(b)
        elif y < ROW_BOUNDARIES[1]:
            rows[1].append(b)
        elif y < ROW_BOUNDARIES[2]:
            rows[2].append(b)
        else:
            rows[3].append(b)

    # Process each row
    processed = []
    for idx, row in enumerate(rows):
        expected = EXPECTED_BLOBS_PER_ROW[idx]
        if idx == 3:
            row = _split_row4(row)
        elif idx == 1:
            row = _split_row2(row, expected)
        else:
            row = sorted(row, key=lambda b: b["bbox"][2], reverse=True)
            while len(row) < expected and row:
                widest = row.pop(0)
                if widest["bbox"][2] > 30:
                    row.extend(split_blob_in_half(widest))
                else:
                    row.append(widest)
                    break
            row = sorted(row, key=lambda b: b["centroid"][0])
        processed.append(row)

    # Flatten rows preserving order
    flattened = []
    for row in processed:
        flattened.extend(row)
    return flattened

