import cv2
import os, traceback, datetime, config
from gif_maker import create_gif_from_profiles
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from scipy.spatial import Delaunay
from scipy.stats import zscore

def tprint(*args, **kwargs):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}]", *args, **kwargs)

def draw_contours(image_shape, contours):
    image = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.drawContours(image, contours, -1, 255, 2)
    return image

def save_image(output_dir, filename, image, force_save=False):
    if not config.EXPORT_INTERMEDIARY_IMAGES and not force_save:
        return
    if image is None or image.size == 0:
        tprint(f"Warning: Attempted to save empty image: {filename}")
        return
    try:
        cv2.imwrite(os.path.join(output_dir, filename), image)
    except Exception as e:
        tprint(f"Error saving image {filename}: {str(e)}")

def visualized_corners(contours, image_shape):
    height, width = image_shape[:2]
    image = np.zeros((height, width), dtype=np.uint8)
    
    cv2.drawContours(image, contours, -1, 255, 2)
    corners = cv2.goodFeaturesToTrack(image, 100, 0.01, 10)
    corners = np.int0(corners)
    print("len of corners", len(corners))
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image, (x, y), 10, 255, -1)
        print("x,y", x,y)

    return corners

def is_valid_contour(contour):
    return len(contour) >= 3 and cv2.contourArea(contour) > 0

def shape_factor(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    return (4 * np.pi * area) / (perimeter * perimeter) if perimeter else 0

def calculate_overlap(contour1, contour2, image_shape):
    mask1 = np.zeros(image_shape[:2], dtype=np.uint8)
    mask2 = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask1, [contour1], 0, 1, -1)
    cv2.drawContours(mask2, [contour2], 0, 1, -1)
    intersection = cv2.bitwise_and(mask1, mask2)
    union = cv2.bitwise_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union)

def make_more_circular(contour):
    if len(contour) < 5:
        return contour
    ellipse = cv2.fitEllipse(contour)
    (x, y), (MA, ma), angle = ellipse
    radius = int(max(MA, ma) / 2)
    return np.array([[[int(x + radius * np.cos(np.deg2rad(i))),
                       int(y + radius * np.sin(np.deg2rad(i)))]] for i in range(360)], dtype=np.int32)

def make_more_square(contour):
    if len(contour) < 4:
        return contour

    x, y, w, h = cv2.boundingRect(contour)
    side_length = max(w, h)  # Use the larger dimension to ensure the square encompasses the entire contour

    center_x = x + w // 2
    center_y = y + h // 2

    half_side = side_length // 2
    square_x1 = max(0, center_x - half_side)
    square_y1 = max(0, center_y - half_side)
    square_x2 = square_x1 + side_length
    square_y2 = square_y1 + side_length

    square_contour = np.array([
        [[square_x1, square_y1]],
        [[square_x2, square_y1]],
        [[square_x2, square_y2]],
        [[square_x1, square_y2]]
    ], dtype=np.int32)

    return square_contour

def make_more_shape(contours, shape_type):
    refined_contours = []
    for contour in contours:
        if shape_type == "circle":
            refined_contour = make_more_circular(contour)
        else:  # square
            refined_contour = make_more_square(contour)
        refined_contours.append(refined_contour)
    return refined_contours

def refined_cluster_contours(contours, image_shape, output_dir):
    total_area = image_shape[0] * image_shape[1]
    min_area = max(total_area * config.MIN_CONTOUR_RATIO, config.MIN_CONTOUR_AREA)
    filtered_contours = [c for c in contours if is_valid_contour(c) and cv2.contourArea(c) >= min_area]
    tprint("Num of Filtered Contours", len(filtered_contours))
    if len(filtered_contours) < 2:
        tprint("Not enough large contours for clustering.")
        return filtered_contours, None, None

    features = np.float32([[shape_factor(c), cv2.contourArea(c) / total_area] for c in filtered_contours])
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    n_clusters = min(4, len(filtered_contours))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, _ = cv2.kmeans(features, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    cluster_image = np.zeros(image_shape[:2] + (3,), dtype=np.uint8)
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(n_clusters)]

    for contour, label in zip(filtered_contours, labels.ravel()):
        color = colors[label]
        cv2.drawContours(cluster_image, [contour], 0, color, 2)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cv2.circle(cluster_image, (cX, cY), 5, color, -1)

    cluster_metrics = []
    for i in range(n_clusters):
        cluster_contours = [filtered_contours[j] for j, label in enumerate(labels.ravel()) if label == i]
        total_cluster_area = sum(cv2.contourArea(c) for c in cluster_contours)
        avg_shape_factor = np.mean([shape_factor(c) for c in cluster_contours])
        cluster_metrics.append((i, total_cluster_area, avg_shape_factor))

    best_cluster = sorted(cluster_metrics, key=lambda x: x[1], reverse=True)[0]
    shape_type = "circle" if abs(best_cluster[2] - 1) < abs(best_cluster[2] - config.SQR_OR_CIRC) else "square"  # Perimeter Ratio
    tprint(f"Shape type: {shape_type}", best_cluster[2])    
    best_cluster_contours = [filtered_contours[i] for i, label in enumerate(labels.ravel()) if label == best_cluster[0]]


    refined_contours = make_more_shape(best_cluster_contours, shape_type)

    return refined_contours, shape_type

def create_adaptive_template(contours, shape_type):
    if not contours:
        raise ValueError("No contours provided to create template.")

    median_area = np.median([cv2.contourArea(c) for c in contours])

    if shape_type == "circle":
        radius = int(np.sqrt(median_area / np.pi))
        center_x = int(np.median([c[:, 0, 0].mean() for c in contours]))
        center_y = int(np.median([c[:, 0, 1].mean() for c in contours]))
        return (center_x, center_y, radius)
    else:  # square
        side = int(np.sqrt(median_area))
        center_x = int(np.median([c[:, 0, 0].mean() for c in contours]))
        center_y = int(np.median([c[:, 0, 1].mean() for c in contours]))
        return (center_x - side // 2, center_y - side // 2, side, side)

def match_template_to_contours(contours, template, image_shape, shape_type):
    matched_contours = []
    excluded_areas = np.zeros(image_shape[:2], dtype=np.uint8)

    for contour in contours:
        if not is_valid_contour(contour):
            continue
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 1, -1)
        if cv2.countNonZero(cv2.bitwise_and(excluded_areas, mask)) == 0:
            matched_contours.append(contour)
            excluded_areas = cv2.bitwise_or(excluded_areas, mask)

    return matched_contours

import numpy as np
import cv2
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

def align_and_resize_contours(contours, image_shape, shape_type):
    median_area = np.median([cv2.contourArea(s) for s in contours])
    median_size = int(np.sqrt(median_area)) if shape_type == "square" else int(np.sqrt(median_area / np.pi))

    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))

    centroids = np.array(centroids)

    Z = linkage(pdist(centroids), method='ward')
    max_d = 0.1 * max(image_shape)  # Set a threshold to form flat clusters
    clusters = fcluster(Z, max_d, criterion='distance')

    vis_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for i, contour in enumerate(contours):
        color = colors[clusters[i] % len(colors)]
        cv2.drawContours(vis_image, [contour], -1, color, 2)


    cluster_centroids = np.array([centroids[clusters == i].mean(axis=0) for i in np.unique(clusters)])

    rows = int(np.sqrt(len(cluster_centroids)))
    cols = int(np.ceil(len(cluster_centroids) / rows))
    grid_x = np.linspace(min(cluster_centroids[:, 0]), max(cluster_centroids[:, 0]), cols)
    grid_y = np.linspace(min(cluster_centroids[:, 1]), max(cluster_centroids[:, 1]), rows)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    for centroid in centroids:
        cv2.circle(vis_image, (int(centroid[0]), int(centroid[1])), 5, (255, 0, 0), -1)

    for point in grid_points:
        cv2.circle(vis_image, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

    adjusted_centroids = []
    for centroid in centroids:
        if len(grid_points) == 0:
            adjusted_centroids.append(centroid)
            continue
        distances = np.linalg.norm(grid_points - centroid, axis=1)
        nearest_index = np.argmin(distances)
        nearest_point = grid_points[nearest_index]
        if np.linalg.norm(nearest_point - centroid) <= 0.1 * median_size:
            adjusted_centroids.append(nearest_point)
        else:
            adjusted_centroids.append(centroid)
        grid_points = np.delete(grid_points, nearest_index, axis=0)  # Remove used grid point

    adjusted_centroids = np.array(adjusted_centroids)

    for centroid in adjusted_centroids:
        cv2.circle(vis_image, (int(centroid[0]), int(centroid[1])), 5, (0, 0, 255), -1)

    resized_contours = []
    for contour, centroid in zip(contours, adjusted_centroids):
        if shape_type == "square":
            new_contour = np.array([
                [int(centroid[0] - median_size // 2), int(centroid[1] - median_size // 2)],
                [int(centroid[0] + median_size // 2), int(centroid[1] - median_size // 2)],
                [int(centroid[0] + median_size // 2), int(centroid[1] + median_size // 2)],
                [int(centroid[0] - median_size // 2), int(centroid[1] + median_size // 2)]
            ], dtype=np.int32)
        else:  # circle
            new_contour = cv2.ellipse2Poly((int(centroid[0]), int(centroid[1])), (median_size, median_size), 0, 0, 360, 1)
            new_contour = new_contour.reshape((-1, 1, 2))

        resized_contours.append(new_contour)

    return resized_contours

def filter_shapes_by_size(shapes, template, image_shape, shape_type):
    template_area = np.pi * template[2]**2 if shape_type == "circle" else template[2] * template[3]
    filtered_shapes = []

    shapes_with_area = [(s, cv2.contourArea(s)) for s in shapes]
    shapes_with_area.sort(key=lambda x: abs(x[1] - template_area))

    excluded_areas = np.zeros(image_shape[:2], dtype=np.uint8)

    for shape, area in shapes_with_area:
        if 0.5 * template_area <= area <= 1.5 * template_area:
            x, y, w, h = cv2.boundingRect(shape)
            aspect_ratio = float(w) / h if h != 0 else 0
            if 0.7 <= aspect_ratio <= 1.3:  # Tolerance from perfect square
                mask = np.zeros(image_shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [shape], 0, 1, -1)
                if cv2.countNonZero(cv2.bitwise_and(excluded_areas, mask)) == 0:
                    filtered_shapes.append(shape)
                    excluded_areas = cv2.bitwise_or(excluded_areas, mask)

    tprint(f"Number of shapes after filtering: {len(filtered_shapes)}")

    resized_shapes = align_and_resize_contours(filtered_shapes, image_shape, shape_type)

    return resized_shapes

def get_shape_size(contour, shape_type):
    x, y, w, h = cv2.boundingRect(contour)
    if shape_type == "circle":
        return max(w, h)  # diameter
    else:  # square
        return max(w, h)  # side length

def template_guided_aggregation(filtered_shapes, contours, template, image_shape, shape_type, iteration=1):
    def contour_distance(c1, c2):
        return cv2.norm(np.mean(c1, axis=0) - np.mean(c2, axis=0))

    def merge_contours(contours_to_merge):
        combined = np.vstack([c.reshape(-1, 2) for c in contours_to_merge])
        return cv2.convexHull(combined)

    template_size = template[2] * 2 if shape_type == "circle" else template[2]
    
    tprint(f"Iteration {iteration}: Input contours: {len(contours)}")
    tprint(f"Iteration {iteration}: Template size: {template_size}")

    aggregated_contours = []
    remaining_contours = list(contours)  # Convert to list if it's not already

    while remaining_contours:
        base_contour = remaining_contours.pop(0)
        base_size = get_shape_size(base_contour, shape_type)
        
        if base_size >= template_size:
            aggregated_contours.append(base_contour)
            continue
        
        cluster = [base_contour]
        cluster_size = base_size
        
        for i in range(len(remaining_contours) - 1, -1, -1):
            if cluster_size >= template_size:
                break
            
            candidate = remaining_contours[i]
            candidate_size = get_shape_size(candidate, shape_type)
            
            distance_threshold = 0.05 * max(image_shape) if iteration == 1 else 0.1 * max(image_shape)
            if contour_distance(base_contour, candidate) < distance_threshold:
                if cluster_size + candidate_size <= 1.2 * template_size:
                    cluster.append(candidate)
                    cluster_size = get_shape_size(merge_contours(cluster), shape_type)
                    remaining_contours.pop(i)
        
        if cluster_size >= 0.5 * template_size:  # Adjust this threshold if needed
            merged = merge_contours(cluster)
            aggregated_contours.append(merged)

    final_contours = []
    for contour in aggregated_contours:
        size = get_shape_size(contour, shape_type)
        if size <= 1.2 * template_size:
            final_contours.append(contour)

    tprint(f"Iteration {iteration}: Output contours: {len(final_contours)}")
    return final_contours

def strict_combine_contours(filtered_shapes, aggregated_contours, template, tpl_shape_type, image_shape, output_dir):
    template_size = template[2] * 2 if tpl_shape_type == "circle" else template[2]
    
    def get_contour_centroid(contour):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None

    filtered_centroids = [get_contour_centroid(c) for c in filtered_shapes]
    
    non_overlapping_contours = []
    for contour in aggregated_contours:
        contour_centroid = get_contour_centroid(contour)
        if contour_centroid is None:
            continue
        
        overlaps = False
        for anchor, anchor_centroid in zip(filtered_shapes, filtered_centroids):
            if anchor_centroid is None:
                continue
            distance = np.linalg.norm(np.array(contour_centroid) - np.array(anchor_centroid))
            if distance < template_size:  # Only check overlap if within a reasonable distance
                if calculate_overlap(contour, anchor, image_shape) > 0:
                    overlaps = True
                    break
        
        if not overlaps:
            non_overlapping_contours.append(contour)

    
    combined_contours = filtered_shapes.copy()

    template_area = np.pi * (template[2] / 2) ** 2 if tpl_shape_type == "circle" else template[2] * template[3]

    for contour in non_overlapping_contours:
        if cv2.contourArea(contour) < 0.25 * template_area:
            continue
        
        overlaps = False
        for existing in combined_contours:
            existing_centroid = get_contour_centroid(existing)
            if existing_centroid is None:
                continue
            distance = np.linalg.norm(np.array(contour_centroid) - np.array(existing_centroid))
            if distance < template_size:  # Only check overlap if within a reasonable distance
                if calculate_overlap(contour, existing, image_shape) > 0:
                    overlaps = True
                    break
        
        if not overlaps:
            combined_contours.append(contour)

    tprint(f"Number of contours: anchors={len(filtered_shapes)}, aggregated={len(aggregated_contours)}, final={len(combined_contours)}")
    return combined_contours

def refine_contours(contours, image_shape):
    refined = []
    for contour in contours:
        if cv2.contourArea(contour) > image_shape[0] * image_shape[1] * 0.001:  # Adjust threshold as needed
            refined.append(contour)
    return refined

def process_contours(contours, image_shape, output_dir, original_image):
    tprint("Entering process_contours function")
    try:
        if isinstance(original_image, str):
            tprint(f"original_image is a file path. Attempting to load image from: {original_image}")
            original_image = cv2.imread(original_image)
            if original_image is None:
                raise ValueError(f"Failed to load image from {original_image}")

        tprint(f"Original image type: {type(original_image)}")
        tprint(f"Original image shape: {original_image.shape if hasattr(original_image, 'shape') else 'No shape attribute'}")

        if not isinstance(original_image, np.ndarray):
            raise ValueError(f"Invalid original_image type: {type(original_image)}")

        if len(original_image.shape) == 2:
            tprint("Converting grayscale image to 3-channel")
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        elif len(original_image.shape) == 3:
            if original_image.shape[2] == 4:
                tprint("Converting 4-channel image to 3-channel")
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGRA2BGR)
            elif original_image.shape[2] != 3:
                raise ValueError(f"Unexpected number of channels: {original_image.shape[2]}")
        else:
            raise ValueError(f"Unexpected image shape: {original_image.shape}")

        valid_contours, tpl_shape_type = refined_cluster_contours(contours, image_shape, output_dir)
        if not valid_contours:
            tprint("No valid contours found after refinement.")
            return [], None, None

        template = create_adaptive_template(valid_contours, tpl_shape_type)
        matched_contours = match_template_to_contours(contours, template, image_shape, tpl_shape_type)
        
        tprint(f"Number of matched contours: {len(matched_contours)}")

        processed_group = [make_more_circular(c) if tpl_shape_type == "circle" else make_more_square(c)
                        for c in matched_contours if is_valid_contour(c)]
        
        tprint(f"Number of processed contours: {len(processed_group)}")

        filtered_shapes = filter_shapes_by_size(processed_group, template, image_shape, tpl_shape_type)
        
        refined_contours = refine_contours(valid_contours, image_shape)
        
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, filtered_shapes, -1, 255, -1)
        
        inverted_mask = cv2.bitwise_not(mask)
        masked_image = cv2.bitwise_and(original_image, original_image, mask=inverted_mask)

        blurred = cv2.bilateralFilter(masked_image, 9, 75, 75)  # Edge-preserving blur
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        next_edges = cv2.Canny(enhanced, 20, 200)  # Aggressive thresholds

        kernel = np.ones((3,3), np.uint8)
        dilated_edges = cv2.dilate(next_edges, kernel, iterations=1)

        new_contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tprint(f"Number of new contours found: {len(new_contours)}")

        aggregated_contours = template_guided_aggregation(filtered_shapes, new_contours, template, image_shape, tpl_shape_type, iteration=1)
        tprint(f"Number of contours after first aggregation: {len(aggregated_contours)}")

        combined_contours = strict_combine_contours(filtered_shapes, aggregated_contours, template, tpl_shape_type, image_shape, output_dir)

        aligned_combined_contours = align_and_resize_contours(combined_contours, image_shape, tpl_shape_type)

        profile_regions = generate_profile_regions(aligned_combined_contours, image_shape)
        
        profiles = extract_profiles(original_image, profile_regions, output_dir)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

        return combined_contours, tpl_shape_type, profiles

    except Exception as e:
        tprint(f"Error in process_contours: {str(e)}")
        traceback.print_exc()
        return [], None, None

def generate_profile_regions(extracting_shapes, image_shape):
    profile_regions = []
    excluded_areas = np.zeros(image_shape[:2], dtype=np.uint8)

    for shape in extracting_shapes:
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [shape], 0, 1, -1)
        if cv2.countNonZero(cv2.bitwise_and(excluded_areas, mask)) == 0:
            x, y, w, h = cv2.boundingRect(shape)        #boundary touching shapes are excluded  
            profile_regions.append((int(x), int(y), int(w), int(h)))
            excluded_areas = cv2.bitwise_or(excluded_areas, mask)
    
    return profile_regions

def extract_profiles(image, regions, output_dir):
    profile_images = []
    for i, (x, y, w, h) in enumerate(regions):
        profile = image[y:y+h, x:x+w]
        if profile is not None and profile.size > 0:
            profile_images.append(profile)

    final_image = image.copy()
    for i, (x, y, w, h) in enumerate(regions):
        cv2.rectangle(final_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(final_image, f'{i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
    return profile_images

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0), 60, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return image, contours

def detect_and_extract_profiles(image_path, output_dir, gif_duration, url, include_qr):
    tprint(f"Processing image: {image_path}")
    image, contours = preprocess_image(image_path)
    tprint(f"Number of contours: {len(contours)}")
    
    if image is None or image.size == 0:
        tprint("Error: Failed to load or preprocess the image.")
        return [], 0, [], []

    profile_regions, shape_type, profiles = process_contours(contours, image.shape[:2], output_dir, image)
    
    if not profile_regions:
        tprint("No profile regions detected.")
        return [], len(contours), [], []

    gif_files = create_gif_from_profiles(profiles, output_dir, gif_duration, url, include_qr)

    return profile_regions, len(profiles), profiles, gif_files
