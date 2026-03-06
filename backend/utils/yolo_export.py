"""YOLO format export tool"""
import json
import shutil
import yaml
import random
from typing import List, Dict, Tuple
from pathlib import Path


class YOLOExporter:
    """YOLO format exporter"""
    
    @staticmethod
    def normalize_bbox(x_min: float, y_min: float, x_max: float, y_max: float,
                      img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Convert bounding box coordinates to YOLO format (normalized center point coordinates and width/height)
        
        Args:
            x_min, y_min, x_max, y_max: Bounding box absolute coordinates
            img_width, img_height: Image dimensions
            
        Returns:
            (center_x, center_y, width, height) normalized coordinates (0~1)
        """
        # Calculate absolute width and height
        box_w = x_max - x_min
        box_h = y_max - y_min
        
        # Calculate absolute center point
        center_x = x_min + (box_w / 2)
        center_y = y_min + (box_h / 2)
        
        # Normalize (keep 6 decimal places)
        yolo_x = round(center_x / img_width, 6)
        yolo_y = round(center_y / img_height, 6)
        yolo_w = round(box_w / img_width, 6)
        yolo_h = round(box_h / img_height, 6)
        
        return yolo_x, yolo_y, yolo_w, yolo_h
    
    @staticmethod
    def normalize_points(points: List[List[float]], img_width: int, img_height: int) -> List[float]:
        """
        Normalize polygon/keypoint coordinates
        
        Args:
            points: [[x, y], ...] or [[x, y, index], ...]
            img_width, img_height: Image dimensions
            
        Returns:
            One-dimensional array [x1, y1, x2, y2, ...] normalized coordinates
        """
        normalized = []
        for point in points:
            x = point[0] if isinstance(point, list) else point['x']
            y = point[1] if isinstance(point, list) else point['y']
            normalized.append(round(x / img_width, 6))
            normalized.append(round(y / img_height, 6))
        return normalized
    
    @staticmethod
    def export_annotation(annotation: Dict, class_id: int, img_width: int, img_height: int) -> str:
        """
        Export single annotation as YOLO format string
        
        Args:
            annotation: Annotation data dictionary
            class_id: Class ID
            img_width, img_height: Image dimensions
            
        Returns:
            YOLO format line: "class_id x y w h" or "class_id x1 y1 x2 y2 ..."
        """
        ann_type = annotation.get('type')
        data = annotation.get('data')
        
        if isinstance(data, str):
            data = json.loads(data)
        
        if ann_type == 'bbox':
            x_min = data['x_min']
            y_min = data['y_min']
            x_max = data['x_max']
            y_max = data['y_max']
            
            yolo_x, yolo_y, yolo_w, yolo_h = YOLOExporter.normalize_bbox(
                x_min, y_min, x_max, y_max, img_width, img_height
            )
            
            return f"{class_id} {yolo_x} {yolo_y} {yolo_w} {yolo_h}"
        
        elif ann_type in ['polygon', 'keypoint']:
            points = data.get('points', [])
            normalized_points = YOLOExporter.normalize_points(points, img_width, img_height)
            points_str = ' '.join(str(p) for p in normalized_points)
            
            return f"{class_id} {points_str}"
        
        else:
            raise ValueError(f"Unsupported annotation type: {ann_type}")
    
    @staticmethod
    def export_image(image_id: int, annotations: List[Dict], class_map: Dict[str, int],
                    img_width: int, img_height: int) -> List[str]:
        """
        Export all annotations for a single image
        
        Args:
            image_id: Image ID
            annotations: Annotation list
            class_map: Mapping from class name to class ID
            img_width, img_height: Image dimensions
            
        Returns:
            List of YOLO format lines
        """
        lines = []
        
        for ann in annotations:
            class_name = ann.get('class_name')
            class_id = class_map.get(class_name, -1)
            
            if class_id < 0:
                continue
            
            try:
                line = YOLOExporter.export_annotation(ann, class_id, img_width, img_height)
                lines.append(line)
            except Exception as e:
                print(f"Error exporting annotation {ann.get('id')}: {e}")
                continue
        
        return lines
    
    @staticmethod
    def export_project(project_data: Dict, output_dir: Path, datasets_root: Path):
        """
        Export entire project as YOLO format data (conforms to Ultralytics official format)
        
        Args:
            project_data: Project data dictionary containing images, annotations, classes
            output_dir: Output directory
            datasets_root: Dataset root directory for parsing image paths
        """
        # Clean old export directory to ensure clean directory structure
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Ultralytics standard directory structure
        images_train_dir = output_dir / "images" / "train"
        images_val_dir = output_dir / "images" / "val"
        labels_train_dir = output_dir / "labels" / "train"
        labels_val_dir = output_dir / "labels" / "val"
        
        images_train_dir.mkdir(parents=True, exist_ok=True)
        images_val_dir.mkdir(parents=True, exist_ok=True)
        labels_train_dir.mkdir(parents=True, exist_ok=True)
        labels_val_dir.mkdir(parents=True, exist_ok=True)
        
        # Build class mapping
        classes = project_data.get('classes', [])
        class_map = {cls['name']: idx for idx, cls in enumerate(classes)}
        class_names = [cls['name'] for cls in classes]
        
        # Get all valid images (images with existing files)
        images = project_data.get('images', [])
        valid_images = []
        
        for image in images:
            src_path = datasets_root / project_data['id'] / image['path']
            if src_path.exists():
                valid_images.append(image)
        
        # Determine split ratio: default 8:2, use 1:1 if image count is less than 10
        total_images = len(valid_images)
        if total_images < 10:
            train_ratio = 0.5  # 1:1
        else:
            train_ratio = 0.8  # 8:2
        
        # Calculate number of training and validation sets
        train_count = max(1, int(total_images * train_ratio))
        val_count = total_images - train_count
        
        # Randomly shuffle image order (using fixed seed for reproducibility)
        random.seed(42)  # Fixed seed to ensure consistent export results
        shuffled_images = valid_images.copy()
        random.shuffle(shuffled_images)
        
        # Split images
        train_images = shuffled_images[:train_count]
        val_images = shuffled_images[train_count:]
        
        # Export training set images and annotations
        train_copied = 0
        for image in train_images:
            # Use actual physical filename from path (not original filename)
            # This ensures the source and destination filenames match
            actual_filename = Path(image['path']).name
            img_stem = Path(actual_filename).stem
            img_width = image['width']
            img_height = image['height']

            # Copy image file using actual filename
            src_path = datasets_root / project_data['id'] / image['path']
            dst_path = images_train_dir / actual_filename
            shutil.copy2(src_path, dst_path)
            train_copied += 1

            # Export annotations
            annotations = image.get('annotations', [])
            if annotations:
                label_lines = YOLOExporter.export_image(
                    image['id'], annotations, class_map, img_width, img_height
                )

                if label_lines:
                    label_file = labels_train_dir / f"{img_stem}.txt"
                    with open(label_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(label_lines))
        
        # Export validation set images and annotations
        val_copied = 0
        for image in val_images:
            # Use actual physical filename from path (not original filename)
            # This ensures the source and destination filenames match
            actual_filename = Path(image['path']).name
            img_stem = Path(actual_filename).stem
            img_width = image['width']
            img_height = image['height']

            # Copy image file using actual filename
            src_path = datasets_root / project_data['id'] / image['path']
            dst_path = images_val_dir / actual_filename
            shutil.copy2(src_path, dst_path)
            val_copied += 1

            # Export annotations
            annotations = image.get('annotations', [])
            if annotations:
                label_lines = YOLOExporter.export_image(
                    image['id'], annotations, class_map, img_width, img_height
                )

                if label_lines:
                    label_file = labels_val_dir / f"{img_stem}.txt"
                    with open(label_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(label_lines))
        
        # Create data.yaml configuration file (Ultralytics standard format)
        data_yaml = {
            'path': str(output_dir.absolute()),  # Dataset root path
            'train': 'images/train',  # Training images relative path
            'val': 'images/val',  # Validation set path
            'nc': len(classes),  # Number of classes
            'names': class_names  # Class names list
        }
        
        yaml_file = output_dir / "data.yaml"
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        # Create class names file (compatible with old format)
        names_file = output_dir / "classes.txt"
        with open(names_file, 'w', encoding='utf-8') as f:
            for cls_name in class_names:
                f.write(f"{cls_name}\n")
        
        return {
            'images_count': total_images,
            'train_count': train_copied,
            'val_count': val_copied,
            'classes_count': len(classes)
        }

